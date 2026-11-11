"""Two-tower retrieval source: audio-aware item tower with ID dropout, history-pooling user tower.

Trained on Listen+ events before a cutoff with sampled softmax (logQ correction, optional mined hard
negatives). Writes item_vecs.npy, user_vecs.npy, model.pt and config.json to --out; retrieval.py reads
the vectors. Runs after `make data` (and `make audio`, unless --no-audio).

  PYTHONPATH=src python -m yrec.twotower --cutoff A --out artifacts/tt_A --eval
  PYTHONPATH=src torchrun --nproc_per_node=4 -m yrec.twotower --cutoff A --out artifacts/tt_A
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from yrec.data import CUTOFFS, PROC, WINDOWS

D = 64  # embedding width of both towers


class TwoTower(nn.Module):
    def __init__(self, item_artist, audio, has_audio, seen, hist_len, id_dropout, tau, logq):
        super().__init__()
        n_items = len(item_artist)
        # per-item lookup tensors, plain attributes so they stay out of model.pt
        self.item_artist, self.audio, self.has_audio, self.seen, self.logq = item_artist, audio, has_audio, seen, logq
        self.p, self.tau = id_dropout, tau
        self.id = nn.Embedding(n_items, D)
        self.no_id = nn.Parameter(torch.zeros(D))
        self.artist = nn.Embedding(int(item_artist.max()) + 1, 32)
        nn.init.normal_(self.id.weight, std=0.1)
        nn.init.normal_(self.artist.weight, std=0.1)
        width = D + 32
        if audio is not None:
            self.audio_mlp = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, D))
            self.missing_audio = nn.Parameter(torch.zeros(D))
            width += D
        self.out = nn.Linear(width, D)
        self.q = nn.Parameter(torch.zeros(D))              # content attention
        self.pos_bias = nn.Parameter(torch.zeros(hist_len))  # learned recency weighting (last = most recent)
        self.mlp = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Linear(256, D))

    def item(self, idx):
        # train: ID dropout; eval: real ID for items seen before the cutoff, "no-ID" vector for cold items
        keep = torch.rand(idx.shape, device=idx.device) >= self.p if self.training else self.seen[idx]
        parts = [torch.where(keep[..., None], self.id(idx), self.no_id), self.artist(self.item_artist[idx])]
        if self.audio is not None:
            a = self.audio_mlp(self.audio[idx].float())
            parts.append(torch.where(self.has_audio[idx, None], a, self.missing_audio))
        return F.normalize(self.out(torch.cat(parts, -1)).float(), dim=-1)

    def user(self, hist, valid):
        return self.pool(self.item(hist), valid)

    def pool(self, h, valid):
        w = torch.softmax((h @ self.q + self.pos_bias).masked_fill(~valid, -1e4), 1)
        p = (w[..., None] * h).sum(1)
        return F.normalize((p + self.mlp(p)).float(), dim=-1)

    def forward(self, hist, valid, pos, hard=None):
        """Sampled softmax over in-batch positives (logQ-corrected) + optional mined hard negatives."""
        # one item-tower pass over the batch's unique items (~4x fewer rows than B * hist_len on real data)
        ids = torch.cat([hist.flatten(), pos] + ([] if hard is None else [hard.clamp(min=0).flatten()]))
        uniq, inv = torch.unique(ids, return_inverse=True)
        vec = self.item(uniq)[inv]
        B, H = hist.shape
        u, v = self.pool(vec[:B * H].view(B, H, D), valid), vec[B * H:B * H + B]
        logits = (u @ v.T).float() / self.tau - self.logq[pos][None]
        eye = torch.eye(len(pos), dtype=torch.bool, device=pos.device)
        logits = logits.masked_fill((pos[None] == pos[:, None]) & ~eye, float("-inf"))  # accidental hits
        if hard is not None:
            hl = (u[:, None] * vec[B * H + B:].view(hard.shape + (D,))).sum(-1) / self.tau
            logits = torch.cat([logits, hl.masked_fill((hard < 0) | (hard == pos[:, None]), float("-inf"))], 1)
        return F.cross_entropy(logits, torch.arange(len(pos), device=pos.device))


def history(ev_item, start, end, hist_len):
    """Last hist_len items in ev_item[start:end) per row, oldest first, left-padded (valid=False)."""
    p = end[:, None] - hist_len + torch.arange(hist_len, device=end.device)
    return ev_item[p.clamp(min=0)].long(), p >= start[:, None]


@torch.no_grad()
def item_vectors(net, chunk=1 << 16):
    net.eval()
    n = len(net.item_artist)
    dev = net.item_artist.device
    v = torch.cat([net.item(torch.arange(s, min(s + chunk, n), device=dev)) for s in range(0, n, chunk)])
    net.train()
    return v


@torch.no_grad()
def user_vectors(net, ev_item, off, hist_len, chunk=1024):
    """Each user's vector from their last hist_len training events; zero if none."""
    net.eval()
    out = []
    for s in range(0, len(off) - 1, chunk):
        start, end = off[s:s + chunk + 1][:-1], off[s + 1:s + chunk + 1]
        out.append(net.user(*history(ev_item, start, end, hist_len)) * (end > start)[:, None])
    net.train()
    return torch.cat(out)


@torch.no_grad()
def mine_hard(net, ev_item, off, hist_len, pair_keys, n_pool, skip=10, slack=256, chunk=256):
    """Per user: top-ranked items not in their training history, after skipping the first `skip`. -1 = none."""
    iv, uv = item_vectors(net), user_vectors(net, ev_item, off, hist_len)
    if iv.is_cuda:
        iv, uv = iv.half(), uv.half()
    k = min(skip + n_pool + slack, len(iv))
    cand = torch.cat([(uv[s:s + chunk] @ iv.T).topk(k, dim=1).indices for s in range(0, len(uv), chunk)])
    bad = torch.isin(torch.arange(len(uv), device=cand.device)[:, None] * len(iv) + cand, pair_keys)
    order = bad.to(torch.int8).argsort(dim=1, stable=True)  # history items to the back, rank order kept
    cand, bad = cand.gather(1, order)[:, skip:skip + n_pool], bad.gather(1, order)[:, skip:skip + n_pool]
    return cand.masked_fill(bad, -1)


def recall_at(user_vecs, item_vecs, win, k=100):
    import faiss
    index = faiss.IndexFlatIP(item_vecs.shape[1])
    index.add(item_vecs)
    tgt = win.groupby("uidx").item_idx.unique()
    _, top = index.search(user_vecs[tgt.index.values], k)
    return float(np.mean([len(set(t) & set(r)) / min(len(t), k) for t, r in zip(tgt.values, top)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", choices=list(CUTOFFS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=4096, help="per GPU")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--hist-len", type=int, default=100)
    ap.add_argument("--min-hist", type=int, default=3, help="min prior events for a training example")
    ap.add_argument("--id-dropout", type=float, default=0.3)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-hard-neg", action="store_true")
    ap.add_argument("--hn-every", type=int, default=1000, help="re-mine hard negatives every N steps")
    ap.add_argument("--hn-num", type=int, default=8, help="hard-negative columns per example")
    ap.add_argument("--hn-pool", type=int, default=64, help="mined negatives kept per user")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--eval", action="store_true", help="Recall@100 on the next window (cutoff A only)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ddp = "WORLD_SIZE" in os.environ
    cuda = torch.cuda.is_available()
    if ddp:
        dist.init_process_group("nccl" if cuda else "gloo")
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(dev)
    torch.manual_seed(args.seed)  # same init everywhere (DDP also broadcasts rank 0's weights)
    log = print if rank == 0 else (lambda *a, **k: None)
    t0 = time.time()

    # ---- data
    items = pd.read_parquet(PROC / "items.parquet", columns=["idx", "artist_idx", "has_audio"]).sort_values("idx")
    n_items, n_users = len(items), len(pd.read_parquet(PROC / "users.parquet", columns=["uidx"]))
    ev = pd.read_parquet(PROC / "lp_events.parquet", columns=["uidx", "item_idx", "ts"])
    cutoff = CUTOFFS[args.cutoff]
    tr = ev[ev.ts < cutoff]  # still sorted by (uidx, ts)
    u, it, ts = tr.uidx.to_numpy(), tr.item_idx.to_numpy(), tr.ts.to_numpy()
    n = len(tr)
    # history of event i = events of the same user strictly before ts_i, i.e. [user start, first event with this (uidx, ts))
    new = np.r_[True, (u[1:] != u[:-1]) | (ts[1:] != ts[:-1])]
    hist_end = np.maximum.accumulate(np.where(new, np.arange(n), 0)).astype(np.int32)
    off = np.searchsorted(u, np.arange(n_users + 1)).astype(np.int64)
    eligible = np.flatnonzero(hist_end - off[u] >= args.min_hist)
    counts = np.bincount(it, minlength=n_items)

    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    ev_u, ev_item, hist_end, off_t, eligible = T(u), T(it), T(hist_end), T(off), T(eligible)
    audio, has_audio = None, torch.zeros(n_items, dtype=torch.bool, device=dev)
    if not args.no_audio:
        if (PROC / "audio.npy").exists():
            audio = T(np.load(PROC / "audio.npy").astype(np.float16, copy=False))
            has_audio = T(items.has_audio.to_numpy())
        else:
            log("audio.npy missing: every item uses the missing-audio vector")
            audio = torch.zeros(1, 128, dtype=torch.float16, device=dev).expand(n_items, 128)
    logq = T(np.log(np.maximum(counts, 1) / n).astype(np.float32))
    seen = T(counts > 0)
    log(f"train events {n:,}  examples {len(eligible):,}  items seen {int(seen.sum()):,}/{n_items:,}  "
        f"users {n_users:,}  load {time.time() - t0:.0f}s")

    net = TwoTower(T(items.artist_idx.to_numpy()), audio, has_audio, seen,
                   args.hist_len, args.id_dropout, args.tau, logq).to(dev)
    model = net
    if ddp:
        model = nn.parallel.DistributedDataParallel(net, device_ids=[dev.index] if cuda else None)
        if cuda:  # the 934K x 64 ID table dominates the all-reduce; send it in fp16
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
            model.register_comm_hook(None, default_hooks.fp16_compress_hook)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, fused=cuda)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    pair_keys = None if args.no_hard_neg else torch.unique(ev_u.long() * n_items + ev_item)

    # ---- train
    gen = torch.Generator(dev).manual_seed(args.seed * 1000 + rank)
    pool, t_mine, run_loss, losses = None, 0.0, 0.0, []
    t_train = tl = time.time()
    for step in range(1, args.steps + 1):
        if pair_keys is not None and step % args.hn_every == 0:
            tm = time.time()
            pool = mine_hard(net, ev_item, off_t, args.hist_len, pair_keys, args.hn_pool)
            t_mine += time.time() - tm
            log(f"step {step}: mined hard negatives in {time.time() - tm:.1f}s, "
                f"{(pool >= 0).float().mean():.0%} slots filled")
        i = eligible[torch.randint(len(eligible), (args.batch,), device=dev, generator=gen)].long()
        uu = ev_u[i].long()
        hist, valid = history(ev_item, off_t[uu], hist_end[i].long(), args.hist_len)
        hard = None
        if pool is not None:
            hard = pool[uu].gather(1, torch.randint(args.hn_pool, (len(i), args.hn_num), device=dev, generator=gen))
        with torch.autocast(dev.type, dtype=torch.float16, enabled=cuda):
            loss = model(hist, valid, ev_item[i].long(), hard)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        run_loss += loss.detach()  # no .item() here: it would sync every step
        if step % args.log_every == 0:
            losses.append(run_loss.item() / args.log_every)
            run_loss, dt = 0.0, time.time() - tl
            log(f"step {step}  loss {losses[-1]:.4f}  {args.batch * world * args.log_every / dt:,.0f} ex/s  "
                f"elapsed {time.time() - t_train:.0f}s")
            tl = time.time()
    wall = time.time() - t_train
    if ddp:
        dist.destroy_process_group()
    if rank != 0:
        return

    # ---- export
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    iv = item_vectors(net).cpu().numpy().astype(np.float32)
    uv = user_vectors(net, ev_item, off_t, args.hist_len).cpu().numpy().astype(np.float32)
    np.save(out / "item_vecs.npy", iv)
    np.save(out / "user_vecs.npy", uv)
    torch.save(net.state_dict(), out / "model.pt")
    assert iv.shape == (n_items, D) and uv.shape == (n_users, D)
    assert np.allclose(np.linalg.norm(iv, axis=1), 1, atol=1e-3)
    un = np.linalg.norm(uv, axis=1)
    assert np.allclose(un[np.diff(off) > 0], 1, atol=1e-3) and np.all(un[np.diff(off) == 0] == 0)
    stats = {"steps": args.steps, "world_size": world, "final_loss": losses[-1] if losses else None,
             "wall_time_s": round(wall, 1), "mining_time_s": round(t_mine, 1),
             "examples_per_s": round(args.steps * args.batch * world / wall),
             "train_events": n, "train_examples": len(eligible)}
    if args.eval:
        if args.cutoff != "A":
            log("--eval skipped: only cutoff A has a non-test next window")
        else:
            lo, hi = WINDOWS["A"]
            stats["recall@100"] = recall_at(uv, iv, ev[(ev.ts >= lo) & (ev.ts < hi)])
            log(f"Recall@100 on window B: {stats['recall@100']:.4f}")
    (out / "config.json").write_text(json.dumps({**vars(args), **stats}, indent=2))
    log(json.dumps(stats))


if __name__ == "__main__":
    main()
