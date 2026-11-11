"""Stage 1: run every retrieval source, merge up to 500 candidates per user, evaluate each source and the union.

Vector sources (two-tower, ALS, audio) are searched with exact FAISS inner-product search.
Popularity is a fixed list and repeat is a lookup of the user's own history.

  python -m yrec.retrieval --cut A     # candidates for the ranker's training day
  python -m yrec.retrieval --cut AB    # candidates for the test day + stage-1 results
  python -m yrec.retrieval --eval-tt artifacts/tt_AB_nohn   # two-tower-only metrics (ablations)
"""
import argparse
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from yrec import metrics, sources
from yrec.data import PROC

VIEWS = ("all", "discovery")
QUOTA = {  # per-source share of the 500 candidates; repeats matter only when re-listens count
    "all": dict(tt=150, als=100, audio=50, pop=25, repeat=175),
    "discovery": dict(tt=200, als=150, audio=100, pop=50),
}
VECTOR_SOURCES = ("tt", "als", "audio")


def search(U, I, users, k, mask=None):
    """Exact top-k inner-product search of users' vectors over items (optionally a subset)."""
    ids = np.flatnonzero(mask) if mask is not None else np.arange(len(I))
    index = faiss.IndexFlatIP(I.shape[1])
    index.add(np.ascontiguousarray(I[ids], dtype=np.float32))
    _, nn = index.search(np.ascontiguousarray(U[users], dtype=np.float32), min(k, len(ids)))
    return ids[nn]


def exclusions(hist, view):
    """{uidx: items never to recommend}: disliked tracks always; the whole history in the discovery view."""
    h = hist[hist.disliked | (hist.n_listen > 0)] if view == "discovery" else hist[hist.disliked]
    return {u: g.to_numpy() for u, g in h.groupby("uidx").item_idx}


def filtered(lists, users, excl, n):
    out = {}
    for u, row in zip(users, lists):
        row = row[~np.isin(row, excl[u])] if u in excl else row
        out[u] = row[:n]
    return out


def vector_sources(cut, hist, n_users, n_items, tt_dir):
    """{name: (user_vecs, item_vecs, retrievable_item_mask)} for the sources that are available."""
    vs = {}
    if tt_dir and Path(tt_dir, "item_vecs.npy").exists():
        vs["tt"] = (np.load(Path(tt_dir, "user_vecs.npy")), np.load(Path(tt_dir, "item_vecs.npy")), None)
    else:
        print(f"two-tower vectors not found in {tt_dir}; skipping that source", flush=True)
    t = time.time()
    vs["als"] = sources.als(hist, n_users, n_items)
    print(f"als fitted in {time.time() - t:.0f}s", flush=True)
    if (PROC / "audio.npy").exists():
        vs["audio"] = sources.audio_profile(cut, hist, n_users)
    else:
        print("audio.npy not found; skipping the audio source", flush=True)
    return vs


def source_lists(cut, view, users, hist, vs, excl):
    """{source: {uidx: ranked items}} after exclusions, each at least 100 long for stage-1 metrics."""
    depth = 1500 if view == "discovery" else 100      # search deeper when the whole history is excluded
    lists = {}
    for name, (U, I, mask) in vs.items():
        k = max(QUOTA[view][name], 100)
        lists[name] = filtered(search(U, I, users, k + depth, mask), users, excl, k)
    pop = sources.popularity(cut, n=3000)
    lists["pop"] = filtered([pop] * len(users), users, excl, max(QUOTA[view]["pop"], 100))
    if view == "all":
        rep = sources.repeat(cut, hist)
        lists["repeat"] = {u: rep.get(u, np.array([], int))[:max(QUOTA[view]["repeat"], 100)] for u in users}
    return lists


def merge(view, users, lists, vs):
    """Long table of candidates: per-source rank where proposed, plus dense scores from every vector model."""
    rows = []
    for name, per_user in lists.items():
        q = QUOTA[view][name]
        for u in users:
            items = per_user[u][:q]
            rows.append(pd.DataFrame({"uidx": u, "item_idx": items, "src": name, "rank": np.arange(len(items))}))
    long = pd.concat(rows, ignore_index=True)
    cand = long.pivot_table(index=["uidx", "item_idx"], columns="src", values="rank", aggfunc="min")
    cand.columns = [f"{c}_rank" for c in cand.columns]
    cand = cand.reset_index()
    for name in ("tt", "als", "audio", "pop", "repeat"):
        if f"{name}_rank" not in cand:
            cand[f"{name}_rank"] = np.nan
    u, i = cand.uidx.to_numpy(), cand.item_idx.to_numpy()
    for name in VECTOR_SOURCES:
        score = np.full(len(cand), np.nan, np.float32)
        if name in vs:
            U, I, mask = vs[name]
            for s in range(0, len(cand), 1_000_000):
                sl = slice(s, s + 1_000_000)
                score[sl] = np.einsum("ij,ij->i", U[u[sl]], I[i[sl]].astype(np.float32))
            if mask is not None:
                score[~mask[i]] = np.nan
        cand[f"{name}_score"] = score
    rank_cols = [c for c in cand if c.endswith("_rank")]
    cand[rank_cols] = cand[rank_cols].astype(np.float32)
    cand[["uidx", "item_idx"]] = cand[["uidx", "item_idx"]].astype(np.int32)
    assert cand.groupby("uidx").size().max() <= 500
    return cand


def run(cut, tt_dir):
    hist, n_users, n_items = sources.load(cut)
    users = np.array(sorted(metrics.targets(cut, "all")))          # users with a Listen+ in the next window
    vs = vector_sources(cut, hist, n_users, n_items, tt_dir)
    for view in VIEWS:
        t = time.time()
        excl = exclusions(hist, view)
        lists = source_lists(cut, view, users, hist, vs, excl)
        cand = merge(view, users, lists, vs)
        cand.to_parquet(PROC / f"cand_{cut}_{view}.parquet", index=False)
        tg = metrics.targets(cut, view)
        per_user = {u: g.to_numpy() for u, g in cand.groupby("uidx").item_idx}
        res = {"candidate_recall@500": metrics.candidate_recall(per_user, tg),
               "candidates_per_user": float(cand.groupby("uidx").size().mean()),
               "sources": {name: metrics.ranking_metrics(l, tg, n_items) for name, l in lists.items()},
               "source_candidate_recall": {name: metrics.candidate_recall(
                   {u: l[u][:QUOTA[view][name]] for u in l}, tg) for name, l in lists.items()},
               "seconds": round(time.time() - t, 1)}
        metrics.save(f"stage1_{view}" if cut == "AB" else f"stage1_{cut}_{view}", res)


def eval_tt(tt_dir, cut="AB"):
    """Two-tower-only retrieval metrics in both views, plus the cold-start slice (tracks unseen before the cutoff)."""
    hist, n_users, n_items = sources.load(cut)
    U, I = np.load(Path(tt_dir, "user_vecs.npy")), np.load(Path(tt_dir, "item_vecs.npy"))
    seen_items = pd.read_parquet(PROC / f"ifeat_{cut}.parquet", columns=["item_idx"]).item_idx.to_numpy()
    res = {}
    for view in VIEWS:
        tg = metrics.targets(cut, view)
        users = np.array(sorted(tg))
        excl = exclusions(hist, view)
        recs = filtered(search(U, I, users, 100 + (1500 if view == "discovery" else 100)), users, excl, 100)
        res[view] = metrics.ranking_metrics(recs, tg, n_items)
    cold = {u: t[~np.isin(t, seen_items)] for u, t in metrics.targets(cut, "discovery").items()}
    cold = {u: t for u, t in cold.items() if len(t)}
    users = np.array(sorted(cold))
    excl = exclusions(hist, "discovery")
    recs = filtered(search(U, I, users, 1600), users, excl, 100)
    res["cold"] = metrics.ranking_metrics(recs, cold, n_items)
    res["cold"]["targets"] = int(sum(len(t) for t in cold.values()))
    metrics.save(f"tt_{Path(tt_dir).name}", res)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cut", choices=["A", "AB"])
    p.add_argument("--tt", help="two-tower output dir (default artifacts/tt_<cut>)")
    p.add_argument("--eval-tt", help="evaluate only this two-tower output dir on the test day")
    a = p.parse_args()
    if a.eval_tt:
        eval_tt(a.eval_tt)
    else:
        run(a.cut, a.tt or f"artifacts/tt_{a.cut}")
