"""Online prototype of the two-stage recommender, serving the state right before the test day (cut AB).

  stage 1: per-source FAISS retrieval (two-tower, ALS, audio) + popularity + repeat, same quotas as retrieval.py
  stage 2: LightGBM LambdaRank over the same features as ranker.py

  PYTHONPATH=src python -m yrec.serve                 # http://127.0.0.1:8000/docs
  PYTHONPATH=src python -m yrec.serve --bench 200     # in-process latency benchmark
"""
import argparse
import json
import os
import resource
import time
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from yrec import sources
from yrec.data import CUTOFFS, PROC
from yrec.ranker import SOURCES
from yrec.retrieval import QUOTA, VIEWS, exclusions, filtered, merge

CUT = "AB"
ARTIFACTS = os.environ.get("ARTIFACTS", "artifacts")
TT_DIR = os.environ.get("TT_DIR", f"{ARTIFACTS}/tt_{CUT}")
S = {}  # everything loaded at startup


def user_rows(uidx, n_users):
    """[positions of user u's rows for u in range(n_users)], without copying the table."""
    order = np.argsort(uidx, kind="stable")
    return np.split(order, np.searchsorted(uidx[order], np.arange(1, n_users)))


def load(artifacts=ARTIFACTS, tt_dir=TT_DIR):
    t0 = time.time()
    boosters = {v: lgb.Booster(model_file=str(Path(artifacts, f"ranker_{v}.txt"))) for v in VIEWS}
    # same files as twotower.load_vectors, read directly: importing yrec.twotower loads torch (~0.5 GB RSS)
    vs = {"tt": (np.load(Path(tt_dir, "user_vecs.npy")), np.load(Path(tt_dir, "item_vecs.npy")), None)}
    hist, n_users, n_items = sources.load(CUT)

    als_file = Path(artifacts, f"als_{CUT}.npz")
    if als_file.exists():
        f = np.load(als_file)
        vs["als"] = (f["user_factors"], f["item_factors"], f["items_seen"])
    else:
        vs["als"] = sources.als(hist, n_users, n_items)
        np.savez(als_file, **dict(zip(["user_factors", "item_factors", "items_seen"], vs["als"])))
    vs["audio"] = sources.audio_profile(CUT, hist, n_users)

    index = {}  # name -> (FAISS index over the retrievable items, their item_idx)
    for name, (_, I, mask) in vs.items():
        ids = np.flatnonzero(mask) if mask is not None else np.arange(len(I))
        index[name] = faiss.IndexFlatIP(I.shape[1]), ids
        index[name][0].add(np.ascontiguousarray(I[ids], dtype=np.float32))

    items = pd.read_parquet(PROC / "items.parquet")  # row i has idx i
    users = pd.read_parquet(PROC / "users.parquet")
    ua = pd.read_parquet(PROC / f"uartist_{CUT}.parquet")
    S.update(
        boosters=boosters, vs=vs, index=index, hist=hist, ua=ua,
        hist_rows=user_rows(hist.uidx.to_numpy(), n_users), ua_rows=user_rows(ua.uidx.to_numpy(), n_users),
        excl={v: exclusions(hist, v) for v in VIEWS},
        repeat=sources.repeat(CUT, hist), pop=sources.popularity(CUT, n=3000),
        ufeat=pd.read_parquet(PROC / f"ufeat_{CUT}.parquet").set_index("uidx"),
        ifeat=pd.read_parquet(PROC / f"ifeat_{CUT}.parquet").set_index("item_idx"),
        artist=items.artist_idx.to_numpy(), has_audio=items.has_audio.to_numpy(),
        item_id=items.item_id.to_numpy(), item_idx=pd.Index(items.item_id),
        uidx=dict(zip(users.uid.tolist(), users.uidx.tolist())),
    )
    for v in VIEWS:  # warm-up; also fails fast if a booster expects a feature that is not built here
        recommend(0, v)
    print(f"loaded in {time.time() - t0:.0f}s, {len(users)} users, {n_items} items", flush=True)


def stage1(u, view):
    """Candidates for one user, as retrieval.source_lists + merge, with the prebuilt indexes."""
    excl = S["excl"][view]
    depth = 1500 if view == "discovery" else 100
    lists = {}
    for name, (index, ids) in S["index"].items():
        k = max(QUOTA[view][name], 100)
        _, nn = index.search(np.ascontiguousarray(S["vs"][name][0][[u]], dtype=np.float32), k + depth)
        lists[name] = filtered(ids[nn], [u], excl, k)
    lists["pop"] = filtered([S["pop"]], [u], excl, max(QUOTA[view]["pop"], 100))
    if view == "all":
        lists["repeat"] = {u: S["repeat"].get(u, np.array([], int))[:max(QUOTA[view]["repeat"], 100)]}
    return merge(view, [u], lists, S["vs"])


def features(cand, u):
    """ranker.features() for one user's candidates: the same columns, built with in-memory lookups."""
    i = cand.item_idx.to_numpy()
    n = len(i)
    uf = S["ufeat"].reindex([u] * n).reset_index(drop=True)
    h = S["hist"].iloc[S["hist_rows"][u]].set_index("item_idx").reindex(i)
    ua = S["ua"].iloc[S["ua_rows"][u]].set_index("artist_idx").reindex(S["artist"][i])
    u_n_lp = uf.u_n_lp.to_numpy()
    extra = pd.DataFrame({
        "has_audio": S["has_audio"][i].astype(np.float32),
        "n_listen": h.n_listen.to_numpy(), "n_lp": h.n_lp.to_numpy(),
        "days_since_play": (CUTOFFS[CUT] - h.last_ts.to_numpy(np.float64)) / 86400.0,
        "avg_ratio": h.avg_ratio.to_numpy(),
        "liked": h.liked.astype(np.float32).to_numpy(), "disliked": h.disliked.astype(np.float32).to_numpy(),
        "ua_n_listen": ua.ua_n_listen.to_numpy(), "ua_n_lp": ua.ua_n_lp.to_numpy(),
        "ua_lp_share": ua.ua_n_lp.to_numpy() / np.where(u_n_lp == 0, np.nan, u_n_lp),
    })
    df = pd.concat([cand.reset_index(drop=True), uf, S["ifeat"].reindex(i).reset_index(drop=True), extra], axis=1)
    df["n_sources"] = df[[f"{s}_rank" for s in SOURCES]].notna().sum(1)
    return df


def recommend(u, view="all", k=20):
    t0 = time.perf_counter()
    cand = stage1(u, view)
    t1 = time.perf_counter()
    booster = S["boosters"][view]
    X = features(cand, u)[booster.feature_name()].to_numpy(np.float32)
    t2 = time.perf_counter()
    score = booster.predict(X)
    top = np.argsort(-score, kind="stable")[:k]
    t3 = time.perf_counter()
    ranks = cand[[f"{s}_rank" for s in SOURCES]].to_numpy()[top]
    items = [{"item_id": int(S["item_id"][cand.item_idx.iat[j]]), "score": float(score[j]),
              "sources": [s for s, r in zip(SOURCES, row) if not np.isnan(r)]} for j, row in zip(top, ranks)]
    ms = lambda a, b: round((b - a) * 1000, 2)
    return {"view": view, "candidates": len(cand), "items": items,
            "timings_ms": {"retrieval": ms(t0, t1), "features": ms(t1, t2), "ranking": ms(t2, t3), "total": ms(t0, t3)}}


@asynccontextmanager
async def lifespan(_):
    if not S:
        load()
    yield


app = FastAPI(title="Yambda two-stage recommender", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "cut": CUT, "users": len(S["uidx"]), "items": len(S["item_id"]),
            "sources": [*S["index"], "pop", "repeat"]}


@app.get("/recommend/{uid}")
def recommend_endpoint(uid: int, view: str = Query("all", pattern="^(all|discovery)$"), k: int = Query(20, ge=1, le=500)):
    if uid not in S["uidx"]:
        raise HTTPException(404, f"unknown user {uid}")
    return {"uid": uid, **recommend(S["uidx"][uid], view, k)}


@app.get("/similar/{item_id}")
def similar(item_id: int, k: int = Query(20, ge=1, le=500)):
    """Nearest tracks by cosine similarity of the (unit-norm) audio embeddings."""
    idx = S["item_idx"].get_indexer([item_id])[0]
    if idx < 0 or not S["has_audio"][idx]:
        raise HTTPException(404, f"unknown track or no audio embedding: {item_id}")
    index, ids = S["index"]["audio"]
    sim, nn = index.search(S["vs"]["audio"][1][[idx]], k + 1)
    keep = ids[nn[0]] != idx
    return {"item_id": item_id, "similar": [{"item_id": int(S["item_id"][j]), "score": float(s)}
                                            for j, s in zip(ids[nn[0]][keep][:k], sim[0][keep][:k])]}


def bench(n, out):
    users = np.random.default_rng(0).choice(len(S["uidx"]), n, replace=False)
    res = {}
    for view in VIEWS:
        t = pd.DataFrame([recommend(int(u), view)["timings_ms"] for u in users])
        res[view] = {c: {"p50": round(t[c].quantile(0.5), 2), "p95": round(t[c].quantile(0.95), 2)} for c in t}
    res |= {"users": n, "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
            "faiss_threads": faiss.omp_get_max_threads()}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=ARTIFACTS, help="ranker_{all,discovery}.txt and the ALS cache")
    ap.add_argument("--tt-dir", default=TT_DIR)
    ap.add_argument("--bench", type=int, help="time this many random users per view instead of serving")
    ap.add_argument("--out", default="results/serving_latency.json")
    a = ap.parse_args()
    load(a.artifacts, a.tt_dir)
    if a.bench:
        bench(a.bench, a.out)
    else:
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000)
