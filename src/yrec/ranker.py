"""Stage 2: LightGBM LambdaRank re-ranking of the stage-1 candidates (one model per view).

Train on cut A candidates labelled with the validation day (B), predict on cut AB candidates,
keep the top 100 per user and evaluate on the test day (C).
Graded labels: 0 = disliked in the window, 1 = no Listen+, 2 = Listen+, 3 = Listen+ and liked.
Needs the candidates from `make retrieval`; writes artifacts/ranker_<view>.txt and results/stage2_<view>.json.

  PYTHONPATH=src python -m yrec.ranker [--view all|discovery|both]
"""
import argparse
import json
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from yrec import metrics
from yrec.data import CUTOFFS, PROC

SOURCES = ("tt", "als", "audio", "pop", "repeat")
HIST = ["n_listen", "n_lp", "days_since_play", "avg_ratio", "liked", "disliked"]
PARAMS = dict(
    objective="lambdarank", metric="ndcg", eval_at=[10],
    # A plain negative gets a little gain so dislikes (0) still rank below it; liked counts double.
    label_gain=[0, 0.1, 1, 2],
    learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    seed=0, verbose=-1,
)


def features(cut, view):
    """Candidates of (cut, view) joined with cutoff-safe features, sorted by uidx (one query per user)."""
    df = duckdb.sql(f"""
        SELECT c.*,
               u.* EXCLUDE (uidx), i.* EXCLUDE (item_idx), it.has_audio::FLOAT AS has_audio,
               h.n_listen, h.n_lp, ({CUTOFFS[cut]} - h.last_ts) / 86400.0 AS days_since_play,
               h.avg_ratio, h.liked::FLOAT AS liked, h.disliked::FLOAT AS disliked,
               ua.ua_n_listen, ua.ua_n_lp, ua.ua_n_lp / nullif(u.u_n_lp, 0) AS ua_lp_share
        FROM '{PROC}/cand_{cut}_{view}.parquet' c
        LEFT JOIN '{PROC}/ufeat_{cut}.parquet' u USING (uidx)
        LEFT JOIN '{PROC}/ifeat_{cut}.parquet' i USING (item_idx)
        LEFT JOIN '{PROC}/items.parquet' it ON it.idx = c.item_idx
        LEFT JOIN '{PROC}/hist_{cut}.parquet' h USING (uidx, item_idx)
        LEFT JOIN '{PROC}/uartist_{cut}.parquet' ua ON ua.uidx = c.uidx AND ua.artist_idx = it.artist_idx
        ORDER BY c.uidx, c.item_idx""").df()
    df["n_sources"] = df[[f"{s}_rank" for s in SOURCES]].notna().sum(1)
    if view == "discovery":
        assert (df.n_listen.fillna(0) == 0).all(), "discovery candidates must not be in the user's history"
        df = df.drop(columns=HIST)  # all empty by construction
    return df


def labels(df, cut, view):
    t = df[["uidx", "item_idx"]].merge(pd.read_parquet(PROC / f"tgt_{cut}.parquet"), how="left")
    lp, liked, disliked, new = (t[c].eq(True).to_numpy() for c in ("lp", "liked", "disliked", "is_new"))
    pos = lp & new if view == "discovery" else lp
    return np.select([disliked, pos & liked, pos], [0, 3, 2], 1)


def feature_cols(df):
    return [c for c in df.columns if c not in ("uidx", "item_idx")]


def dataset(df, y, mask, **kw):
    _, groups = np.unique(df.uidx.to_numpy()[mask], return_counts=True)  # df is sorted by uidx
    return lgb.Dataset(df.loc[mask, feature_cols(df)].astype("float32"), y[mask], group=groups, **kw)


def train(view, rounds=2000):
    df = features("A", view)
    y = labels(df, "A", view)
    has_pos = pd.Series(y >= 2).groupby(df.uidx.to_numpy()).transform("any").to_numpy()
    users = np.unique(df.uidx[has_pos])
    val_users = users[np.random.default_rng(0).random(len(users)) < 0.1]
    va = has_pos & df.uidx.isin(val_users).to_numpy()
    tr = has_pos & ~va
    dtr = dataset(df, y, tr)
    dva = dataset(df, y, va, reference=dtr)
    t0 = time.time()
    booster = lgb.train(PARAMS, dtr, rounds, valid_sets=[dva], valid_names=["val"],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])
    stats = {"train_rows": int(tr.sum()), "train_users": int(len(users) - len(val_users)),
             "train_positives": int((y[tr] >= 2).sum()), "val_users": int(len(val_users)),
             "candidate_users_A": int(df.uidx.nunique()), "users_with_positive_A": int(len(users)),
             "best_iteration": booster.best_iteration, "val_ndcg@10": float(booster.best_score["val"]["ndcg@10"]),
             "train_time_s": round(time.time() - t0, 1)}
    return booster, stats


def top(df, score, k=100):
    """{uidx: item_idx ranked by score (descending)}, top k."""
    d = pd.DataFrame({"uidx": df.uidx.to_numpy(), "item_idx": df.item_idx.to_numpy(), "s": score})
    d = d.sort_values(["uidx", "s"], ascending=[True, False], kind="stable")
    d = d[d.groupby("uidx").cumcount() < k]
    return {u: g.to_numpy() for u, g in d.groupby("uidx").item_idx}


def best_source(df, view, tgts, n_items, results):
    """NDCG@10 of the best single source's own ordering: from stage 1 results if present, else from the candidates."""
    f = Path(results, f"stage1_{view}.json")
    src = {s: m["ndcg@10"] for s, m in json.loads(f.read_text()).get("sources", {}).items()
           if "ndcg@10" in m} if f.exists() else {}
    origin = "stage1"
    if not src:
        origin = "candidates"
        for s in SOURCES:
            r = df[f"{s}_rank"].to_numpy()
            ok = ~np.isnan(r)
            if ok.any():
                src[s] = metrics.ranking_metrics(top(df.loc[ok, ["uidx", "item_idx"]], -r[ok]), tgts, n_items)["ndcg@10"]
    s = max(src, key=src.get)
    return {"source": s, "ndcg@10": src[s], "from": origin, "all_sources": src}


def run(view, results, artifacts):
    booster, stats = train(view)
    Path(artifacts).mkdir(parents=True, exist_ok=True)
    booster.save_model(str(Path(artifacts, f"ranker_{view}.txt")))

    df = features("AB", view)
    score = booster.predict(df[feature_cols(df)].astype("float32"), num_iteration=booster.best_iteration)
    tgts = metrics.targets("AB", view)
    n_items = len(pd.read_parquet(PROC / "items.parquet", columns=["idx"]))
    res = metrics.ranking_metrics(top(df, score), tgts, n_items)
    gain = booster.feature_importance("gain")
    order = np.argsort(-gain)[:25]
    res |= {"best_single_source": best_source(df, view, tgts, n_items, results),
            "feature_importance_gain": {booster.feature_name()[i]: round(float(gain[i]), 1) for i in order},
            "features": booster.feature_name(), "params": PARAMS, **stats}
    metrics.save(f"stage2_{view}", res, path=results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", choices=["all", "discovery", "both"], default="both")
    ap.add_argument("--results", default="results")
    ap.add_argument("--artifacts", default="artifacts")
    a = ap.parse_args()
    for v in ["all", "discovery"] if a.view == "both" else [a.view]:
        run(v, a.results, a.artifacts)
