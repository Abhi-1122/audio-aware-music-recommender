"""Recall@K, NDCG@K, catalogue Coverage@K and stage-1 candidate recall, in two evaluation views.

Views (targets are Listen+ (user, track) pairs in the window after a cutoff):
  all       - every Listen+ track; rankings may contain tracks the user already knows
  discovery - only tracks the user never played before the cutoff; rankings must exclude history
"""
import json

import numpy as np
import pandas as pd

from yrec.data import PROC

KS = (10, 50, 100)


def targets(cut, view):
    """{uidx: np.array of target item_idx} for the window after `cut`."""
    t = pd.read_parquet(PROC / f"tgt_{cut}.parquet", columns=["uidx", "item_idx", "lp", "is_new"])
    t = t[t.lp & t.is_new] if view == "discovery" else t[t.lp]
    return {u: g.to_numpy() for u, g in t.groupby("uidx").item_idx}


def ranking_metrics(recs, tgts, n_items, ks=KS):
    """recs: {uidx: ranked item_idx array}. Users with targets but no recs count as zero.

    Recall@k = hits@k / min(#targets, k); NDCG with binary gains; Coverage@k = unique items / n_items.
    """
    out = {f"{m}@{k}": [] for m in ("recall", "ndcg") for k in ks}
    disc = 1 / np.log2(np.arange(2, max(ks) + 2))
    for u, tg in tgts.items():
        r = np.asarray(recs.get(u, []))[: max(ks)]
        hit = np.isin(r, tg).astype(float)
        for k in ks:
            out[f"recall@{k}"].append(hit[:k].sum() / min(len(tg), k))
            out[f"ndcg@{k}"].append((hit[:k] * disc[: len(hit[:k])]).sum() / disc[: min(len(tg), k)].sum())
    res = {m: float(np.mean(v)) for m, v in out.items()}
    for k in ks:
        res[f"coverage@{k}"] = len(np.unique(np.concatenate([np.asarray(r)[:k] for r in recs.values()]))) / n_items
    res["users"] = len(tgts)
    return res


def candidate_recall(cands, tgts):
    """Share of target (user, track) pairs present in the user's candidate set (micro-averaged)."""
    hit = sum(np.isin(tg, cands.get(u, [])).sum() for u, tg in tgts.items())
    return float(hit / sum(len(tg) for tg in tgts.values()))


def save(name, res, path="results"):
    from pathlib import Path
    Path(path).mkdir(exist_ok=True)
    Path(path, f"{name}.json").write_text(json.dumps(res, indent=2))
    print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in flatten(res).items()}, flush=True)


def flatten(d, pre=""):
    out = {}
    for k, v in d.items():
        out.update(flatten(v, f"{pre}{k}.") if isinstance(v, dict) else {f"{pre}{k}": v})
    return out


if __name__ == "__main__":  # self-check
    from sklearn.metrics import ndcg_score
    tg = {0: np.array([5, 7]), 1: np.array([3])}
    perfect = ranking_metrics({0: [5, 7, 1], 1: [3, 2, 1]}, tg, n_items=10, ks=(1, 3))
    assert perfect["recall@3"] == 1 and perfect["ndcg@3"] == 1 and perfect["recall@1"] == 1
    miss = ranking_metrics({0: [1, 2, 3]}, tg, n_items=10, ks=(1, 3))  # user 1 has no recs -> 0
    assert miss["recall@3"] == 0 and miss["ndcg@3"] == 0 and miss["coverage@3"] == 0.3
    r = ranking_metrics({0: [1, 5, 2, 7]}, {0: np.array([5, 7])}, n_items=10, ks=(4,))
    rel = np.zeros((1, 10)); rel[0, [5, 7]] = 1
    score = np.zeros((1, 10)); score[0, [1, 5, 2, 7]] = [4, 3, 2, 1]
    assert abs(r["ndcg@4"] - ndcg_score(rel, score, k=4)) < 1e-9
    assert candidate_recall({0: [5, 9], 1: [3]}, tg) == 2 / 3
    print("metrics self-check ok")
