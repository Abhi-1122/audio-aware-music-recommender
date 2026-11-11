"""Stage-1 retrieval sources. Each returns user and item vectors (searched with FAISS) or ranked lists.

  popularity - fixed top-N list by plays in the last 7 days before the cutoff
  repeat     - the user's own tracks, by recency-weighted Listen+ count (a lookup, no search)
  als        - implicit-feedback matrix factorization (collaborative filtering)
  audio      - content-based: user profile = recency-weighted mean of their tracks' audio embeddings
  tt         - two-tower model (trained separately, see twotower.py)
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from yrec.data import CUTOFFS, PROC

DAY = 86_400


def load(cut):
    """History table + sizes."""
    hist = pd.read_parquet(PROC / f"hist_{cut}.parquet")
    n_users = len(pd.read_parquet(PROC / "users.parquet"))
    n_items = len(pd.read_parquet(PROC / "items.parquet", columns=["idx"]))
    return hist, n_users, n_items


def recency(hist, cut, half_life_days):
    age = (CUTOFFS[cut] - hist.last_ts.fillna(0).to_numpy()) / DAY
    return 0.5 ** (age / half_life_days)


def popularity(cut, n=500):
    f = pd.read_parquet(PROC / f"ifeat_{cut}.parquet", columns=["item_idx", "i_pop_7d"])
    return f.nlargest(n, "i_pop_7d").item_idx.to_numpy()


def repeat(cut, hist):
    """{uidx: history items ranked by recency-weighted Listen+ count}; disliked tracks dropped."""
    h = hist[(hist.n_lp > 0) & ~hist.disliked].copy()
    h["s"] = (h.n_lp + 0.1 * h.n_listen) * recency(h, cut, half_life_days=14)
    h = h.sort_values(["uidx", "s"], ascending=[True, False])
    return {u: g.to_numpy() for u, g in h.groupby("uidx").item_idx}


def confidence(hist):
    """Implicit-feedback weight per (user, track): log-scaled Listen+ count, plus a like bonus."""
    return np.log1p(hist.n_lp.to_numpy()) + 1.0 * hist.liked.to_numpy() + 0.1 * np.log1p(hist.n_listen.to_numpy())


def als(hist, n_users, n_items, factors=64, iters=15, seed=0):
    from implicit.cpu.als import AlternatingLeastSquares
    from threadpoolctl import threadpool_limits
    w = confidence(hist) * ~hist.disliked.to_numpy()
    keep = w > 0
    m = sp.csr_matrix((w[keep].astype(np.float32), (hist.uidx[keep], hist.item_idx[keep])), shape=(n_users, n_items))
    model = AlternatingLeastSquares(factors=factors, regularization=0.05, alpha=10.0, iterations=iters, random_state=seed)
    with threadpool_limits(1, "blas"):  # implicit parallelises itself; nested BLAS threads slow it down
        model.fit(m, show_progress=False)
    items_seen = np.asarray(m.sum(0)).ravel() > 0     # ALS can only place tracks it has seen
    return model.user_factors, model.item_factors, items_seen


def audio_profile(cut, hist, n_users):
    """User vector = normalised, recency- and play-weighted mean of the audio of their Listen+ tracks."""
    audio = np.load(PROC / "audio.npy").astype(np.float32)
    h = hist[hist.n_lp > 0]
    w = (np.log1p(h.n_lp.to_numpy()) * recency(h, cut, half_life_days=30)).astype(np.float32)
    m = sp.csr_matrix((w, (h.uidx, h.item_idx)), shape=(n_users, len(audio)))
    users = m @ audio
    users /= np.linalg.norm(users, axis=1, keepdims=True) + 1e-9
    return users, audio, np.abs(audio).sum(1) > 0
