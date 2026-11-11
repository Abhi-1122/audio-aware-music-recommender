"""Data pipeline (first step): raw Yambda-50M parquet -> processed tables under DATA_DIR/proc.

Runs sql/01_load.sql once, then sql/02_split.sql and sql/03_features.sql for each history cutoff.
Raw input: the flat 50M parquet files of the Hugging Face dataset yandex/yambda in DATA_DIR/raw.

  PYTHONPATH=src python -m yrec.data build                               # users, items, events, hist / tgt / features
  PYTHONPATH=src python -m yrec.data audio data/raw/embeddings.parquet   # audio.npy, refreshes items.has_audio
  PYTHONPATH=src python -m yrec.data items                               # rewrite items.parquet only

Time windows (seconds; the official Yambda split with 30-minute gaps):
  A  = [0, A_END)            retrievers for the ranker's training period
  B  = [B_START, B_END)      ranker training labels (validation day)
  AB = [0, B_END)            retrievers for the test period
  C  = [C_START, inf)        test day
"""
import os
from pathlib import Path

DAY, GAP = 86_400, 1_800
C_START = 26_000_000 - DAY
B_END = C_START - GAP
B_START = B_END - DAY
A_END = B_START - GAP
CUTOFFS = {"A": A_END, "AB": B_END}          # history cutoff -> next window
WINDOWS = {"A": (B_START, B_END), "AB": (C_START, 10**12)}

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
RAW, PROC = DATA_DIR / "raw", DATA_DIR / "proc"
SQL = Path(__file__).resolve().parents[2] / "sql"


def run_sql(con, name, **params):
    """Execute sql/<name> with its {placeholders} (raw, proc and params) filled in."""
    sql = (SQL / name).read_text().format(raw=RAW, proc=PROC, **params)
    con.execute(sql)


def build():
    """Raw events -> PROC: users, items, Listen+ events, and per-cutoff history, labels and features."""
    import duckdb
    PROC.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    run_sql(con, "01_load.sql")
    for cut, cutoff in CUTOFFS.items():
        wstart, wend = WINDOWS[cut]
        assert cutoff < wstart, "history must end before its label window starts"
        run_sql(con, "02_split.sql", cut=cut, cutoff=cutoff, wstart=wstart, wend=wend)
        run_sql(con, "03_features.sql", cut=cut, cutoff=cutoff)
        # leakage guards: history ends before the cutoff, every labelled user has history
        assert con.sql(f"SELECT max(last_ts) FROM '{PROC}/hist_{cut}.parquet'").fetchone()[0] < cutoff
        assert con.sql(f"SELECT count(*) FROM '{PROC}/tgt_{cut}.parquet' WHERE uidx NOT IN "
                       f"(SELECT uidx FROM '{PROC}/hist_{cut}.parquet')").fetchone()[0] == 0
        print(cut, con.sql(f"SELECT count(*), sum(lp::INT), sum((lp AND is_new)::INT) "
                           f"FROM '{PROC}/tgt_{cut}.parquet'").fetchone(), flush=True)
    write_items(con)


def write_items(con=None):
    """items.parquet; has_audio comes from audio.npy when it exists."""
    import duckdb, numpy as np, pandas as pd
    con = con or duckdb.connect()
    items = con.sql("SELECT * FROM items ORDER BY idx").df() if con.sql(
        "SELECT count(*) FROM information_schema.tables WHERE table_name='items'").fetchone()[0] \
        else pd.read_parquet(PROC / "items.parquet")
    audio = PROC / "audio.npy"
    items["has_audio"] = (np.abs(np.load(audio, mmap_mode="r")).sum(1) > 0) if audio.exists() else False
    items.to_parquet(PROC / "items.parquet", index=False)
    print("items", len(items), "with audio", int(items.has_audio.sum()), flush=True)


def audio(src):
    """Filter the 13.8 GB global embeddings file to the catalogue's tracks -> audio.npy (n_items, 128) float16."""
    import duckdb, numpy as np
    con = duckdb.connect()
    t = con.sql(f"""SELECT i.idx, e.normalized_embed AS v
                    FROM read_parquet('{src}') e JOIN read_parquet('{PROC}/items.parquet') i USING (item_id)""").df()
    n = con.sql(f"SELECT count(*) FROM '{PROC}/items.parquet'").fetchone()[0]
    out = np.zeros((n, 128), np.float16)
    out[t.idx.to_numpy()] = np.stack(t.v.to_numpy()).astype(np.float16)
    np.save(PROC / "audio.npy", out)
    write_items()


if __name__ == "__main__":
    import sys
    {"build": build, "audio": lambda: audio(sys.argv[2]), "items": write_items}[sys.argv[1]]()
