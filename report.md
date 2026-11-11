# Audio-aware two-stage music recommendation on Yambda-50M

## Summary

This report describes a music recommender built and evaluated on Yambda-50M, a public release of real Yandex Music interaction logs. The system follows the retrieve-then-rank design used in production recommendation platforms:

- **Stage 1 (retrieval):** five candidate sources propose up to 500 tracks per user. They are a PyTorch two-tower model with an audio-aware track encoder, ALS collaborative filtering, audio content-based retrieval, a popularity list and the user's own recent tracks. The three vector sources are searched with FAISS.
- **Stage 2 (ranking):** a LightGBM LambdaRank model re-ranks those candidates with 45 engineered features.

Evaluation uses a leakage-safe global temporal split and two views of success:

- **All listens:** any track the user plays with at least 50% completion.
- **Discovery:** only tracks the user has never played before.

Results on the held-out test day:

| Measure | All listens | Discovery |
|---|---|---|
| Final NDCG@10 | **0.225** | **0.056** |
| Gain over the best single retriever | +64% | +70% |

- **Published models:** on the all-listens view, final NDCG@10 is close to three times that of the strongest model published with the dataset (ItemKNN, 0.078). The two-tower retriever alone reaches 0.137.
- **Cold start:** for tracks with no plays before the test day, the audio-aware track encoder retrieves 8.4% of test listens within the top 100. ID-only collaborative filtering cannot retrieve those tracks at all.
- **Serving:** a FastAPI prototype answers a full two-stage request in 86 ms (p50) on CPU.

## 1. Problem

A music service has to choose, from a catalogue of close to a million tracks, the handful of tracks a listener will want next. Three properties of the domain shape the design.

1. **Repeat consumption.** People replay favourite tracks every day. A model can score well by recommending what a user already plays. That serves "keep listening" surfaces and says little about discovery.
2. **A long tail and new releases.** Popularity is concentrated in a few tracks, while new tracks arrive with no interaction history. Collaborative filtering cannot place a track it has never seen.
3. **Scale and latency.** Scoring every track for every user with a rich model is too slow, so the work is split. Cheap retrieval narrows the catalogue to a few hundred candidates, and an expressive ranker orders them.

The project set out to:
- build such a two-stage system on real interaction logs;
- measure each stage on its own;
- report re-listening and discovery as two results;
- test whether audio content makes new tracks reachable.

## 2. Data

Yambda-50M is the smallest of three released scales (50M, 500M and 5B events) and contains the following.

| Quantity | Value |
|---|---|
| Users | 10,000 |
| Tracks | 934,057 |
| Listens | 46,467,212 |
| Likes / dislikes | 881,456 / 107,776 |
| Time span | 301 days |
| Tracks with an audio embedding | 847,921 (91%) |

Each listen records the user, the track, a timestamp at 5-second resolution, whether the play was organic or recommendation-driven, the played ratio and the track length. Likes, dislikes and their reversals are separate event files. Every track has an artist and album mapping, and most tracks have a 128-dimensional audio embedding produced by a CNN trained with contrastive learning.

The exploratory analysis (`notebooks/01_eda.ipynb`) found the patterns that drove the design:

- **Repeat listening dominates.** 71% of all listens are re-listens. On the test day, 69% of the (user, track) pairs played to at least 50% were tracks the user already knew; only 31% were new to that user.
- **The tail is long.** The top 1% of tracks receive 49% of listens, while 63% of tracks have five plays or fewer.
- **Listen+.** 63% of listens pass the 50% completion threshold. This is the dataset's own definition of a positive listen, written Listen+.
- **Organic and recommended plays are balanced.** 48% of listens come from recommendation surfaces, and these are skipped less often than organic plays (32% vs 41%).
- **Cold tracks are rare.** On the test day, 1.0% of Listen+ targets (1,329 pairs) are tracks with no plays at all before the cutoff.

## 3. Evaluation protocol

### 3.1 Time windows

The split follows the global temporal protocol published with the dataset. All users share the same time boundaries, so no model sees the future.

| Window | Span | Purpose |
|---|---|---|
| A | start → validation day − 30 min | history for the retrievers whose candidates train the ranker |
| B | the validation day | labels for ranker training |
| C | the final day (test) | labels for final evaluation |

The ranker must never train on the same period its retrievers learned from, or it learns to trust retriever scores that are overfitted to its own labels. The schedule therefore runs in two passes:

1. Retrievers are fitted on A and generate candidates for B. B's actual listens label those candidates, and LightGBM trains on them.
2. For the test, the retrievers are refitted on A+B and generate candidates for C. The B-trained ranker then scores them.

Features use only events before the relevant cutoff. The data build asserts that no history row reaches its cutoff, and that every labelled user has history.

### 3.2 Two views

- **All listens.** Targets are every Listen+ (user, track) pair in the window. Rankings may include tracks the user already knows. This matches the dataset's official protocol.
- **Discovery.** Targets are Listen+ pairs whose track the user never played before the cutoff. Rankings exclude the user's entire history. This view isolates the harder and, for a growing catalogue, more valuable problem of finding something new.

A third slice, **cold-start**, keeps only discovery targets whose track had zero plays by anyone before the cutoff.

### 3.3 Metrics

- **Recall@K:** hits in the top K divided by min(number of targets, K).
- **NDCG@K:** binary gains, averaged per user.
- **Coverage@K:** the share of the catalogue appearing in anyone's top K, which guards against a model that only recommends hits.
- **Candidate recall@500:** the share of target pairs present anywhere in a user's merged candidate set. This is the ceiling for what stage 2 can reach.

The metric code carries an assert-based self-check, including a cross-check of NDCG against scikit-learn's implementation.

## 4. Data pipeline

The pipeline is written in DuckDB SQL (three files in `sql/`) driven by a short Python module, and builds in 20 seconds on a 16-core laptop.

1. **Load** (`01_load.sql`). Raw parquet becomes typed tables with dense user, track, artist and album indices. Each listen gets a Listen+ flag. The step writes the Listen+ event stream, sorted by time, that the two-tower model trains on.
2. **Split** (`02_split.sql`, once per cutoff).
   - A **history table** with one row per (user, track) before the cutoff: play count, Listen+ count, last play time, average played ratio, and the final like and dislike state. Unlikes and undislikes revert the earlier action.
   - A **label table** for the following window, with Listen+, liked, disliked and an "is new to this user" flag.
3. **Features** (`03_features.sql`, once per cutoff).
   - **User features:** volume, skip rate, organic share, repeat ratio, artist diversity, recency and like counts.
   - **Track features:** popularity over 1, 7 and 30 days, trend, skip rate, like and dislike rates, age and artist popularity.
   - **User×artist affinity.**

Implicit feedback becomes a confidence weight per (user, track) for ALS: a log-scaled Listen+ count plus a like bonus and a small weight for partial plays. Disliked tracks are excluded from training and never recommended.

## 5. Stage 1: retrieval

### 5.1 Sources

| Source | Method | Role |
|---|---|---|
| Two-tower | PyTorch dual encoder (Section 5.2), FAISS search | main personalised retriever; reaches cold tracks |
| ALS | `implicit` matrix factorisation, 64 factors, FAISS search over item factors | classic collaborative filtering for warm tracks |
| Audio content-based | user profile = recency- and play-weighted mean of the audio embeddings of their Listen+ tracks; FAISS search over track audio | pure content similarity; widest coverage |
| Popularity | top tracks by plays in the last 7 days | trend signal and fallback |
| Repeat | the user's own tracks, by recency-weighted Listen+ count | re-listening (all-listens view only) |

All vector search uses exact FAISS inner-product indexes over the full catalogue. At this scale, exact search over 934K × 64 vectors takes milliseconds per user.

### 5.2 Two-tower model

**Track tower.** Three representations of a track are concatenated, projected to 64 dimensions and L2-normalised:
- a 64-d ID embedding;
- a 32-d embedding of its artist (147,618 artists);
- an MLP (128 → 128 → 64) over its audio embedding.

During training, each track's ID embedding is swapped for a shared learned "no-ID" vector 30% of the time (ID dropout). This forces the tower to place tracks well from audio and artist alone. At inference, tracks that were never played before the cutoff use the "no-ID" vector, so a brand-new track lands near tracks that sound similar or share its artist. Tracks without audio use a learned "missing audio" vector.

**User tower.** The user is represented by their last 100 Listen+ tracks before the target time, encoded by the same track tower. They are pooled with softmax attention that combines a content score and a learned per-position recency bias, then passed through a residual MLP. There is no per-user embedding table, so the tower generalises to users with little history.

**Objective.** Training uses a sampled softmax over the batch.
- Every other positive in a batch of 4,096 serves as a negative.
- A logQ correction subtracts each track's log training frequency, so popular tracks lose no score for appearing often as negatives.
- Accidental hits are masked.
- Every 1,000 steps, the current model mines **hard negatives**: top-ranked tracks the user never played. The top 10 ranks are skipped to avoid false negatives, and a pool of 64 is kept per user. Eight of those are added to each example as extra softmax columns.

**Training.**
- Training draws 20,000 steps × 4,096 examples (82M samples) from 29M Listen+ events, in fp16 mixed precision with Adam.
- The model has 64.6M parameters.
- On one RTX 2080 Ti a run takes about 7.5 minutes (179–193K examples/s); on a laptop RTX 3070 Ti it takes 22–24 minutes.
- The same code runs under `torchrun` with DistributedDataParallel and fp16 gradient compression.

### 5.3 Candidate merge

Each source fills a fixed quota, and duplicates are merged. The quotas differ by view because the repeat source is useless for discovery.

| View | Two-tower | ALS | Audio | Popularity | Repeat |
|---|---|---|---|---|---|
| All listens | 150 | 100 | 50 | 25 | 175 |
| Discovery | 200 | 150 | 100 | 50 | 0 |

For every candidate, the dense scores of all three vector models are kept, along with each proposing source's rank. These become ranker features.

### 5.4 Stage-1 results (test day)

| Source | All: Recall@100 | All: NDCG@10 | Discovery: Recall@100 | Discovery: NDCG@10 | Coverage@100 |
|---|---|---|---|---|---|
| Two-tower | 0.226 | 0.137 | 0.119 | 0.033 | 0.103 |
| Repeat | 0.303 | 0.131 | n/a | n/a | 0.112 |
| ALS | 0.093 | 0.031 | 0.068 | 0.014 | 0.028 |
| Popularity | 0.057 | 0.022 | 0.057 | 0.013 | 0.000 |
| Audio | 0.022 | 0.008 | 0.011 | 0.002 | 0.144 |
| **Merged, candidate recall@500** | **0.400** | | **0.203** | | |

- **Two-tower vs ALS.** The two-tower model is the strongest learned retriever. It recalls 2.4× as many test-day listens as ALS, and 1.7× as many on discovery, while covering 3.7× more of the catalogue.
- **Repeat.** The simple repeat lookup has the highest recall on all listens, which confirms how much of music consumption is replay.
- **Audio.** Pure audio similarity is a weak predictor of the next play but has the widest coverage.
- **Merging.** Merging sources raises the reachable share of listens to 40% (all listens) and 20% (discovery), well above any single source.

## 6. Stage 2: ranking

### 6.1 Model

LightGBM `lambdarank` optimises NDCG, with one query per user, and a separate model is trained for each view. Labels are graded:

| Grade | Meaning | Gain |
|---|---|---|
| 0 | disliked during the window (explicit negative) | 0 |
| 1 | not played to 50% | 0.1 |
| 2 | Listen+ | 1 |
| 3 | Listen+ and liked | 2 |

Ordinary negatives get a small positive gain, so that dislikes rank below them.

The model uses 63 leaves, a learning rate of 0.05, and feature and row subsampling of 0.8. It trains with early stopping on a 10% user hold-out of the training day.

**Features.** There are 45 features in the all-listens view and 39 in discovery, where the user×track history features are empty by construction:
- the retrievers' scores and ranks, and how many sources proposed the candidate;
- user features and track features;
- user×track history: plays, Listen+ count, days since last play, average played ratio, like and dislike state;
- user×artist affinity.

**Training set.** In the all-listens view: 1.58M candidate rows from 3,837 users, with 46.7K positives; training takes 15 seconds. In the discovery view: 856K rows and 7.5K positives.

### 6.2 Results (test day)

| View | Best single retriever NDCG@10 | LightGBM NDCG@10 | Recall@10 | Recall@100 | Coverage@100 |
|---|---|---|---|---|---|
| All listens | 0.137 (two-tower) | **0.225** (+64%) | 0.224 | 0.368 | 0.100 |
| Discovery | 0.033 (two-tower) | **0.056** (+70%) | 0.061 | 0.160 | 0.071 |

The ranker turns the merged candidates into a ranking much better than any single source.
- **All listens:** Recall@100 rises from 0.303 for the best single source to 0.368, against a ceiling of 0.400.
- **Discovery:** Recall@100 rises from 0.119 to 0.160, against a ceiling of 0.203.
- **What drives each view:** feature importance shows the split between the two problems.
  - On all listens, the top features describe the user's relationship with the track: days since last play, average played ratio, the two-tower rank and score, and the user's repeat ratio.
  - On discovery, the top features describe the track and the user's taste: 7-day trend, ALS and two-tower scores, user–artist affinity, artist popularity, track age and audio similarity.

### 6.3 Reference: models published with the dataset

The dataset paper (Ploshkin et al., 2025, Table 6) reports Yambda-50M Listen+ results under the same split. Its protocol matches this project's all-listens view.

| Model | NDCG@10 | Recall@100 |
|---|---|---|
| MostPop (published) | 0.019 | 0.032 |
| iALS (published) | 0.041 | 0.081 |
| SASRec (published) | 0.075 | 0.103 |
| ItemKNN (published, best) | 0.078 | 0.130 |
| ALS, this project (untuned) | 0.031 | 0.093 |
| Two-tower, this project | 0.137 | 0.226 |
| Two-stage pipeline, this project | **0.225** | **0.368** |

Two notes on comparability:
- **Metric code.** The published numbers came from an earlier version of the dataset authors' metric code. That version aggregated NDCG across users before dividing, and normalised Recall by the number of targets, where the current code uses min(targets, K). The authors state that this changed absolute values and left model rankings intact. The comparison therefore serves as a reference point with known differences.
- **Calibration.** The untuned ALS here scores close to the paper's tuned iALS, which suggests the evaluation code is on a comparable scale.

Most of the gap to the published models comes from two places: modelling the user's own history (repeat source and history features), and the learned re-ranking stage.

## 7. Ablations

Three two-tower variants were trained with the same budget and evaluated on the test day. Each is a single run, so differences of a few percent are within run-to-run noise.

| Variant | All: Recall@100 | All: NDCG@10 | Discovery: Recall@100 | Cold: Recall@100 | Cold: NDCG@10 |
|---|---|---|---|---|---|
| Main (ID + artist + audio, hard negatives) | 0.226 | 0.137 | 0.119 | 0.084 | 0.034 |
| No audio branch | 0.221 | 0.138 | 0.113 | 0.082 | 0.024 |
| No mined hard negatives | 0.223 | 0.138 | 0.121 | 0.084 | 0.029 |

- **Audio features.** Adding audio leaves all-listens accuracy unchanged, raises discovery Recall@100 by 5%, and raises cold-start NDCG@10 by 39%. The gain sits where it was expected: tracks without a useful ID embedding.
- **Cold start.** Even without audio, the ID-dropout design retrieves 8.2% of cold-start listens within the top 100 through the artist embedding. ID-only collaborative filtering (ALS) scores zero on this slice, because it has no vector for unseen tracks.
- **Hard negatives.** At this training budget, mined hard negatives gave no consistent improvement. They helped cold-start NDCG@10 and were neutral elsewhere. One probable reason: in-batch negatives drawn from 4,096 examples, together with logQ correction, already supply informative negatives for a 934K catalogue. A longer schedule or harder mining (smaller skip window, larger pool) might change this.

## 8. Serving prototype

`src/yrec/serve.py` wraps both stages in a FastAPI service that answers as of the test-day cutoff.
- **`/recommend/{uid}?view=all|discovery&k=20`** returns ranked tracks, the sources that proposed each one, and per-stage timings.
- **`/similar/{item_id}`** returns audio nearest neighbours, a "sounds like" lookup that needs no listening history.

The service rebuilds a single user's candidates with the same functions, quotas and exclusions as the batch pipeline. A consistency check confirmed that its candidates and feature values match the batch output row for row.

Latency was measured over 200 random users per view, on a 16-thread laptop CPU, with exact search:

| Stage | All listens p50 / p95 (ms) | Discovery p50 / p95 (ms) |
|---|---|---|
| Retrieval (3 FAISS searches + lists + merge) | 76.6 / 82.9 | 77.9 / 85.1 |
| Feature assembly | 4.2 / 5.4 | 4.1 / 5.5 |
| LightGBM scoring | 5.0 / 8.5 | 4.2 / 7.4 |
| **Total** | **85.7 / 92.6** | **86.9 / 92.5** |

Peak memory of the loaded service is 4.3 GB.

Retrieval dominates, through three exact FAISS searches over the full catalogue. Approximate indexes (IVF or HNSW) and running the three searches in parallel are the obvious next steps if lower latency is needed.

## 9. Engineering

- **Compute.**
  - GPU training ran on a Slurm cluster (RTX 2080 Ti nodes), with CPU stages on 38-core nodes. A laptop RTX 3070 Ti trained models in parallel.
  - Jobs were chained with Slurm dependencies, so each stage started as soon as its inputs existed.
  - Code moved between machines with `rsync`, and version control stayed in a single place.
- **Reproducibility.** Every stage is a `make` target. Dependencies are pinned, and every run writes its configuration and metrics as JSON (`results/`, `artifacts/*/config.json`). Random seeds are fixed.
- **Correctness guards.**
  - leakage asserts in the data build;
  - a metric self-check against scikit-learn;
  - discovery candidates asserted to exclude user history;
  - a serving-versus-batch consistency check.

## 10. Limitations and next steps

- **Single runs.** Ablation differences of a few percent are not significant. Repeated seeds or bootstrap confidence intervals over users would settle them.
- **One test day.** Results come from a single held-out day, which matches the official protocol. Rolling evaluation over several days would show stability.
- **Offline only.** Offline metrics on logged data inherit the bias of the recommendation surfaces that produced half of the listens. An online A/B test is the real arbiter.
- **Sequence modelling.** The user tower pools recent history without modelling order beyond a recency bias. A transformer user encoder (SASRec-style) could capture session dynamics.
- **Scale.** The same code would run on Yambda-500M or Yambda-5B. Larger data could change the hard-negative result and make approximate nearest-neighbour search necessary.

## Appendix: reproduction

```bash
pip install -r requirements.txt
make download data audio                 # raw data -> processed tables + audio embeddings
make tt_A tt_AB tt_AB_nohn tt_AB_noaudio # two-tower models (GPU)
make retrieval rank ablations            # stage 1, stage 2, ablation metrics -> results/
PYTHONPATH=src python -m yrec.serve --bench 200
```

All reported numbers are in `results/*.json`. The notebooks in `notebooks/` regenerate every table and figure from those files.
