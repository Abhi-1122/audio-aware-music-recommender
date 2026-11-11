# Pipeline targets, in order (`make help` lists them):
#   download       raw Yambda-50M files from Hugging Face -> data/raw (~15 GB incl. audio embeddings)
#   setup          install CUDA torch + requirements, write logs/freeze.txt
#   data           raw Yambda-50M parquet -> data/proc (sql/ via yrec.data), then the metrics self-check
#   audio          audio embeddings -> data/proc/audio.npy
#   tt_A, tt_AB    two-tower models for cutoffs A and AB (-> artifacts/tt_*)
#   tt_AB_nohn     ablation: no mined hard negatives
#   tt_AB_noaudio  ablation: no audio in the item tower
#   retrieval      stage 1 candidates + metrics for cutoffs A and AB (retrieval_A / retrieval_AB run one)
#   rank           stage 2 LightGBM ranker, trained on A, evaluated on the test day
#   ablations      two-tower-only metrics for every artifacts/tt_AB* variant
# Serving (not a target): PYTHONPATH=src python -m yrec.serve [--bench 200]

.PHONY: help download setup data audio tt_A tt_AB tt_AB_nohn tt_AB_noaudio retrieval retrieval_A retrieval_AB rank ablations
PY = PYTHONPATH=src python

help:
	@sed -n '/^# Pipeline/,/^# Serving/p' Makefile | sed 's/^# \{0,1\}//'

HF = https://huggingface.co/datasets/yandex/yambda/resolve/main

download:
	mkdir -p data/raw
	for f in listens likes dislikes unlikes undislikes; do curl -sSL -o data/raw/$$f.parquet $(HF)/flat/50m/$$f.parquet; done
	for f in artist_item_mapping album_item_mapping embeddings; do curl -sSL -o data/raw/$$f.parquet $(HF)/$$f.parquet; done

setup:
	pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
	pip install -r requirements.txt
	python -c "import torch,faiss,implicit,lightgbm,duckdb,sklearn; print(torch.__version__, torch.cuda.get_arch_list())"
	pip freeze > logs/freeze.txt

data:
	$(PY) -m yrec.data build
	$(PY) -m yrec.metrics

audio:
	$(PY) -m yrec.data audio data/raw/embeddings.parquet

TT = $(PY) -m yrec.twotower --steps 20000

tt_A:
	$(TT) --cutoff A --out artifacts/tt_A --eval

tt_AB:
	$(TT) --cutoff AB --out artifacts/tt_AB

tt_AB_nohn:
	$(TT) --cutoff AB --out artifacts/tt_AB_nohn --no-hard-neg

tt_AB_noaudio:
	$(TT) --cutoff AB --out artifacts/tt_AB_noaudio --no-audio

# stage 1 for both cutoffs (needs artifacts/tt_A and artifacts/tt_AB)
retrieval:
	$(PY) -m yrec.retrieval --cut A
	$(PY) -m yrec.retrieval --cut AB

retrieval_A:
	$(PY) -m yrec.retrieval --cut A

retrieval_AB:
	$(PY) -m yrec.retrieval --cut AB

rank:
	$(PY) -m yrec.ranker

# two-tower-only metrics (both views + cold-start slice) for every trained variant
ablations:
	for d in artifacts/tt_AB*; do $(PY) -m yrec.retrieval --eval-tt $$d; done
