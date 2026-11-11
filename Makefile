.PHONY: setup data audio tt_A tt_AB tt_AB_nohn tt_AB_noaudio tt_AB_s1 tt_AB_noaudio_s1 retrieval rank ablations
PY = PYTHONPATH=src python

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

tt_AB_s1:
	$(TT) --cutoff AB --out artifacts/tt_AB_s1 --seed 1

tt_AB_noaudio_s1:
	$(TT) --cutoff AB --out artifacts/tt_AB_noaudio_s1 --no-audio --seed 1

# stage 1 for both cutoffs (needs artifacts/tt_A and artifacts/tt_AB), then stage 2
retrieval:
	$(PY) -m yrec.retrieval --cut A
	$(PY) -m yrec.retrieval --cut AB

rank:
	$(PY) -m yrec.ranker

# two-tower-only metrics (both views + cold-start slice) for every trained variant
ablations:
	for d in artifacts/tt_AB*; do $(PY) -m yrec.retrieval --eval-tt $$d; done
