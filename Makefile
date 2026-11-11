.PHONY: setup data audio
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
