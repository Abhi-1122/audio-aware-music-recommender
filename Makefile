PY = PYTHONPATH=src python

setup:
	pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
	pip install -r requirements.txt
	python -c "import torch,faiss,implicit,lightgbm,duckdb,sklearn; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_arch_list(), faiss.get_num_gpus())"
	pip freeze > logs/freeze.txt
