#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH --qos=medium
#SBATCH -N 1
#SBATCH -c 38
#SBATCH --mem=110G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
# CPU job:  sbatch -J data --export=ALL,TARGET=data job.sbatch
# GPU job:  sbatch -J tt -c 10 --mem=30G --gres=gpu:4 --constraint=2080ti --exclude=gnode060 --export=ALL,TARGET=twotower job.sbatch
set -euo pipefail
source ~/venv312/bin/activate
cd ~/recsys
export SCRATCH_DIR=/scratch/$USER/yambda  # node-local, only for big raw downloads
mkdir -p "$SCRATCH_DIR" logs
nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || true
make ${TARGET:?set TARGET}
