#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH --qos=medium
#SBATCH -N 1
#SBATCH -c 38
#SBATCH --mem=110G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
# Slurm wrapper: runs one Makefile target (TARGET) on a cluster node. Partition/account lines are site-specific.
# CPU job:  sbatch -J data --export=ALL,TARGET=data job.sbatch
# GPU job:  sbatch -J tt -c 8 --mem=24G --gres=gpu:1 --constraint=2080ti --exclude=gnode060 --export=ALL,TARGET=tt_A job.sbatch
set -euo pipefail
source ~/venv312/bin/activate
cd ~/recsys
mkdir -p logs
nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || true
make ${TARGET:?set TARGET}
