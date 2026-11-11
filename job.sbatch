#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH --qos=medium
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode060
#SBATCH -c 40
#SBATCH --mem=120G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
# usage: sbatch -J data --export=ALL,TARGET=data job.sbatch
set -euo pipefail
source ~/venv312/bin/activate
cd ~/recsys
export SCRATCH_DIR=/scratch/$USER/yambda  # node-local, for big raw downloads only
mkdir -p "$SCRATCH_DIR" logs
nvidia-smi --query-gpu=index,name --format=csv,noheader
make "${TARGET:?set TARGET}"
