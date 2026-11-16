#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
# recent PyTorch builds (CUDA 13) no longer run on V100s
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g
#SBATCH --output=logs/%x_%j.out
# Full pipeline for one system (scout model, region, final model, final check),
# with the settings of scripts/train.py. The region tests run on the CPUs of the job.
#   sbatch cluster/train.sh cartpole runs/train/cartpole
source cluster/env.sh
SYSTEM=$1; OUT=$2; shift 2
python scripts/train.py "$SYSTEM" --out "$OUT" "$@"
