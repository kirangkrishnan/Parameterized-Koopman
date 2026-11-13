#!/bin/bash
#SBATCH --job-name=retrain_tol
#SBATCH --time=36:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
# recent PyTorch builds (CUDA 13) no longer run on V100s
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g
#SBATCH --output=logs/%x_%A_%a.out
# Rebuild the region (and retrain) with a different tolerance, one per array task:
#   sbatch --array=0-2 cluster/retrain_tolerance.sh cartpole
source cluster/env.sh
TOLS=(10 15 20)
TOL=${TOLS[$CHUNK]}
python scripts/train.py "$1" --threshold "$TOL" --out "runs/train_tol$TOL"
