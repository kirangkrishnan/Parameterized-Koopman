#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
# recent PyTorch builds (CUDA 13) no longer run on V100s
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g
#SBATCH --output=logs/%x_%A_%a.out
# One seed per array task:
#   sbatch --array=0-4 cluster/ablation.sh retraining
#   sbatch --array=0-4 cluster/ablation.sh dictionary
source cluster/env.sh
SEEDS=(0 8 17 42 99)
SEED=${SEEDS[$CHUNK]}
case $1 in
  retraining) python scripts/ablation_retraining.py --seeds "$SEED" ;;
  dictionary) python scripts/ablation_dictionary.py --seeds "$SEED" --out "runs/ablation_dictionary/seed_$SEED" ;;
  *) echo "usage: $0 retraining|dictionary"; exit 1 ;;
esac
