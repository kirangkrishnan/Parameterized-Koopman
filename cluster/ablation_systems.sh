#!/bin/bash
#SBATCH --job-name=ablation_b
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
# recent PyTorch builds (CUDA 13) no longer run on V100s
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g
#SBATCH --output=logs/%x_%A_%a.out
# Ablation B on another benchmark, one seed per array task, test parameters
# from the deployment region:
#   sbatch --array=0-4 cluster/ablation_systems.sh slow_manifold_u2
source cluster/env.sh
SEEDS=(0 8 17 42 99)
SEED=${SEEDS[$CHUNK]}
python scripts/ablation_dictionary.py --system "$1" --test-params region --seeds "$SEED" \
  --out "runs/ablation_dictionary/$1_region/seed_$SEED"
