#!/bin/bash
#SBATCH --job-name=ablation_dz
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
# recent PyTorch builds (CUDA 13) no longer run on V100s
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g
#SBATCH --output=logs/%x_%A_%a.out
# Ablation B at a smaller latent dimension, seeds 0, 8, 17 (one per array task):
#   sbatch --array=0-2 cluster/ablation_dz.sh vdp 4
# A fixed dictionary has to span the eigenfunctions of every closed loop at once,
# a conditioned one only those of the current one, so the gap should open up as
# the latent dimension shrinks.
source cluster/env.sh
SEEDS=(0 8 17)
SEED=${SEEDS[$CHUNK]}
python scripts/ablation_dictionary.py --system "$1" --dz "$2" --test-params region --seeds "$SEED" \
  --out "runs/ablation_dz/$1_dz$2/seed_$SEED"
