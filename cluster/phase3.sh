#!/bin/bash
#SBATCH --job-name=phase3
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x_%j.out
# Phase 3 for a model in checkpoints/ (trained before train.py ran Phase 3 itself):
#   sbatch cluster/phase3.sh vdp
source cluster/env.sh
python scripts/verify_region.py "$1" --out runs/verified "${@:2}"
