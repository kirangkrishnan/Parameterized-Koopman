#!/bin/bash
#SBATCH --job-name=control
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%A_%a.out
# Control from many initial states, split over an array:
#   sbatch --array=0-4 cluster/control.sh cartpole ours_x
#   sbatch --array=0-4 cluster/control.sh mujoco_ip nmpc --T 60 --kick 0.5
source cluster/env.sh
SYSTEM=$1; METHOD=$2; shift 2
python scripts/control_benchmark.py --system "$SYSTEM" --method "$METHOD" \
    --chunk "$CHUNK" --n-chunks "$N_CHUNKS" "$@"
