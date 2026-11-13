#!/bin/bash
#SBATCH --job-name=prediction
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
# Prediction benchmark with a given source of test parameters:
#   sbatch cluster/prediction.sh cartpole fresh
#   sbatch cluster/prediction.sh mujoco_ip fresh --retrain
source cluster/env.sh
SYSTEM=$1; MODE=$2; shift 2
python scripts/prediction_benchmark.py "$SYSTEM" --thetas "$MODE" "$@"
