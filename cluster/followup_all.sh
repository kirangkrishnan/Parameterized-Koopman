#!/bin/bash
#SBATCH --job-name=followup_all
#SBATCH --time=36:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x_%j.out
# All follow-ups in one job, one after the other, instead of one short job per
# model (Triton asks for jobs that last hours, not minutes). Each argument is
# SYSTEM:CHECKPOINT_DIR:RESULTS_DIR; a run whose training failed is skipped.
#   sbatch --dependency=afterany:<train jobs> cluster/followup_all.sh \
#       cartpole:runs/v2/cartpole_A:runs/v2/results/cartpole_A ...
source cluster/env.sh
for spec in "$@"; do
  IFS=: read -r SYSTEM CKPT RUNS <<< "$spec"
  if [ ! -f "$CKPT/$SYSTEM.pt" ] || [ ! -f "$CKPT/${SYSTEM}_phase3.json" ]; then
    echo "skipping $spec (no finished model)"; continue
  fi
  echo "=== $spec  $(date)"
  bash cluster/followup.sh "$SYSTEM" "$CKPT" "$RUNS"
done
