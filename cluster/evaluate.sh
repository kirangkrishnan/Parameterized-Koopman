#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x_%j.out
# One long job for every system: independent check of the verified region on new
# parameters, then the prediction benchmark (Table II, Fig. 7). Uses the models in
# checkpoints/ unless CKPT is set. Afterwards run scripts/collect_results.py on the
# login node (it takes seconds).
#   sbatch cluster/evaluate.sh
source cluster/env.sh
CKPT=${CKPT:-checkpoints}
for s in vdp slow_manifold manipulator cartpole mujoco_ip mujoco_idp; do
  python scripts/region_check.py "$s" --checkpoints "$CKPT" --seed 2031
  python scripts/prediction_benchmark.py "$s" --checkpoints "$CKPT"
done
