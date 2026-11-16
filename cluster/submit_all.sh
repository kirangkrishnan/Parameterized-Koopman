#!/bin/bash
# The cluster experiments of the paper, using the models in checkpoints/. Run from
# the repository root after cluster/setup_env.sh. Every job runs for hours.
#   bash cluster/submit_all.sh
# To retrain a model and its region instead: sbatch cluster/train.sh <system> runs/train/<system>
set -e
mkdir -p logs
echo "region check and prediction benchmark, all systems in one job"
sbatch cluster/evaluate.sh
echo "ablation A (cartpole) and ablation B (cartpole), one seed per array task"
sbatch --array=0-4 cluster/ablation.sh retraining
sbatch --array=0-4 cluster/ablation.sh dictionary
echo "ablation B on the other analytic benchmarks"
for s in vdp slow_manifold manipulator; do
  sbatch --array=0-4 cluster/ablation_systems.sh "$s"
done
echo "when everything has finished: python scripts/collect_results.py"
