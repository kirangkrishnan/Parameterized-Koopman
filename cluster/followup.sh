#!/bin/bash
#SBATCH --job-name=followup
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x_%j.out
# Everything that depends on a model and its region, for one folder of
# checkpoints: region check on a new seed, prediction on fresh parameters and the
# control benchmark of the one-shot methods. Results go under RUNS, with their
# own summary.md.
#   sbatch cluster/followup.sh cartpole runs/v2/cartpole_A runs/v2/cartpole_A/results
source cluster/env.sh
SYSTEM=$1; CKPT=$2; RUNS=$3
mkdir -p "$RUNS"
python scripts/region_check.py "$SYSTEM" --checkpoints "$CKPT" --seed 2031 --out "$RUNS/region_check"
METHODS=()
case $SYSTEM in
  mujoco_*) METHODS=(--methods Ours PKNN "PKNN (re-lift)" "Deep Koopman" EDMD Kernel-EDMD) ;;
esac
python scripts/prediction_benchmark.py "$SYSTEM" --thetas fresh --checkpoints "$CKPT" \
  --out "$RUNS/prediction_benchmark" "${METHODS[@]}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for m in ours_x ours_z centroid sim_opt; do
  python scripts/control_benchmark.py --system "$SYSTEM" --method "$m" --checkpoints "$CKPT" \
    --out "$RUNS/control_benchmark" > "$RUNS/control_$m.log" 2>&1 &
done
wait
python scripts/collect_results.py --runs "$RUNS"
