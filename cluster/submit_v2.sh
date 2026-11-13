#!/bin/bash
# Second round: Phase 3 with the final model, retraining where it fails, and
# ablation B on every analytic benchmark. Run from the repository root.
#   bash cluster/submit_v2.sh
set -e
mkdir -p logs
dep() { sbatch --parsable "$@"; }

echo "Phase 3 on the models in checkpoints/"
for s in vdp slow_manifold slow_manifold_u2 manipulator cartpole mujoco_ip mujoco_idp; do
  id=$(dep cluster/phase3.sh "$s"); echo "  $s phase3 $id"
  sbatch --dependency=afterok:$id cluster/followup.sh "$s" runs/verified "runs/v2/results/${s}_verified" >/dev/null
done

echo "retraining: region built at 10% over 12 steps, Phase 3 at 15% over 12 steps"
for s in slow_manifold_u2 manipulator cartpole; do
  for v in A B; do
    extra=(); [ $v = B ] && extra=(--x0-margin 1.2)
    out=runs/v2/${s}_$v
    id=$(dep cluster/train_v2.sh "$s" "$out" --threshold 10 --check-steps 12 "${extra[@]}")
    echo "  $s $v $id"
    sbatch --dependency=afterok:$id cluster/followup.sh "$s" "$out" "runs/v2/results/${s}_$v" >/dev/null
  done
done
# the MuJoCo test states lie well inside the training box, so no margin variant;
# the construction tolerance is varied instead
for s in mujoco_ip mujoco_idp; do
  for t in 10 15; do
    out=runs/v2/${s}_t$t
    id=$(dep cluster/train_v2.sh "$s" "$out" --threshold $t --check-steps 12)
    echo "  $s t$t $id"
    sbatch --dependency=afterok:$id cluster/followup.sh "$s" "$out" "runs/v2/results/${s}_t$t" >/dev/null
  done
done

echo "ablation B on every analytic benchmark"
for s in vdp slow_manifold slow_manifold_u2 manipulator cartpole; do
  sbatch --array=0-4 cluster/ablation_systems.sh "$s"
done
