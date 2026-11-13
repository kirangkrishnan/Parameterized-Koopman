#!/bin/bash
# Submit every experiment, then a summary job that runs when all of them are done.
# Run from the repository root after cluster/setup_env.sh:
#   bash cluster/submit_all.sh
# Set WITH_TOLERANCE=1 to also rebuild the cartpole region at 10/15/20% (long).
set -e
mkdir -p logs
ids=()
sub() { ids+=("$(sbatch --parsable "$@")"); echo "  ${ids[-1]}  $*"; }

echo "region check (Phase 3 on fresh parameters)"
for s in vdp slow_manifold slow_manifold_u2 manipulator cartpole mujoco_ip; do
  sub --array=0-9 cluster/region_check.sh "$s"
done
sub --array=0-29 cluster/region_check.sh mujoco_idp

echo "prediction on fresh parameters from C and on the whole box"
for s in vdp slow_manifold_u2 manipulator cartpole; do
  for mode in fresh box; do sub cluster/prediction.sh "$s" "$mode"; done
done
for s in mujoco_ip mujoco_idp; do
  for mode in fresh box; do sub cluster/prediction.sh "$s" "$mode" --retrain; done
done

echo "control from 50 initial states"
for s in vdp slow_manifold manipulator cartpole mujoco_ip mujoco_idp; do
  for m in ours_x ours_z centroid nmpc kmpc_edmd kmpc_deep; do
    case "$s:$m" in
      mujoco_ip:kmpc_edmd|mujoco_idp:kmpc_edmd|mujoco_idp:kmpc_deep|mujoco_idp:nmpc) continue ;;
    esac
    sub --array=0-4 cluster/control.sh "$s" "$m"
  done
  sub --array=0-9 cluster/control.sh "$s" sim_opt
done
echo "control: long MuJoCo episodes with a mid-episode perturbation"
for m in ours_x ours_z centroid; do
  sub --array=0-4 cluster/control.sh mujoco_ip "$m" --T 60 --kick 0.5
  sub --array=0-4 cluster/control.sh mujoco_idp "$m" --T 60 --kick 0.5
done
sub --array=0-4 cluster/control.sh mujoco_ip nmpc --T 60 --kick 0.5

echo "ablations (GPU)"
sub --array=0-4 cluster/ablation.sh retraining
sub --array=0-4 cluster/ablation.sh dictionary

if [ "${WITH_TOLERANCE:-0}" = 1 ]; then
  echo "region tolerance sensitivity (GPU)"
  sub --array=0-2 cluster/retrain_tolerance.sh cartpole
fi

dep=$(IFS=:; echo "${ids[*]}")
echo "summary job:"
sbatch --dependency=afterany:"$dep" cluster/collect.sh
echo "${#ids[@]} jobs submitted; results will be in runs/summary.md"
