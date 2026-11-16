#!/bin/bash
#SBATCH --job-name=region_check
#SBATCH --time=06:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%A_%a.out
# Phase 3 on fresh parameters, split over an array:
#   sbatch --array=0-9 cluster/region_check.sh vdp
source cluster/env.sh
SYSTEM=$1; shift
python scripts/region_check.py "$SYSTEM" --chunk "$CHUNK" --n-chunks "$N_CHUNKS" "$@"
