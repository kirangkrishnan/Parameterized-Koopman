#!/bin/bash
#SBATCH --job-name=collect
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x_%j.out
source cluster/env.sh
python scripts/collect_results.py
