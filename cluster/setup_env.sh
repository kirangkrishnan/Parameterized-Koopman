#!/bin/bash
# One-time setup on the login node, run from the repository root:
#   bash cluster/setup_env.sh
set -e
module load mamba 2>/dev/null || module load miniconda
mamba create -y -n pkoopman python=3.11
source activate pkoopman
pip install -r requirements.txt
python -c "import torch, casadi, gymnasium, mujoco; print('environment ok')"
