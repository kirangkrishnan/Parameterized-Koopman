# Sourced by every job script. Written for Aalto Triton; on another SLURM
# cluster change the module line and the environment name.
#   PKOOPMAN_ENV     name of the conda environment (default pkoopman)
#   PKOOPMAN_PREFIX  path of an existing environment; if set it is used directly
if [ -n "${PKOOPMAN_PREFIX:-}" ]; then
  export PATH="$PKOOPMAN_PREFIX/bin:$PATH"
else
  PKOOPMAN_ENV=${PKOOPMAN_ENV:-pkoopman}
  module load mamba 2>/dev/null || module load miniconda 2>/dev/null
  source activate "$PKOOPMAN_ENV"
fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
cd "${SLURM_SUBMIT_DIR:-.}"
# number of pieces of an array job (older SLURM versions do not set the count)
N_CHUNKS=${N_CHUNKS:-${SLURM_ARRAY_TASK_COUNT:-1}}
CHUNK=${SLURM_ARRAY_TASK_ID:-0}
echo "$(hostname) $(python --version 2>&1) chunk $CHUNK of $N_CHUNKS"
