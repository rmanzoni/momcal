#!/bin/bash
# PSI Tier-3: one data fit.   sbatch psi_t3/fit.sbatch configs/rds2018_v1.yaml [--set key=value ...]
# Check partitions and limits with `sinfo` and adapt --partition / --time; a full fit takes about 1 h on one core.
#SBATCH --job-name=scalesmear-fit
#SBATCH --partition=standard
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
ENV_DIR="${ENV_DIR:-$REPO/.venv}"
export RDS_NTUPLES="${RDS_NTUPLES:-/work/$USER/rds/ntuples}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1      # one core, same arithmetic order every time
source "$ENV_DIR/bin/activate"
cd "$REPO"
echo "host $(hostname)  job $SLURM_JOB_ID  config $1"
scalesmear run "$@"
