#!/bin/bash
# PSI Tier-3: closure toys as a job array; one toy per task, seed = array index.
#   sbatch --array=1-30 psi_t3/toys.sbatch configs/rds2018_v1.yaml
#   (when all are done)  scalesmear toys configs/rds2018_v1.yaml      -> toys_summary.json, toys_pulls.png
#SBATCH --job-name=scalesmear-toy
#SBATCH --partition=standard
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
ENV_DIR="${ENV_DIR:-$REPO/.venv}"
export RDS_NTUPLES="${RDS_NTUPLES:-/work/$USER/rds/ntuples}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source "$ENV_DIR/bin/activate"
cd "$REPO"
CFG="$1"; shift
scalesmear toy "$CFG" --seed "$SLURM_ARRAY_TASK_ID" "$@"
