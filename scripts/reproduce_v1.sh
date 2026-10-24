#!/usr/bin/env bash
# Rerun the v1 fit from the ROOT ntuples and compare with the published correction file.
#   RDS_NTUPLES=/path/to/ntuples scripts/reproduce_v1.sh [OUTDIR]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${RDS_NTUPLES:?set RDS_NTUPLES to the folder with rs_slim_21oct2026.root and sig_slim_21oct2026.root}"
OUT="${1:-$REPO/results/reproduce_v1}"
scalesmear run "$REPO/configs/rds2018_v1.yaml" --set output.dir="$OUT"
python "$REPO/scripts/compare_results.py" "$REPO/applier/data/rds_trkcalib_2018BParking_v1.json" \
       "$OUT/rds_trkcalib_RDs_2018BParking_DsPhiPi_trkcalib_v1.json"
