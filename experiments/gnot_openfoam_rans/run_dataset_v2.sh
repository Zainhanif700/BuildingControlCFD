#!/usr/bin/env bash
# New setup: all settings of scenarios_v2.txt (W1, W3, W6; door 1 open, door 2 closed) on the 8 cm mesh, NP at a time.
# Dataset files go to data_v2/; finished settings are skipped. SKIP_EXTRACT=1: OpenFOAM only (for a machine without GPU).
# Usage: bash run_dataset_v2.sh [NP] [names...]      e.g. bash run_dataset_v2.sh 4   or   bash run_dataset_v2.sh 1 N04
set -u
NP="${1:-4}"; shift || true
HERE="$(cd "$(dirname "$0")" && pwd)"
SCEN="$HERE/scenarios_v2.txt"
[ -f "$SCEN" ] || { echo "missing $SCEN (run scenarios_v2.py)"; exit 1; }
cd "$HERE"
mkdir -p logs data_v2
export DATA_SUBDIR=data_v2
grep -v '^#' "$SCEN" | awk -v only="$*" 'BEGIN { n = split(only, a, " "); for (i = 1; i <= n; i++) w[a[i]] = 1 }
    NF && (n == 0 || ($1 in w)) { print $1, $3, $4, "0.08" }' \
    | xargs -P "$NP" -n 4 bash "$HERE/run_one_rans.sh"
echo "extracted: $(ls data_v2/N*.npz 2>/dev/null | wc -l) of $(grep -vc '^#' "$SCEN") settings"
