#!/usr/bin/env bash
# All window settings of ../gnot_openfoam/scenarios.txt (except all windows closed), NP at a time; finished ones are skipped.
# Usage: bash run_dataset_rans.sh [NP]
set -u
NP="${1:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SCEN="$HERE/../gnot_openfoam/scenarios.txt"
[ -f "$SCEN" ] || { echo "missing $SCEN (run ../gnot_openfoam/scenarios.py)"; exit 1; }
cd "$HERE"
mkdir -p logs data
grep -v '^#' "$SCEN" | awk '$3 != "0,0,0,0,0,0,0,0" {print $1, $3}' \
    | xargs -P "$NP" -n 2 bash "$HERE/run_one_rans.sh"
echo "extracted: $(ls data/S*.npz 2>/dev/null | wc -l) of $(grep -v '^#' "$SCEN" | awk '$3 != "0,0,0,0,0,0,0,0"' | wc -l) scenarios"
