#!/usr/bin/env bash
# All RANS scenarios: the SAME 40 window settings as the laminar dataset (../gnot_openfoam/scenarios.txt,
# same train/test split), NP at a time (default 4 of 6 cores). S00 (all windows closed) is left out:
# without any air flow there is no turbulent mixing, and molecular diffusion alone is not a realistic
# model of a closed occupied room (people, heat) -- stated as a limitation.
# Re-running skips everything already extracted.
# Usage (from experiments/gnot_openfoam_rans, in tmux):  bash run_dataset_rans.sh [NP]
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
