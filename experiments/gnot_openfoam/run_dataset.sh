#!/usr/bin/env bash
# All window settings of scenarios.txt for the laminar dataset, NP at a time; finished ones are skipped.
# Usage: bash run_dataset.sh [NP]
set -u
NP="${1:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${CFD_PY:-$HOME/anaconda3/envs/cfd/bin/python}"
cd "$HERE"
[ -f scenarios.txt ] || "$PY" scenarios.py
mkdir -p logs data
if [ ! -f data/S00.npz ]; then
    "$PY" extract_case.py --closed --name S00 --t-co2 1800 --fv torch > logs/S00.log 2>&1 \
        && echo "S00: closed room done" || echo "S00: FAILED -- see logs/S00.log"
fi
grep -v '^#' scenarios.txt | awk '$3 != "0,0,0,0,0,0,0,0" {print $1, $3}' \
    | xargs -P "$NP" -n 2 bash "$HERE/run_one.sh"
echo "extracted: $(ls data/S*.npz 2>/dev/null | wc -l) of $(grep -vc '^#' scenarios.txt) scenarios"
