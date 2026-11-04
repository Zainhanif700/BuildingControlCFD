#!/usr/bin/env bash
# Test: does saving the OpenFOAM flow more often fix our CO2 in the fast cases?
# Runs one setting of scenarios_v2.txt to 150 s with a save every second (own cases/ folder here),
# then compare_sampling.py computes our CO2 from saves every 1/2/5/10 s and compares it with OpenFOAM's CO2.
# Usage: SAMPLING_CASES=<folder for the case> bash run_sampling_test.sh [name, default N28] [cores, default 4]
set -euo pipefail
NAME="${1:-N28}"; NP="${2:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
RANS="$(cd "$HERE/../gnot_openfoam_rans" && pwd)"
PY="$(command -v "${CFD_PY:-python3}")"
read -r V DOORS < <(awk -v n="$NAME" '$1 == n { print $3, $4 }' "$RANS/scenarios_v2.txt") || true
[ -n "${V:-}" ] || { echo "$NAME not in scenarios_v2.txt"; exit 1; }
CASES="${SAMPLING_CASES:-$HERE/cases}"
CASE="$CASES/${NAME}ws1_dx0.08"
echo "[$(date '+%F %T')] $NAME (V $V, doors $DOORS): OpenFOAM to 150 s, save every 1 s, $NP cores, python $PY"
if [ ! -f "$CASE/done" ]; then
    rm -rf "$CASE"
    mkdir -p "$CASES"
    (cd "$RANS" && "$PY" make_rans_case.py --V "$V" --doors "$DOORS" --dx 0.08 --name "${NAME}ws1" \
        --t-end 150 --write-interval 1 --out "$CASES" > /dev/null)
    FOAM_NP="$NP" bash "$RANS/run_rans_case.sh" "$CASE" > "$CASE.log" 2>&1
    grep -q "^End" "$CASE/log.pimpleFoam" || { echo "OpenFOAM did not reach the end -- see $CASE.log"; exit 1; }
    touch "$CASE/done"
    echo "[$(date '+%F %T')] OpenFOAM done, case size $(du -sh "$CASE" | cut -f1)"
fi
"$PY" "$HERE/compare_sampling.py" --case "$CASE"
echo "[$(date '+%F %T')] FINISHED"
