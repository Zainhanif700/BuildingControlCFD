#!/usr/bin/env bash
# One window setting end to end for the laminar dataset: case, OpenFOAM run, extraction, delete the case.
# Usage: bash run_one.sh <name> <V1,...,V8>
set -u
NAME="${1:?usage: run_one.sh <name> <V1,...,V8>}"
V="${2:?usage: run_one.sh <name> <V1,...,V8>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${CFD_PY:-$HOME/anaconda3/envs/cfd/bin/python}"
CASES="$HERE/cases"; LOGS="$HERE/logs/cases"
mkdir -p "$CASES" "$LOGS" "$HERE/data"
[ -f "$HERE/data/$NAME.npz" ] && { echo "$NAME: already extracted -- skipped"; exit 0; }
LOG="$LOGS/$NAME.log"
CASE="$CASES/${NAME}_dx0.1"
if (
    set -e
    echo "[$(date '+%F %T')] $NAME: V = $V"
    cd "$HERE/../gnot/openfoam"
    "$PY" make_openfoam_case.py --V "$V" --name "$NAME" --dx 0.1 --t-end 180 --write-interval 10 --out "$CASES"
    ( source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate foam && bash run_openfoam_case.sh "$CASE" 1 )
    cd "$HERE"
    "$PY" extract_case.py --case "$CASE" --name "$NAME" --t-co2 1800 --fv torch
    [ "${KEEP_CASES:-0}" = 1 ] || rm -rf "$CASE"
    echo "[$(date '+%F %T')] $NAME: done"
) > "$LOG" 2>&1; then
    grep -E "steadiness|saved" "$LOG" | sed "s/^/$NAME: /"
else
    echo "$NAME: FAILED -- see $LOG"; exit 1
fi
