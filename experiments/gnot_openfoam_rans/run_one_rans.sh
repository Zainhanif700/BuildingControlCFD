#!/usr/bin/env bash
# One window setting end to end: write the case, run OpenFOAM, extract the dataset file, delete the case.
# Usage: bash run_one_rans.sh <name> <V1,...,V8>
set -u
NAME="${1:?usage: run_one_rans.sh <name> <V1,...,V8>}"
V="${2:?usage: run_one_rans.sh <name> <V1,...,V8>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${CFD_PY:-$HOME/anaconda3/envs/cfd/bin/python}"
mkdir -p "$HERE/cases" "$HERE/logs/cases" "$HERE/data"
[ -f "$HERE/data/$NAME.npz" ] && { echo "$NAME: already extracted -- skipped"; exit 0; }
LOG="$HERE/logs/cases/$NAME.log"
CASE="$HERE/cases/${NAME}_dx0.1"
if (
    set -e
    echo "[$(date '+%F %T')] $NAME: V = $V"
    cd "$HERE"
    "$PY" make_rans_case.py --V "$V" --name "$NAME" --t-end 600 --write-interval 10
    ( source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate foam && bash run_rans_case.sh "$CASE" )
    "$PY" extract_rans.py --case "$CASE" --name "$NAME"
    [ "${KEEP_CASES:-0}" = 1 ] || rm -rf "$CASE"
    echo "[$(date '+%F %T')] $NAME: done"
) > "$LOG" 2>&1; then
    grep -E "fluctuation|check at|saved" "$LOG" | sed "s/^/$NAME: /"
else
    echo "$NAME: FAILED -- see $LOG"; exit 1
fi
