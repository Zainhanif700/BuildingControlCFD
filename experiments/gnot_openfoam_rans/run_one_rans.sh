#!/usr/bin/env bash
# One window setting end to end: write the case, run OpenFOAM, extract the dataset file, delete the case.
# Usage: bash run_one_rans.sh <name> <V1,...,V8> [doors, default 1,1] [dx, default 0.1]
#   DATA_SUBDIR=data_v2 -> dataset folder;  SKIP_EXTRACT=1 -> only run OpenFOAM and keep the case (no GPU needed)
set -u
NAME="${1:?usage: run_one_rans.sh <name> <V1,...,V8> [doors] [dx]}"
V="${2:?usage: run_one_rans.sh <name> <V1,...,V8> [doors] [dx]}"
DOORS="${3:-1,1}"
DX="${4:-0.1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${CFD_PY:-$HOME/anaconda3/envs/cfd/bin/python}"
DATA="$HERE/${DATA_SUBDIR:-data}"
mkdir -p "$HERE/cases" "$HERE/logs/cases" "$DATA"
[ -f "$DATA/$NAME.npz" ] && { echo "$NAME: already extracted -- skipped"; exit 0; }
LOG="$HERE/logs/cases/$NAME.log"
CASE="$HERE/cases/${NAME}_dx${DX}"
[ "${SKIP_EXTRACT:-0}" = 1 ] && [ -f "$CASE/done" ] && { echo "$NAME: OpenFOAM already done -- skipped"; exit 0; }
if (
    set -e
    echo "[$(date '+%F %T')] $NAME: V = $V, doors = $DOORS, dx = $DX"
    cd "$HERE"
    "$PY" make_rans_case.py --V "$V" --doors "$DOORS" --dx "$DX" --name "$NAME" --t-end 600 --write-interval 10
    ( source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate foam && bash run_rans_case.sh "$CASE" )
    touch "$CASE/done"
    if [ "${SKIP_EXTRACT:-0}" = 1 ]; then
        echo "[$(date '+%F %T')] $NAME: OpenFOAM done, case kept for extraction (saved)"
        exit 0
    fi
    "$PY" extract_rans.py --case "$CASE" --name "$NAME" --out "$DATA"
    [ "${KEEP_CASES:-0}" = 1 ] || rm -rf "$CASE"
    echo "[$(date '+%F %T')] $NAME: done"
) > "$LOG" 2>&1; then
    grep -E "fluctuation|check at|saved" "$LOG" | sed "s/^/$NAME: /"
else
    echo "$NAME: FAILED -- see $LOG"; exit 1
fi
