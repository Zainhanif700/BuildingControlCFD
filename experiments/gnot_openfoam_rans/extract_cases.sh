#!/usr/bin/env bash
# Extract every finished OpenFOAM case in cases/ (marked 'done') that has no dataset file yet, one after the other (GPU).
# Used after OpenFOAM ran with SKIP_EXTRACT=1, e.g. on a machine without GPU.
# Usage: bash extract_cases.sh [data folder, default data_v2] [case pattern, default N*_dx0.08]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${CFD_PY:-$HOME/anaconda3/envs/cfd/bin/python}"
DATA="${1:-data_v2}"; case "$DATA" in /*) ;; *) DATA="$HERE/$DATA" ;; esac
PATTERN="${2:-N*_dx0.08}"
mkdir -p "$DATA" "$HERE/logs/cases"
for CASE in "$HERE"/cases/$PATTERN; do
    [ -f "$CASE/done" ] || continue
    NAME="$(basename "$CASE")"; NAME="${NAME%_dx*}"
    [ -f "$DATA/$NAME.npz" ] && { echo "$NAME: already extracted -- skipped"; continue; }
    if "$PY" "$HERE/extract_rans.py" --case "$CASE" --name "$NAME" --out "$DATA" > "$HERE/logs/cases/${NAME}_extract.log" 2>&1; then
        grep -E "check at|saved" "$HERE/logs/cases/${NAME}_extract.log" | sed "s/^/$NAME: /"
        [ "${KEEP_CASES:-0}" = 1 ] || rm -rf "$CASE"
    else
        echo "$NAME: FAILED -- see logs/cases/${NAME}_extract.log"
    fi
done
