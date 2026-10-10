#!/usr/bin/env bash
# Mesh study: S02 (weak ventilation) and S08 (strong) on 8, 10 and 15 cm meshes, NP runs at a time (default 4); cases are kept.
# Usage: bash run_mesh_study.sh [NP]      then: python3 compare_meshes.py cases/S02_mesh_dx0.15 cases/S02_mesh_dx0.1 cases/S02_mesh_dx0.08
set -u
NP="${1:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${CFD_PY:-$HOME/anaconda3/envs/cfd/bin/python}"
mkdir -p "$HERE/cases" "$HERE/logs/mesh"

one() {   # name V dx
    local NAME="$1" V="$2" DX="$3"
    local CASE="$HERE/cases/${NAME}_mesh_dx${DX}" LOG="$HERE/logs/mesh/${NAME}_dx${DX}.log"
    [ -f "$CASE/done" ] && { echo "$NAME dx $DX: done -- skipped"; return 0; }
    if (
        set -e
        echo "[$(date '+%F %T')] $NAME dx $DX: V = $V"
        cd "$HERE"
        "$PY" make_rans_case.py --V "$V" --name "${NAME}_mesh" --dx "$DX" --t-end 600 --write-interval 10
        ( source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate foam && bash run_rans_case.sh "$CASE" )
        touch "$CASE/done"
        echo "[$(date '+%F %T')] $NAME dx $DX: done"
    ) > "$LOG" 2>&1; then
        echo "$NAME dx $DX: done ($(grep -m1 -oE 'cells: *[0-9]+' "$CASE/log.checkMesh"))"
    else
        echo "$NAME dx $DX: FAILED -- see $LOG"
    fi
}
export -f one
export HERE PY
# the slowest runs first, so the 15 cm runs fill the cores when the 10 cm ones finish
printf '%s\n' "S02 0,0,0,0.5,0.5,0.5,0,0 0.08" "S08 0,0,0,2,2,2,0,0 0.08" \
              "S02 0,0,0,0.5,0.5,0.5,0,0 0.1"  "S08 0,0,0,2,2,2,0,0 0.1" \
              "S02 0,0,0,0.5,0.5,0.5,0,0 0.15" "S08 0,0,0,2,2,2,0,0 0.15" \
    | xargs -P "$NP" -L 1 bash -c 'one $0 $1 $2'
