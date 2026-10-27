#!/usr/bin/env bash
# Runs one setting of scenarios_v2.txt completely (600 s) with a larger time step (maxCo), makes its dataset file
# and compares it with the existing one from the normal time step (data_v2/<name>.npz).
# Usage: bash check_timestep.sh [name, default N12] [maxCo, default 2] [cores, default 4]
set -euo pipefail
NAME="${1:-N12}"; CO="${2:-2}"; NP="${3:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${CFD_PY:-$HOME/anaconda3/envs/cfd/bin/python}"
cd "$HERE"
[ -f "data_v2/$NAME.npz" ] || { echo "data_v2/$NAME.npz missing"; exit 1; }
read -r V DOORS < <(awk -v n="$NAME" '$1 == n { print $3, $4 }' scenarios_v2.txt) || true
[ -n "${V:-}" ] || { echo "$NAME not in scenarios_v2.txt"; exit 1; }
TAG="${NAME}co${CO//./p}"
CASE="$HERE/cases/${TAG}_dx0.08"
mkdir -p data_check
echo "[$(date '+%F %T')] $TAG: $NAME with maxCo $CO on $NP cores"
if [ ! -f "$CASE/done" ]; then
    rm -rf "$CASE"
    "$PY" make_rans_case.py --V "$V" --doors "$DOORS" --dx 0.08 --name "$TAG" --t-end 600 --write-interval 10 > /dev/null
    sed -i "s/^maxCo .*/maxCo           $CO;/" "$CASE/system/controlDict"
    grep -q "^maxCo           $CO;" "$CASE/system/controlDict" || { echo "could not set maxCo"; exit 1; }
    ( source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate foam && FOAM_NP="$NP" bash run_rans_case.sh "$CASE" ) > "$CASE.log" 2>&1
    grep -q "^End" "$CASE/log.pimpleFoam" || { echo "OpenFOAM did not reach the end -- see $CASE.log"; exit 1; }
    touch "$CASE/done"
    echo "[$(date '+%F %T')] OpenFOAM done ($(grep '^ExecutionTime' "$CASE/log.pimpleFoam" | tail -1 | awk '{print $7}') s)"
fi
[ -f "data_check/$TAG.npz" ] || "$PY" extract_rans.py --case "$CASE" --name "$TAG" --out data_check | grep -E "check at|saved"
"$PY" compare_datasets.py "data_v2/$NAME.npz" "data_check/$TAG.npz"
