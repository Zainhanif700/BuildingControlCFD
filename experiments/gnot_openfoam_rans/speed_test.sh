#!/usr/bin/env bash
# Speed test on one window setting: current settings vs. relaxed tolerances (relaxed) and twice the time step (maxco2),
# on 1 core and on NP cores. Every run simulates the same T seconds and is compared with the current settings on 1 core.
# Finished runs are kept, so the test can be extended or restarted.
# Usage: bash speed_test.sh [name, default N12] [seconds, default 10] [cores, default 4]
set -eu
NAME="${1:-N12}"; TEND="${2:-10}"; NP="${3:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${CFD_PY:-$HOME/anaconda3/envs/cfd/bin/python}"
cd "$HERE"
read -r V DOORS < <(awk -v n="$NAME" '$1 == n { print $3, $4 }' scenarios_v2.txt) || true
[ -n "${V:-}" ] || { echo "$NAME not in scenarios_v2.txt"; exit 1; }
OUT="$HERE/speedtest/$NAME"
rm -rf "$OUT/new" && mkdir -p "$OUT"
"$PY" make_rans_case.py --V "$V" --doors "$DOORS" --dx 0.08 --name "$NAME" --t-end "$TEND" --write-interval "$TEND" --out "$OUT/new" > "$OUT/make_case.log"
NEW="$OUT/new/${NAME}_dx0.08"
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate foam

relax() {
    sed -i -e 's/tolerance 1e-7; relTol 0.01;/tolerance 1e-6; relTol 0.05;/' \
           -e 's/tolerance 1e-8; relTol 0;/tolerance 1e-6; relTol 0;/' "$1/system/fvSolution"
    [ "$(grep -c 'tolerance 1e-6' "$1/system/fvSolution")" = 2 ] || { echo "could not change $1/system/fvSolution"; exit 1; }
}
maxco2() {
    sed -i 's/^maxCo .*/maxCo           2.0;/' "$1/system/controlDict"
    grep -q "^maxCo           2.0;" "$1/system/controlDict" || { echo "could not change $1/system/controlDict"; exit 1; }
}

for run in current_1 relaxed_1 current_$NP relaxed_$NP maxco2_1 maxco2_$NP; do
    CASE="$OUT/$run"
    if grep -q "^End" "$CASE/log.pimpleFoam" 2> /dev/null; then echo "  $run: done before -- kept"; continue; fi
    rm -rf "$CASE" && cp -r "$NEW" "$CASE"
    case "$run" in relaxed_*) relax "$CASE" ;; maxco2_*) maxco2 "$CASE" ;; esac
    echo "[$(date '+%F %T')] $run"
    if FOAM_NP="${run##*_}" bash run_rans_case.sh "$CASE" > "$CASE/log.run" 2>&1; then
        grep "^ExecutionTime" "$CASE/log.pimpleFoam" | tail -1 | awk -v r="$run" '{ printf "  %-12s %8.0f s\n", r, $7 }'
    else
        echo "  $run FAILED -- see $CASE/log.run"
    fi
done
"$PY" compare_runs.py "$OUT" current_1 | tee "$OUT/result.txt"
