#!/usr/bin/env bash
# Speed test on one window setting: current solver settings vs. relaxed tolerances, on 1 core and on NP cores.
# Every run simulates the same T seconds; the results are compared with the current settings on 1 core.
# Usage: bash speed_test.sh [name, default N12] [seconds, default 10] [cores, default 4]
set -eu
NAME="${1:-N12}"; TEND="${2:-10}"; NP="${3:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${CFD_PY:-$HOME/anaconda3/envs/cfd/bin/python}"
cd "$HERE"
read -r V DOORS < <(awk -v n="$NAME" '$1 == n { print $3, $4 }' scenarios_v2.txt) || true
[ -n "${V:-}" ] || { echo "$NAME not in scenarios_v2.txt"; exit 1; }
OUT="$HERE/speedtest/$NAME"
rm -rf "$OUT" && mkdir -p "$OUT"
"$PY" make_rans_case.py --V "$V" --doors "$DOORS" --dx 0.08 --name "$NAME" --t-end "$TEND" --write-interval "$TEND" --out "$OUT/new" > "$OUT/make_case.log"
NEW="$OUT/new/${NAME}_dx0.08"
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate foam

relax() {
    sed -i -e 's/tolerance 1e-7; relTol 0.01;/tolerance 1e-6; relTol 0.05;/' \
           -e 's/tolerance 1e-8; relTol 0;/tolerance 1e-6; relTol 0;/' "$1/system/fvSolution"
    [ "$(grep -c 'tolerance 1e-6' "$1/system/fvSolution")" = 2 ] || { echo "could not change $1/system/fvSolution"; exit 1; }
}

for run in current_1 relaxed_1 current_$NP relaxed_$NP; do
    CASE="$OUT/$run"
    cp -r "$NEW" "$CASE"
    case "$run" in relaxed_*) relax "$CASE" ;; esac
    echo "[$(date '+%F %T')] $run"
    FOAM_NP="${run##*_}" bash run_rans_case.sh "$CASE" > "$CASE/log.run" 2>&1 || { echo "$run FAILED -- see $CASE/log.run"; exit 1; }
    grep "^ExecutionTime" "$CASE/log.pimpleFoam" | tail -1 | awk -v r="$run" '{ printf "  %-12s %8.0f s\n", r, $7 }' | tee -a "$OUT/times.txt"
done
"$PY" compare_runs.py "$OUT" current_1 | tee "$OUT/result.txt"
