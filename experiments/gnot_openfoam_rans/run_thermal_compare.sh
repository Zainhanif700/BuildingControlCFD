#!/bin/bash
# The same scenario as run_thermal_test.sh but WITHOUT temperature (normal pimpleFoam case), then the comparison with
# the thermal case that run_thermal_test.sh made (it must have run before, with the same name and flow time).
# Usage (env cfd):  bash run_thermal_compare.sh [name N01] [cores 4] [t_end 20]
# Result: cases_thermal/<name>_ref_dx0.08, and the summary printed at the end (also in thermal_compare_<name>.txt).
set -euo pipefail
NAME="${1:-N01}"; NP="${2:-4}"; TEND="${3:-20}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$(command -v "${CFD_PY:-python3}")"
read -r V DOORS < <(awk -v n="$NAME" '$1 == n { print $3, $4 }' "$HERE/scenarios_v4.txt")
[ -n "${V:-}" ] || { echo "$NAME not in scenarios_v4.txt"; exit 1; }
OUT="$HERE/cases_thermal"
REF="$OUT/${NAME}_ref_dx0.08"
TH="$OUT/${NAME}_thermal_dx0.08"
[ -d "$TH" ] || { echo "run run_thermal_test.sh $NAME $NP $TEND first ($TH is missing)"; exit 1; }
rm -rf "$REF"
cd "$HERE"
"$PY" make_rans_case.py --V "$V" --doors "$DOORS" --name "${NAME}_ref" --dx 0.08 \
    --t-end "$TEND" --write-interval 5 --out "$OUT"
START=$(date +%s)
FOAM_NP="$NP" bash run_rans_case.sh "$REF" > "$REF.log" 2>&1 || { echo "run failed -- see $REF.log and the newest log.* in $REF"; exit 1; }
WALL=$(( $(date +%s) - START ))
{
echo "NORMAL CASE $NAME ($V, doors $DOORS), $NP cores, flow time $TEND s: wall time of mesh + run $WALL s ($(( WALL / 60 )) min)"
"$PY" compare_thermal.py "$REF" "$TH"
} 2>&1 | tee "$HERE/thermal_compare_$NAME.txt"
