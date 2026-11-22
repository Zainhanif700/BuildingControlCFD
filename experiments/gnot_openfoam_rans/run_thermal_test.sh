#!/bin/bash
# Quick test of the thermal (buoyancy) case, only 20 s of flow by default: does it run, does the heat source work,
# how long does it take (compare: the normal quick test, T_END=20 on 4 cores, README_batch.md says about 15 min).
# Usage (from the python environment that has numpy, e.g. cfd):  bash run_thermal_test.sh [name N01] [cores 4] [t_end 20]
# Result: cases_thermal/<name>_thermal_dx0.08.log and the summary printed at the end (also in thermal_test_<name>.txt).
set -euo pipefail
NAME="${1:-N01}"; NP="${2:-4}"; TEND="${3:-20}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$(command -v "${CFD_PY:-python3}")"
read -r V DOORS < <(awk -v n="$NAME" '$1 == n { print $3, $4 }' "$HERE/scenarios_v4.txt")
[ -n "${V:-}" ] || { echo "$NAME not in scenarios_v4.txt"; exit 1; }
OUT="$HERE/cases_thermal"
CASE="$OUT/${NAME}_thermal_dx0.08"
rm -rf "$CASE"
cd "$HERE"
"$PY" make_rans_case_thermal.py --V "$V" --doors "$DOORS" --name "${NAME}_thermal" --dx 0.08 \
    --t-end "$TEND" --write-interval 5 --out "$OUT"
START=$(date +%s)
FOAM_NP="$NP" bash run_rans_case.sh "$CASE" > "$CASE.log" 2>&1 || { echo "run failed -- see $CASE.log and the newest log.* in $CASE"; exit 1; }
WALL=$(( $(date +%s) - START ))
LOG="$CASE/log.buoyantBoussinesqPimpleFoam"
{
echo "THERMAL TEST $NAME ($V, doors $DOORS), $NP cores, flow time $TEND s"
echo "wall time of mesh + run: $WALL s ($(( WALL / 60 )) min)"
grep -E "cells:" "$CASE/log.checkMesh" | head -1 || true
echo "-- last solver time and Courant number:"
grep -E "^Time =" "$LOG" | tail -1
grep -E "Courant Number" "$LOG" | tail -1 || true
grep -E "ExecutionTime" "$LOG" | tail -1 || true
echo "-- was the heat source read? (lines with heatPeople / fvOption in the solver log):"
grep -i -E "heatPeople|fvOption|scalarSemiImplicitSource" "$LOG" | head -5 || echo "NOTHING FOUND: the heat source may not be used"
echo "-- temperature [K] and CO2 (volume average): time, T, s  (seating box, then whole room)"
for f in seatAverage roomAverage; do
    echo "$f:"; cat "$CASE"/postProcessing/$f/*/volFieldValue.dat 2>/dev/null | grep -v '^#' | sed -n '1p;$p' || echo "  no output file"
done
echo "-- fields at the last time:"
ls "$CASE" | grep -E '^[0-9.]+$' | sort -g | tail -2
echo "(a working heat source: T in the seating box is clearly above T of the whole room at the end;"
echo " expected: starts at 293.15 K, the cold window air (273.15 K) lowers the room average)"
} 2>&1 | tee "$HERE/thermal_test_$NAME.txt"
