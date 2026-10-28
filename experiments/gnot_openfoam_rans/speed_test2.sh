#!/usr/bin/env bash
# Second speed test: small changes that should not change the physics, each on 1 core, all at the same time
# (so they share the machine equally):
#   base     current settings (with a time profile of the solver)
#   renum    mesh renumbered for better memory access (renumberMesh)
#   noco2    OpenFOAM's own CO2 field switched off (we compute the CO2 on the GPU anyway)
#   dicgs    pressure solver with the DICGaussSeidel smoother
# Usage: bash speed_test2.sh [name, default N12] [seconds, default 10]
set -eu
NAME="${1:-N12}"; TEND="${2:-10}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${CFD_PY:-$HOME/anaconda3/envs/cfd/bin/python}"
cd "$HERE"
read -r V DOORS < <(awk -v n="$NAME" '$1 == n { print $3, $4 }' scenarios_v2.txt) || true
[ -n "${V:-}" ] || { echo "$NAME not in scenarios_v2.txt"; exit 1; }
OUT="$HERE/speedtest2/$NAME"
rm -rf "$OUT" && mkdir -p "$OUT"
"$PY" make_rans_case.py --V "$V" --doors "$DOORS" --dx 0.08 --name "$NAME" --t-end "$TEND" --write-interval "$TEND" --out "$OUT/new" > "$OUT/make_case.log"
NEW="$OUT/new/${NAME}_dx0.08"
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate foam

for run in base renum noco2 dicgs; do
    CASE="$OUT/$run"
    cp -r "$NEW" "$CASE"
    case "$run" in
        base)  printf '\nprofiling { active true; cpuInfo false; memInfo false; sysInfo false; }\n' >> "$CASE/system/controlDict" ;;
        noco2) sed -i 's/^\( *\)type            scalarTransport;/&\n\1enabled         false;/' "$CASE/system/controlDict"
               grep -q "enabled         false;" "$CASE/system/controlDict" || { echo "could not switch off co2"; exit 1; } ;;
        dicgs) sed -i 's/p      { solver GAMG; smoother GaussSeidel;/p      { solver GAMG; smoother DICGaussSeidel;/' "$CASE/system/fvSolution"
               grep -q "DICGaussSeidel" "$CASE/system/fvSolution" || { echo "could not change the smoother"; exit 1; } ;;
    esac
    RENUMBER=$([ "$run" = renum ] && echo 1 || echo 0) bash run_rans_case.sh "$CASE" > "$CASE/log.run" 2>&1 &
    sleep 5
done
echo "[$(date '+%F %T')] 4 runs started"
wait || true
for run in base renum noco2 dicgs; do
    grep -q "^End" "$OUT/$run/log.pimpleFoam" 2> /dev/null || echo "  $run FAILED -- see $OUT/$run/log.run"
done
"$PY" compare_runs.py "$OUT" base | tee "$OUT/result.txt"
echo "time profile of the base run (largest parts):"
P=$(ls -d "$OUT"/base/[0-9]*/uniform/profiling* 2> /dev/null | tail -1)
[ -n "$P" ] && zcat -f "$P" | grep -E "description|totalTime" | paste - - | sed 's/[";]//g' \
    | awk '{ printf "%10.0f s  %s\n", $NF, $2 }' | sort -gr | head -12
