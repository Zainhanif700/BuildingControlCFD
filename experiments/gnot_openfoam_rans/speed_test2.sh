#!/usr/bin/env bash
# Speed test of small changes that should not change the physics. Every variant runs on 1 core, all at the same
# time (so they share the machine equally), with a time profile of the solver. Variants:
#   base    current settings              renum   mesh renumbered (renumberMesh)
#   noco2   OpenFOAM's own CO2 field off  dicgs   pressure smoother DICGaussSeidel
#   yplus   yPlus only computed when it is written (not every time step)
#   direct  pressure solver: direct solve on the coarsest multigrid level
#   both    yplus + direct
# Usage: bash speed_test2.sh [name, default N12] [seconds, default 10] [variants, default: base renum noco2 dicgs]
set -eu
NAME="${1:-N12}"; TEND="${2:-10}"
[ $# -gt 2 ] && shift 2 || set --
VARIANTS="${*:-base renum noco2 dicgs}"
case " $VARIANTS " in *" base "*) ;; *) VARIANTS="base $VARIANTS" ;; esac
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${CFD_PY:-$HOME/anaconda3/envs/cfd/bin/python}"
cd "$HERE"
read -r V DOORS < <(awk -v n="$NAME" '$1 == n { print $3, $4 }' scenarios_v2.txt) || true
[ -n "${V:-}" ] || { echo "$NAME not in scenarios_v2.txt"; exit 1; }
OUT="$HERE/speedtest2/${NAME}_$(echo $VARIANTS | tr ' ' '-')"
rm -rf "$OUT" && mkdir -p "$OUT"
"$PY" make_rans_case.py --V "$V" --doors "$DOORS" --dx 0.08 --name "$NAME" --t-end "$TEND" --write-interval "$TEND" --out "$OUT/new" > "$OUT/make_case.log"
NEW="$OUT/new/${NAME}_dx0.08"
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate foam

change() {   # change <case> <file> <sed expression> <text that must be there afterwards>
    sed -i "$3" "$1/system/$2"
    grep -q "$4" "$1/system/$2" || { echo "could not change $1/system/$2"; exit 1; }
}
for run in $VARIANTS; do
    CASE="$OUT/$run"
    cp -r "$NEW" "$CASE"
    printf '\nprofiling { active true; cpuInfo false; memInfo false; sysInfo false; }\n' >> "$CASE/system/controlDict"
    case "$run" in
        base|renum) ;;
        noco2)  change "$CASE" controlDict 's/^\( *\)type            scalarTransport;/&\n\1enabled         false;/' "enabled         false;" ;;
        dicgs)  change "$CASE" fvSolution 's/smoother GaussSeidel;/smoother DICGaussSeidel;/' "DICGaussSeidel" ;;
        yplus)  change "$CASE" controlDict '/yPlus {/s/writeControl writeTime;/executeControl writeTime; writeControl writeTime;/' "executeControl writeTime" ;;
        direct) change "$CASE" fvSolution 's/smoother GaussSeidel;/smoother GaussSeidel; directSolveCoarsest true;/' "directSolveCoarsest true" ;;
        both)   change "$CASE" controlDict '/yPlus {/s/writeControl writeTime;/executeControl writeTime; writeControl writeTime;/' "executeControl writeTime"
                change "$CASE" fvSolution 's/smoother GaussSeidel;/smoother GaussSeidel; directSolveCoarsest true;/' "directSolveCoarsest true" ;;
        *)      echo "unknown variant $run"; exit 1 ;;
    esac
    RENUMBER=$([ "$run" = renum ] && echo 1 || echo 0) bash run_rans_case.sh "$CASE" > "$CASE/log.run" 2>&1 &
    sleep 5
done
echo "[$(date '+%F %T')] started: $VARIANTS"
wait || true
for run in $VARIANTS; do
    grep -q "^End" "$OUT/$run/log.pimpleFoam" 2> /dev/null || echo "  $run FAILED -- see $OUT/$run/log.run"
done
"$PY" compare_runs.py "$OUT" base | tee "$OUT/result.txt"
echo "time profile (s): total / pressure solver / coarsest level / function objects"
for run in $VARIANTS; do
    P=$(ls -d "$OUT/$run"/[0-9]*/uniform/profiling* 2> /dev/null | tail -1)
    [ -n "$P" ] || continue
    zcat -f "$P" | grep -E "description|totalTime" | paste - - | sed 's/[";]//g' | awk -v r="$run" '
        $2 == "application::main" { m = $NF } $2 == "fvMatrix::solve.p" { p = $NF }
        $2 == "lduMatrix::solver.coarsestLevelCorr" { c = $NF } $2 == "functionObjects.execute()" { f = $NF }
        END { printf "  %-8s %7.0f %7.0f %7.0f %7.0f\n", r, m, p, c, f }'
done
