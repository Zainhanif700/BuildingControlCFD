#!/usr/bin/env bash
# One setting for the all-openings dataset: OpenFOAM 0-150 s with a save every second (the real start-up flow),
# then continued to 600 s with a save every 10 s (for the mean flow). Resumable: a finished part is not run again.
# Usage: SAMPLING_CASES=<cases folder> CFD_PY=<python> bash run_full_case.sh <name> [cores, default 4]
#        SAMPLING_SCEN=<scenario file> (default scenarios_v4.txt); <cases>/<name>ws1.skip: setting runs elsewhere
set -euo pipefail
NAME="${1:?usage: bash run_full_case.sh <name> [cores]}"; NP="${2:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
RANS="$(cd "$HERE/../gnot_openfoam_rans" && pwd)"
PY="$(command -v "${CFD_PY:-python3}")"
SCEN="${SAMPLING_SCEN:-$RANS/scenarios_v4.txt}"
read -r V DOORS < <(awk -v n="$NAME" '$1 == n { print $3, $4 }' "$SCEN") || true
[ -n "${V:-}" ] || { echo "$NAME not in $SCEN"; exit 1; }
CASES="${SAMPLING_CASES:-$HERE/cases}"
CASE="$CASES/${NAME}ws1_dx0.08"
if [ -f "$CASES/${NAME}ws1.skip" ]; then
    echo "[$(date '+%F %T')] $NAME: runs on another machine -- skipped"; exit 0
fi
if [ -f "$CASE/done" ]; then
    echo "[$(date '+%F %T')] $NAME: already done"; exit 0
fi
echo "[$(date '+%F %T')] $NAME (V $V, doors $DOORS): $NP cores, python $PY"
if [ ! -f "$CASE/done150" ]; then
    rm -rf "$CASE"
    mkdir -p "$CASES"
    (cd "$RANS" && "$PY" make_rans_case.py --V="$V" --doors "$DOORS" --dx 0.08 --name "${NAME}ws1" \
        --t-end 150 --write-interval 1 --out "$CASES" > /dev/null)
    FOAM_NP="$NP" bash "$RANS/run_rans_case.sh" "$CASE" > "$CASE.log" 2>&1
    grep -q "^End" "$CASE/log.pimpleFoam" || { echo "OpenFOAM 0-150 s did not reach the end -- see $CASE.log"; exit 1; }
    touch "$CASE/done150"
    echo "[$(date '+%F %T')] $NAME: 0-150 s done ($(du -sh "$CASE" | cut -f1))"
fi
bash "$HERE/continue_case.sh" "$CASE" 150 600 "$NP" 10 >> "$CASE.log" 2>&1 \
    || { echo "OpenFOAM 150-600 s failed -- see $CASE.log and $CASE/log.pimpleFoam.cont"; exit 1; }
n1=$(ls -d "$CASE"/[0-9]* | awk -F/ '{t=$NF} t+0 > 0 && t+0 <= 150' | wc -l)
n2=$(ls -d "$CASE"/[0-9]* | awk -F/ '{t=$NF} t+0 > 150' | wc -l)
[ "$n1" -ge 150 ] && [ "$n2" -ge 45 ] || { echo "$NAME: unexpected saves ($n1 up to 150 s, $n2 after) -- not marked done"; exit 1; }
touch "$CASE/done"
echo "[$(date '+%F %T')] $NAME: done, $n1 saves to 150 s and $n2 after ($(du -sh "$CASE" | cut -f1))"
