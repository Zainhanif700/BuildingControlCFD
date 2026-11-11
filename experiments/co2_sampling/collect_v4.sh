#!/usr/bin/env bash
# On iai-19, for the all-openings settings (names starting with A in the scenario file): every few minutes
# 1. copy finished cases from the other machines (only U, nut, s and the cell centres),
# 2. build the dataset file of every finished case (build_v4.py) and, with DELETE_AFTER_BUILD=1, delete the local
#    case afterwards (the dataset file keeps the flow, so the transitions do not need the case),
# 3. build the transition of a new setting as soon as its own and its partner's dataset files exist.
# Stops when every new setting has its transition file.
# Usage: SAMPLING_CASES=<local cases> V4_OUT=<data_v4> TRANS_OUT=<data_transitions_v4> CFD_PY=<python> \
#        [DELETE_AFTER_BUILD=1] bash collect_v4.sh <hosts file or /dev/null> [interval s]
set -uo pipefail
HOSTS="${1:?usage: bash collect_v4.sh <hosts file> [interval s]}"; EVERY="${2:-600}"
HERE="$(cd "$(dirname "$0")" && pwd)"
RANS="$(cd "$HERE/../gnot_openfoam_rans" && pwd)"
PY="$(command -v "${CFD_PY:-python3}")"
CASES="${SAMPLING_CASES:?set SAMPLING_CASES}"
OUT="${V4_OUT:?set V4_OUT}"
TOUT="${TRANS_OUT:?set TRANS_OUT}"
SCEN="${SAMPLING_SCEN:-$RANS/scenarios_v4.txt}"
REMOTE="BuildingControlCFD/experiments/co2_sampling/cases"
NEW="$(awk '!/^#/ && NF && $1 ~ /^A/ && $2 == "test" {print $1}' "$SCEN") $(awk '!/^#/ && NF && $1 ~ /^A/ && $2 != "test" {print $1}' "$SCEN")"
N_NEW=$(echo $NEW | wc -w)
N_ALL=$(awk '!/^#/ && NF' "$SCEN" | wc -l)
mkdir -p "$OUT" "$TOUT"
PAIRS="$(cd "$HERE" && "$PY" -c "
import sys
from build_v4 import scenario_names, partner
names = scenario_names(sys.argv[1])
print('\n'.join(n + ' ' + partner(n, names) for n in names))
" "$SCEN")" || { echo "could not compute the transition partners"; exit 1; }
partner_of() { echo "$PAIRS" | awk -v n="$1" '$1 == n {print $2}'; }
echo "[$(date '+%F %T')] $N_NEW new settings, $N_ALL in total; dataset files now: $(ls "$OUT"/*.npz 2>/dev/null | wc -l)"
while true; do
    while read -r H; do
        case "$H" in ""|\#*) continue ;; esac
        for d in $(ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$H" "cd $REMOTE 2>/dev/null && ls -d A*ws1_dx0.08/done 2>/dev/null" | sed 's|/done$||'); do
            n="${d%%ws1_dx0.08}"
            { [ -f "$CASES/$d/done" ] || [ -f "$OUT/$n.npz" ]; } && continue
            echo "[$(date '+%F %T')] copy $d from $H"
            if rsync -a -m --include='/scenario.txt' --include='/0/' --include='/0/C*' --include='/[1-9]*/' \
                    --include='/[1-9]*/U' --include='/[1-9]*/U.gz' --include='/[1-9]*/nut' --include='/[1-9]*/nut.gz' \
                    --include='/[1-9]*/s' --include='/[1-9]*/s.gz' --exclude='*' "$H:$REMOTE/$d/" "$CASES/$d/" < /dev/null; then
                touch "$CASES/$d/done"
                echo "[$(date '+%F %T')]   copied, $(du -sh "$CASES/$d" | cut -f1)"
            else
                echo "[$(date '+%F %T')]   copy FAILED, will retry"
            fi
        done
    done < "$HOSTS"
    for n in $NEW; do
        c="$CASES/${n}ws1_dx0.08"
        { [ -f "$c/done" ] && [ ! -f "$OUT/$n.npz" ]; } || continue
        echo "[$(date '+%F %T')] build $n (dataset)"
        if "$PY" "$HERE/build_v4.py" --name "$n" --case "$c" --out "$OUT" --scen "$SCEN" --trans-out "$TOUT" --no-transition \
                && [ -s "$OUT/$n.npz" ]; then
            [ "${DELETE_AFTER_BUILD:-0}" = 1 ] && rm -rf "$c" "$c.log" && echo "[$(date '+%F %T')]   local case deleted"
        else
            echo "[$(date '+%F %T')]   build of $n FAILED"
        fi
    done
    for n in $NEW; do
        a="$(partner_of "$n")"
        { [ ! -f "$TOUT/$n.npz" ] && [ -f "$OUT/$n.npz" ] && [ -n "$a" ] && [ -f "$OUT/$a.npz" ]; } || continue
        echo "[$(date '+%F %T')] build $n (transition, start from $a)"
        "$PY" "$HERE/build_v4.py" --name "$n" --out "$OUT" --scen "$SCEN" --trans-out "$TOUT" --only-transition \
            || echo "[$(date '+%F %T')]   transition of $n FAILED"
    done
    n_ds=$(ls "$OUT"/*.npz 2>/dev/null | wc -l)
    n_tr=$(for n in $NEW; do [ -f "$TOUT/$n.npz" ] && echo x; done | wc -l)
    echo "[$(date '+%F %T')] dataset files: $n_ds of $N_ALL, new transitions: $n_tr of $N_NEW"
    [ "$n_tr" -ge "$N_NEW" ] && break
    sleep "$EVERY"
done
echo "[$(date '+%F %T')] FINISHED"
