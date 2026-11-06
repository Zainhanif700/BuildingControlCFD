#!/usr/bin/env bash
# On iai-19: every few minutes, copy finished 1-s cases from the other machines (only U, nut, s, cell centres)
# and build the data_v3 file of every finished case (local ones too). Stops when all 28 files exist.
# Usage: SAMPLING_CASES=<local cases folder> V2_DIR=<data_v2 folder> V3_OUT=<data_v3 folder> CFD_PY=<python> \
#        bash collect_and_build.sh <hosts file> [interval s]
# hosts file (private, never in git): one line per machine, e.g. root@<address>
set -uo pipefail
HOSTS="${1:?usage: bash collect_and_build.sh <hosts file> [interval s]}"; EVERY="${2:-600}"
HERE="$(cd "$(dirname "$0")" && pwd)"
RANS="$(cd "$HERE/../gnot_openfoam_rans" && pwd)"
PY="$(command -v "${CFD_PY:-python3}")"
CASES="${SAMPLING_CASES:?set SAMPLING_CASES to the local cases folder}"
V2="${V2_DIR:?set V2_DIR to the data_v2 folder}"
OUT="${V3_OUT:?set V3_OUT to the data_v3 folder}"
[ -f "$V2/N01.npz" ] || { echo "$V2/N01.npz not found"; exit 1; }
REMOTE="BuildingControlCFD/experiments/co2_sampling/cases"
NAMES="$(awk '!/^#/ && NF && $2 == "test" {print $1}' "$RANS/scenarios_v2.txt") $(awk '!/^#/ && NF && $2 != "test" {print $1}' "$RANS/scenarios_v2.txt")"
N_ALL=$(echo $NAMES | wc -w)
mkdir -p "$OUT"
while true; do
    while read -r H; do
        case "$H" in ""|\#*) continue ;; esac
        for d in $(ssh -n -o BatchMode=yes -o ConnectTimeout=20 "$H" "cd $REMOTE 2>/dev/null && ls -d *ws1_dx0.08/done 2>/dev/null" | sed 's|/done$||'); do
            [ -f "$CASES/$d/done" ] && continue
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
    for n in $NAMES; do
        c="$CASES/${n}ws1_dx0.08"
        { [ -f "$c/done" ] && [ ! -f "$OUT/$n.npz" ]; } || continue
        echo "[$(date '+%F %T')] build $n"
        "$PY" "$HERE/build_v3.py" --name "$n" --case "$c" --v2 "$V2/$n.npz" --out "$OUT" \
            || echo "[$(date '+%F %T')]   build of $n FAILED"
    done
    n_done=$(ls "$OUT"/N*.npz 2>/dev/null | wc -l)
    echo "[$(date '+%F %T')] data_v3: $n_done of $N_ALL"
    [ "$n_done" -ge "$N_ALL" ] && break
    sleep "$EVERY"
done
echo "[$(date '+%F %T')] FINISHED"
