#!/usr/bin/env bash
# Continue an OpenFOAM case that stopped early (e.g. the server restarted) from its last save, and mark it 'done'.
# The dataset file is made afterwards on the GPU machine with extract_cases.sh.
# Usage: bash resume_rans.sh <name> [dx, default 0.08]      e.g. bash resume_rans.sh N23
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="${1:?usage: bash resume_rans.sh <name> [dx]}"
CASE="$HERE/cases/${NAME}_dx${2:-0.08}"
LOG="$HERE/logs/cases/$NAME.log"
mkdir -p "$HERE/logs/cases"
[ -d "$CASE" ] || { echo "$NAME: no case folder $CASE"; exit 1; }
[ -f "$CASE/done" ] && { echo "$NAME: OpenFOAM already done -- skipped"; exit 0; }
# every step must succeed (set -e does not work inside 'if ( ... )', so each step ends with '|| exit 1')
if (
    cd "$CASE" || exit 1
    source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate foam || exit 1
    if [ -d processor0 ]; then           # stopped during a run on several cores
        reconstructPar -newTimes > log.reconstructPar 2>&1 || exit 1
        rm -rf processor*
    fi
    LAST=$(ls | grep -E '^[0-9.]+$' | sort -g | tail -1)
    for f in U p phi k omega nut; do
        [ -f "$LAST/$f.gz" ] || [ -f "$LAST/$f" ] || { echo "the save at $LAST s has no $f -- cannot continue"; exit 1; }
    done
    echo "[$(date '+%F %T')] $NAME: continue from the save at $LAST s"
    foamDictionary -entry startFrom -set latestTime system/controlDict > /dev/null || exit 1
    # the old log can end in zeros after a crash; clean it so the status check can read it
    if [ -f log.pimpleFoam ]; then tr -d '\000' < log.pimpleFoam > log.tmp && mv log.tmp log.pimpleFoam || exit 1; fi
    pimpleFoam >> log.pimpleFoam 2>&1 || { echo "pimpleFoam failed -- see $CASE/log.pimpleFoam"; exit 1; }
    tail -n 50 log.pimpleFoam | grep -q "^End" || { echo "OpenFOAM did not reach the end -- see $CASE/log.pimpleFoam"; exit 1; }
    for d in [0-9]*; do                  # same clean-up as run_rans_case.sh: p/phi only at the comparison times
        case "$d" in 0|0.orig|30|60|120|180) ;; *) rm -f "$d"/p "$d"/p.gz "$d"/phi "$d"/phi.gz ;; esac
    done
    touch done || exit 1
    echo "[$(date '+%F %T')] $NAME: OpenFOAM done, case kept for extraction (saved)"
) >> "$LOG" 2>&1; then
    grep -E "continue from|saved" "$LOG" | sed "s/^/$NAME: /"
else
    echo "$NAME: FAILED -- see $LOG"; exit 1
fi
