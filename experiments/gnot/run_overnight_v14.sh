#!/usr/bin/env bash
# TEMPORARY overnight helper (2026-09-27). DELETE after use:
#     git rm experiments/gnot/run_overnight_v14.sh
#
# Order (every step logged to run_overnight_v14.log; any failure STOPS the script):
#   1. check the live code is v14 (uniform sampling) -- i.e. you already pulled
#   2. wait for the running D2 probe (lbfgs_probe.py) to finish
#   3. v14 smoke test -> smoke_test_v14.log   (fails -> no training)
#   4. v14 training   -> train_gnot_v14.log   (~4.7 h)
#   5. validation + D1 re-diagnosis of the v14 final checkpoint -> v14_validation.txt
#
# Usage (after `git pull myfork main`, in its own tmux session):
#   tmux new -s overnight
#   cd ~/BuildingControlCFD/experiments/gnot && bash run_overnight_v14.sh

set -u
GNOT="$HOME/BuildingControlCFD/experiments/gnot"
LOG="$GNOT/run_overnight_v14.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
cd "$GNOT" || exit 1

# 1. sanity checks BEFORE waiting
grep -q '^VERSION = "v14_uniform"' train_gnot.py || { log "ABORT: live train_gnot.py is not v14_uniform -- run 'git pull myfork main' first."; exit 1; }
grep -q '^SOURCE_SAMPLE_FRAC = 0.0' point_sampler.py || { log "ABORT: point_sampler.py is not uniform (SOURCE_SAMPLE_FRAC) -- pull first."; exit 1; }
if [ -d checkpoints/v14_uniform ] && ls checkpoints/v14_uniform/*.pth >/dev/null 2>&1; then
    log "ABORT: checkpoints/v14_uniform already has checkpoints -- not overwriting."; exit 1
fi

# 2. wait for D2 (it keeps running from code it loaded at start, so pulling v14 does not affect it)
PID=$(pgrep -f "python3 -u lbfgs_probe.py" | head -n 1)
if [ -n "$PID" ]; then
    log "waiting for the D2 L-BFGS probe (PID $PID) to finish -- checking every 60 s ..."
    while kill -0 "$PID" 2>/dev/null; do sleep 60; done
    log "D2 finished (its results are in lbfgs_probe.log)."
else
    log "no running lbfgs_probe.py found -- assuming D2 already finished."
fi
sleep 20

# 3. smoke test
log "running v14 smoke test -> smoke_test_v14.log"
if python3 staged_smoke_test.py > smoke_test_v14.log 2>&1; then
    log "v14 smoke test PASSED."
else
    log "ABORT: v14 smoke test FAILED -- see smoke_test_v14.log. NOT starting training."; exit 1
fi

# 4. training
log "launching v14 training -> train_gnot_v14.log"
python3 -u train_gnot.py 2>&1 | tee train_gnot_v14.log
FINAL="checkpoints/v14_uniform/gnot_v14_uniform_final.pth"
if [ ! -f "$FINAL" ]; then
    log "ABORT: $FINAL not found -- v14 did not finish normally. See train_gnot_v14.log."; exit 1
fi
log "v14 training finished."

# 5. validation + re-diagnosis
log "running validation + residual diagnosis -> v14_validation.txt"
{
    echo "=== validate_closed_room.py (v14 final) ==="
    python3 validate_closed_room.py "$FINAL"
    echo
    echo "=== diagnose_residual_map.py (v14 final) ==="
    python3 diagnose_residual_map.py "$FINAL"
} > v14_validation.txt 2>&1
log "ALL DONE -- read v14_validation.txt and lbfgs_probe.log."
