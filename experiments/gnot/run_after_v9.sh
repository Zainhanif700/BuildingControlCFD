#!/usr/bin/env bash
# TEMPORARY overnight helper (2026-09-25). DELETE after use:
#     git rm experiments/gnot/run_after_v9.sh
#
# What it does, in order (every step is logged to run_after_v9.log):
#   1. waits until the running v9 training process exits
#   2. checks v9 finished normally (final checkpoint exists) -- otherwise STOPS
#   3. runs the v9 diagnostics with the FROZEN v9 code in milestones/v9_zeroflow_bc/
#      -> milestones/v9_zeroflow_bc/v9_final_diagnostics.log
#   4. copies v9's final checkpoint into that milestone folder
#   5. runs the v10 smoke test with the live code -- if it fails, STOPS (no v10)
#   6. launches v10 training -> train_gnot_v10.log
#
# Usage (in its own tmux session, after `git pull`):
#   tmux new -s after_v9
#   cd ~/BuildingControlCFD/experiments/gnot && bash run_after_v9.sh

set -u
GNOT="$HOME/BuildingControlCFD/experiments/gnot"
MS="$GNOT/milestones/v9_zeroflow_bc"
CK="$GNOT/checkpoints/v9_zeroflow_bc"
R="../../checkpoints/v9_zeroflow_bc"          # same folder, relative to $MS
LOG="$GNOT/run_after_v9.log"
DIAG="$MS/v9_final_diagnostics.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

cd "$GNOT" || exit 1

# sanity checks BEFORE waiting hours: live code must already be v10, frozen v9 must exist
if ! grep -q '^VERSION = "v10_hardic"' train_gnot.py; then
    log "ABORT: live train_gnot.py is not v10_hardic -- run 'git pull myfork main' first."; exit 1
fi
if [ ! -f "$MS/probe_co2_time.py" ]; then
    log "ABORT: $MS is missing -- run 'git pull myfork main' first."; exit 1
fi

# 1. wait for v9
PID=$(pgrep -f "python3 -u train_gnot.py" | head -n 1)
if [ -n "$PID" ]; then
    log "waiting for v9 training (PID $PID) to finish -- checking every 60 s ..."
    while kill -0 "$PID" 2>/dev/null; do sleep 60; done
    log "v9 process has exited."
else
    log "no running train_gnot.py found -- assuming v9 has already finished."
fi
sleep 30

# 2. did v9 finish normally?
if [ ! -f "$CK/gnot_v9_zeroflow_bc_final.pth" ]; then
    log "ABORT: $CK/gnot_v9_zeroflow_bc_final.pth not found -- v9 did NOT finish normally."
    log "       Not starting v10. Check train_gnot_v9.log."
    exit 1
fi
log "v9 finished normally (final checkpoint found)."

# 3. v9 diagnostics with the frozen v9 code
log "running v9 diagnostics -> $DIAG"
cd "$MS" || exit 1
{
    echo "=== v9_zeroflow_bc final diagnostics, $(date) ==="
    echo; echo "--- co2_residual_breakdown (final) ---"
    python3 co2_residual_breakdown.py "$R/gnot_v9_zeroflow_bc_final.pth"
    echo; echo "--- probe_co2_time (iter10000, iter15000, final) ---"
    python3 probe_co2_time.py "$R/gnot_v9_zeroflow_bc_iter10000.pth" "$R/gnot_v9_zeroflow_bc_iter15000.pth" "$R/gnot_v9_zeroflow_bc_final.pth"
    echo; echo "--- closed_window_diagnostic (final, closed) ---"
    python3 closed_window_diagnostic.py "$R/gnot_v9_zeroflow_bc_final.pth" closed
    echo; echo "--- closed_window_diagnostic (final, open) ---"
    python3 closed_window_diagnostic.py "$R/gnot_v9_zeroflow_bc_final.pth" open
    echo; echo "--- probe_source_co2 (iter5000 ... final) ---"
    python3 probe_source_co2.py "$R/gnot_v9_zeroflow_bc_iter5000.pth" "$R/gnot_v9_zeroflow_bc_iter10000.pth" "$R/gnot_v9_zeroflow_bc_iter15000.pth" "$R/gnot_v9_zeroflow_bc_final.pth"
} > "$DIAG" 2>&1
log "v9 diagnostics written (errors, if any, are inside that log)."
# the frozen train_gnot.py creates an empty checkpoints/ dir next to itself on import -- remove it
rmdir "$MS/checkpoints/v9_zeroflow_bc" "$MS/checkpoints" 2>/dev/null || true

# 4. keep the final v9 checkpoint with its milestone
cp "$CK/gnot_v9_zeroflow_bc_final.pth" "$MS/" && log "copied v9 final checkpoint into $MS"

# 5. v10 smoke test (live code)
cd "$GNOT" || exit 1
log "running v10 smoke test -> smoke_test_v10.log"
if python3 staged_smoke_test.py > smoke_test_v10.log 2>&1; then
    log "v10 smoke test PASSED."
else
    log "ABORT: v10 smoke test FAILED -- see smoke_test_v10.log. NOT starting v10."
    exit 1
fi

# 6. launch v10
log "launching v10 training -> train_gnot_v10.log"
python3 -u train_gnot.py 2>&1 | tee train_gnot_v10.log
log "v10 training process ended."
