#!/usr/bin/env bash
# Runs EVERY check on one checkpoint, closed AND open windows, and writes one log.
#
#   1. code smoke test (staged_smoke_test.py)            -- is the code itself consistent?
#   2. closed-room validation vs finite-difference ref   -- accuracy where a reference exists
#   3. residual diagnosis D1 (closed room)               -- where the remaining error comes from
#   4. window-BC cross-check                             -- inflow right for every window setting?
#   5. level-3 physics consistency, open + closed        -- air balance, leakage, CO2 budget,
#                                                           PDE residuals over a window sweep
#   6. level-2 CO2 check                                 -- model CO2 vs an independent FV CO2
#                                                           solve driven by the model's own flow
#
# Usage (from experiments/gnot):
#   bash post_training_checks.sh checkpoints/v16_fixes/gnot_v16_fixes_final.pth          # on CPU
#   bash post_training_checks.sh checkpoints/v16_fixes/gnot_v16_fixes_iter5000.pth       # mid-run
#   bash post_training_checks.sh <checkpoint> --gpu      # only when no training is running
#
# Results: checks_<checkpoint name>.log, figures in figures/<version>/ (for an iterN
# checkpoint they are moved to figures/<version>_iterN/ so the final run never mixes with them).
set -u
CKPT="${1:?usage: bash post_training_checks.sh <checkpoint> [--gpu]}"
[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT"; exit 1; }
if [ "${2:-}" = "--gpu" ]; then DEV=cuda; else DEV=cpu; export CUDA_VISIBLE_DEVICES=""; fi
NAME=$(basename "$CKPT" .pth)
[ -f staged_smoke_test.py ] || { echo "run this from experiments/gnot"; exit 1; }
case "$CKPT" in *"'"*) echo "checkpoint path must not contain a quote"; exit 1 ;; esac
VERSION=$(python3 -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu').get('version','unknown'))" "$CKPT")
[ -n "$VERSION" ] && [ "$VERSION" != unknown ] || { echo "could not read the version from $CKPT"; exit 1; }
LOG="checks_${NAME}.log"
START=$(date +%s)

run() {  # run <title> <command...>: print a header, run, report PASS/FAIL by exit code
    local title="$1"; shift
    echo; echo "=================================================================="
    echo "== $title"; echo "=================================================================="
    if "$@"; then echo "-- [$title] finished OK"; else echo "-- [$title] FAILED (exit $?)"; FAILED="$FAILED | $title"; fi
}

FAILED=""
{
    echo "post-training checks: $CKPT  (version=$VERSION, device=$DEV, $(date))"
    run "1 code smoke test" bash -c "python3 staged_smoke_test.py 2>&1 | grep -E 'STAGE|PASS|FAIL|all-closed share|RESULT|Error' ; exit \${PIPESTATUS[0]}"
    run "2 closed-room validation (FD reference)" bash -c "python3 validate_closed_room.py '$CKPT' 2>&1 | grep -A8 SUMMARY; exit \${PIPESTATUS[0]}"
    run "3 residual diagnosis D1 (closed room)" bash -c "python3 diagnose_residual_map.py '$CKPT' 2>&1 | tail -22; exit \${PIPESTATUS[0]}"
    run "4 window-BC cross-check" python3 crosscheck_windows.py "$CKPT"
    run "5 level-3 physics consistency (open + closed)" bash -c "python3 check_physics_consistency.py '$CKPT' --device $DEV 2>&1 | tail -52; exit \${PIPESTATUS[0]}"
    run "6 level-2 CO2 vs FV solve with the model's own flow" bash -c "python3 check_co2_with_model_flow.py '$CKPT' --device $DEV 2>&1 | tail -26; exit \${PIPESTATUS[0]}"
    echo; echo "=================================================================="
    if [ -z "$FAILED" ]; then echo "ALL CHECK SCRIPTS RAN ($(( ($(date +%s) - START) / 60 )) min). Read the numbers above."
    else echo "SOME CHECK SCRIPTS FAILED: $FAILED"; fi
} 2>&1 | tee "$LOG"

case "$NAME" in
    *iter*)
        TAG="${VERSION}_${NAME##*_}"
        if [ -d "figures/$VERSION" ]; then rm -rf "figures/$TAG"; mv "figures/$VERSION" "figures/$TAG"; fi
        echo "figures moved to figures/$TAG" | tee -a "$LOG" ;;
esac
echo "log written to $LOG"
grep -q "SOME CHECK SCRIPTS FAILED" "$LOG" && exit 1
exit 0
