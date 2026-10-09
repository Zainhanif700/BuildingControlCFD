#!/usr/bin/env bash
# Runs all checks on one physics-only checkpoint (smoke test, closed room, windows, physics, CO2) into one log.
# Usage: bash post_training_checks.sh <checkpoint> [--gpu] [--skip-smoke] [--quick]
set -u
CKPT="${1:?usage: bash post_training_checks.sh <checkpoint> [--gpu] [--skip-smoke]}"
[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT"; exit 1; }
DEV=cpu; SKIP_SMOKE=0; QUICK=0
for opt in "${@:2}"; do
    case "$opt" in
        --gpu) DEV=cuda ;;
        --skip-smoke) SKIP_SMOKE=1 ;;
        --quick) QUICK=1 ;;      # skip level 2 (the slowest step, ~25-35 min on CPU); mid-run only
        *) echo "unknown option: $opt (use --gpu, --skip-smoke, --quick)"; exit 1 ;;
    esac
done
[ "$DEV" = cpu ] && export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
NAME=$(basename "$CKPT" .pth)
[ -f staged_smoke_test.py ] || { echo "run this from experiments/pinn"; exit 1; }
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
    if [ "$SKIP_SMOKE" = 1 ]; then
        echo; echo "== 1 code smoke test: SKIPPED (--skip-smoke; run it once per code version)"
    else
        run "1 code smoke test" bash -c "python3 staged_smoke_test.py 2>&1 | grep --line-buffered -E 'STAGE|PASS|FAIL|all-closed share|RESULT|Error' ; exit \${PIPESTATUS[0]}"
    fi
    run "2 closed-room validation (FD reference)" bash -c "python3 validate_closed_room.py '$CKPT' 2>&1 | grep -A8 SUMMARY; exit \${PIPESTATUS[0]}"
    run "3 residual diagnosis D1 (closed room)" bash -c "python3 diagnose_residual_map.py '$CKPT' 2>&1 | tail -22; exit \${PIPESTATUS[0]}"
    run "4 window-BC cross-check" python3 crosscheck_windows.py "$CKPT"
    run "5 level-3 physics consistency (open + closed)" bash -c "python3 check_physics_consistency.py '$CKPT' --device $DEV 2>&1 | tail -52; exit \${PIPESTATUS[0]}"
    if [ "$QUICK" = 1 ]; then
        echo; echo "== 6 level-2 CO2 check: SKIPPED (--quick; run it on the final model)"
    else
        run "6 level-2 CO2 vs FV solve with the model's own flow" bash -c "python3 check_co2_with_model_flow.py '$CKPT' --device $DEV 2>&1 | tail -26; exit \${PIPESTATUS[0]}"
    fi
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
