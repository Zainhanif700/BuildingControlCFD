#!/bin/bash
# Updated 2026-11-24: moved to runs/; the script changes into the folder above, so it works from anywhere.
# Second random split (scenarios_v4_split2.txt): does the main result of the thesis hold on another test set?
# Same settings as the final model v4_tr_tau09_long (90 % change samples, tau input, 40,000 steps, 5 networks),
# then the history-change test on the new test scenarios. About 3 h 15 (5 x 38 min measured on the first split).
# Usage (in tmux):  bash runs/run_split2.sh
cd "$(dirname "$0")/.." || exit 1
echo "=== $(date) start v4_tr_split2"
python3 train_forecast.py --data-dir data_transitions_v4 --transitions --scenarios scenarios_v4_split2.txt \
    --tag v4_tr_split2 --z 1.6 --c-offset 400 --open-flags --pad-prob 0.9 --tau --iters 40000 > v4_tr_split2.log 2>&1
echo "=== $(date) training finished (exit $?)"
python3 eval_history_change.py --ckpt checkpoints/v4_tr_split2 --data-dir data_transitions_v4 \
    --scenarios scenarios_v4_split2.txt > v4_tr_split2_history_change.log 2>&1
echo "=== $(date) history-change test finished (exit $?)"
echo "=== $(date) ALL DONE"
