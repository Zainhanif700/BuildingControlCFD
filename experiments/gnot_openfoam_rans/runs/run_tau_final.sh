#!/bin/bash
# Updated 2026-11-24: moved to runs/; the script changes into the folder above, so it works from anywhere.
# Overnight run (branch tau-input), round 3: the combination of the two best settings so far.
#  1) v4_tr_tau09_long: transition model with tau input, 90 % of the samples with the window change inside the history,
#     40,000 steps (about 4 h 40 min, estimate from the earlier runs). Then the history-change test.
#  2) v4_dir_long: direct-run model (fresh-air start, no window change) with 40,000 steps, so that the transition
#     and the direct-run model are trained in the same way if the final model is replaced (about 3-4 h, estimate).
# Usage (in tmux):  bash runs/run_tau_final.sh   -- part 1 is the important one; part 2 can be stopped after it.
cd "$(dirname "$0")/.." || exit 1

echo "=== $(date) start v4_tr_tau09_long"
python3 train_forecast.py --data-dir data_transitions_v4 --transitions --scenarios scenarios_v4.txt \
    --tag v4_tr_tau09_long --z 1.6 --c-offset 400 --open-flags --pad-prob 0.9 --tau --iters 40000 > v4_tr_tau09_long.log 2>&1
echo "=== $(date) training v4_tr_tau09_long finished (exit $?)"
python3 eval_history_change.py --ckpt checkpoints/v4_tr_tau09_long --data-dir data_transitions_v4 \
    --scenarios scenarios_v4.txt > v4_tr_tau09_long_history_change.log 2>&1
echo "=== $(date) history-change test v4_tr_tau09_long finished (exit $?)"

echo "=== $(date) start v4_dir_long"
python3 train_forecast.py --data-dir data_v4 --scenarios scenarios_v4.txt \
    --tag v4_dir_long --z 1.6 --c-offset 400 --open-flags --iters 40000 > v4_dir_long.log 2>&1
echo "=== $(date) training v4_dir_long finished (exit $?)"
echo "=== $(date) ALL DONE"
