#!/bin/bash
# Second overnight experiment (branch tau-input). The first tau runs (30 % and 60 % change samples) improved the case
# "window change inside the history" (41.4 % -> 29.0 % strict), but the model is not even good on the TRAINING scenarios
# for the first minutes (43.6 % at t0 = 0), so it underfits these samples. Two variants:
#   v4_tr_tau09:      90 % of the samples have the change inside the history, 20,000 steps  (about 2 h 20 min)
#   v4_tr_tau06_long: 60 % change samples, 40,000 steps                                      (about 4 h 40 min)
# The history-change test runs after each training. Usage (in tmux):  bash run_tau_more.sh
cd "$(dirname "$0")" || exit 1
run () {  # tag, pad-prob, iters
  echo "=== $(date) start $1 (pad-prob $2, iters $3, with tau)"
  python3 train_forecast.py --data-dir data_transitions_v4 --transitions --scenarios scenarios_v4.txt \
      --tag "$1" --z 1.6 --c-offset 400 --open-flags --pad-prob "$2" --tau --iters "$3" > "$1.log" 2>&1
  echo "=== $(date) training $1 finished (exit $?)"
  python3 eval_history_change.py --ckpt "checkpoints/$1" --data-dir data_transitions_v4 \
      --scenarios scenarios_v4.txt > "$1_history_change.log" 2>&1
  echo "=== $(date) history-change test $1 finished (exit $?)"
}
run v4_tr_tau09 0.9 20000
run v4_tr_tau06_long 0.6 40000
echo "=== $(date) ALL DONE"
