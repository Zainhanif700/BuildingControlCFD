#!/bin/bash
# Overnight experiment (branch tau-input): can the window change inside the history be improved further?
# Same settings as the adopted model v4_tr_tau09_long (90 % change samples, tau input, 40,000 steps, 5 networks),
# with one more ingredient each:
#   v4_tr_film : FiLM, the window setting, the people and tau modulate every block of the network
#   v4_tr_phys : physics input, fresh-air flow, CO2 level and time constant of a well-mixed room, and at every
#                point the distance of the CO2 to that level
# Success rule (fixed before the run): the error with the window change inside the history must fall below about 18 %
# (v4_tr_tau09_long: 21.3 %) and the normal test must stay at about 0.40 % (strict).
# Waits until the training in tmux session tau3 (v4_dir_long) has finished, so the GPU is free.
# About 3.5 h per variant (estimate). Usage (in tmux):  bash run_film_physics.sh
cd "$(dirname "$0")" || exit 1
while tmux has-session -t tau3 2>/dev/null; do sleep 60; done
run () {  # tag, extra flag
  echo "=== $(date) start $1 ($2)"
  python3 train_forecast.py --data-dir data_transitions_v4 --transitions --scenarios scenarios_v4.txt \
      --tag "$1" --z 1.6 --c-offset 400 --open-flags --pad-prob 0.9 --tau "$2" --iters 40000 > "$1.log" 2>&1
  echo "=== $(date) training $1 finished (exit $?)"
  python3 eval_history_change.py --ckpt "checkpoints/$1" --data-dir data_transitions_v4 \
      --scenarios scenarios_v4.txt > "$1_history_change.log" 2>&1
  echo "=== $(date) history-change test $1 finished (exit $?)"
}
run v4_tr_film --film
run v4_tr_phys --physics
echo "=== $(date) ALL DONE"
