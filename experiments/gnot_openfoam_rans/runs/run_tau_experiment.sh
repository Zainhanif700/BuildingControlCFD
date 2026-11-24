#!/bin/bash
# Updated 2026-11-24: moved to runs/; the script changes into the folder above, so it works from anywhere.
# Overnight experiment (branch tau-input): the model also gets the TIME SINCE THE LAST WINDOW CHANGE as an input
# (the real stations measure open/closed, so this can be known in the room), and 30 % / 60 % of the training samples
# have the window change inside the 6-minute history. Then the history-change test is run on each new model.
# Same settings as the final model (v4_tr): 5 networks, 20,000 steps, --open-flags, scenarios_v4.txt.
# About 1.5 h of GPU time per variant. Usage (in tmux):  bash runs/run_tau_experiment.sh
cd "$(dirname "$0")/.." || exit 1
for P in 0.3 0.6; do
  TAG=v4_tr_tau${P/./}
  echo "=== $(date) start $TAG (pad-prob $P, with tau)"
  python3 train_forecast.py --data-dir data_transitions_v4 --transitions --scenarios scenarios_v4.txt \
      --tag "$TAG" --z 1.6 --c-offset 400 --open-flags --pad-prob "$P" --tau > "${TAG}.log" 2>&1
  echo "=== $(date) training $TAG finished (exit $?)"
  python3 eval_history_change.py --ckpt "checkpoints/$TAG" --data-dir data_transitions_v4 \
      --scenarios scenarios_v4.txt > "${TAG}_history_change.log" 2>&1
  echo "=== $(date) history-change test $TAG finished (exit $?)"
done
echo "=== $(date) ALL DONE"
