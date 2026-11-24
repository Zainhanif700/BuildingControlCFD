#!/bin/bash
# Updated 2026-11-24: moved to runs/; the script changes into the folder above, so it works from anywhere.
# Overnight experiment: train the final transition model with samples that have the window change INSIDE the
# 6-minute history (30 % and 60 % of the samples), then run the history-change test on each new model.
# Same settings as the final model (v4_tr): 5 networks, 20,000 steps, --open-flags, scenarios_v4.txt.
# Needs about 1.5 h of GPU time per variant (the final model took about 19 min per network).
# Usage (in tmux, so that an SSH disconnect does not kill it):  bash runs/run_pad_experiment.sh
cd "$(dirname "$0")/.." || exit 1
for P in 0.3 0.6; do
  TAG=v4_tr_pad${P/./}
  echo "=== $(date) start $TAG (pad-prob $P)"
  python3 train_forecast.py --data-dir data_transitions_v4 --transitions --scenarios scenarios_v4.txt \
      --tag "$TAG" --z 1.6 --c-offset 400 --open-flags --pad-prob "$P" > "${TAG}.log" 2>&1
  echo "=== $(date) training $TAG finished (exit $?)"
  python3 eval_history_change.py --ckpt "checkpoints/$TAG" --data-dir data_transitions_v4 \
      --scenarios scenarios_v4.txt > "${TAG}_history_change.log" 2>&1
  echo "=== $(date) history-change test $TAG finished (exit $?)"
done
echo "=== $(date) ALL DONE"
