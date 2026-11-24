#!/bin/bash
# Learning curve of the FINAL setup (transition model with tau input, 90 % of the samples with the window change inside
# the history, 40,000 steps, 5 networks): the same training with fewer training settings, the same 16 test settings.
# The subsets of the 52 training settings are random and nested (13 inside 26 inside 39). Finished sizes are skipped.
# Updated 2026-11-24. One size takes about 3 h 10 min on an RTX A2000 (5 networks, about 38 min each; estimated).
# Usage (in tmux, env cfd, in this folder with data_transitions_v4):  bash run_learning_curve_v4.sh [sizes, default 13 26 39]
# Result: logs/v4_tr_tau09_lc_n<size>.log and a table at the end. The size 52 is the final model (v4_tr_tau09_long).
set -e
cd "$(dirname "$0")"
PY="${CFD_PY:-python3}"
SCEN=scenarios_v4.txt; DATA=data_transitions_v4; TAG=v4_tr_tau09_lc
SIZES="${*:-13 26 39}"
mkdir -p logs
"$PY" make_lc_scenarios.py $SIZES --scenarios "$SCEN" --data-dir "$DATA" --name "$TAG"
for n in $SIZES; do
    if [ -f "checkpoints/${TAG}_n$n/member4.pth" ]; then echo "n = $n: done -- skipped"; continue; fi
    rm -rf "checkpoints/${TAG}_n$n"
    echo "=== $(date) start n = $n"
    "$PY" train_forecast.py --data-dir "$DATA" --transitions --scenarios "scenarios_lc/${TAG}_n$n.txt" \
        --tag "${TAG}_n$n" --z 1.6 --c-offset 400 --open-flags --pad-prob 0.9 --tau --iters 40000 \
        > "logs/${TAG}_n$n.log" 2>&1
    echo "=== $(date) finished n = $n"
done
echo
echo "LEARNING CURVE (test, ensemble vs persistence; n = 52 is the final model)"
for n in $SIZES; do
    echo "== n = $n"; grep -E "^  test " "logs/${TAG}_n$n.log" || true
done
echo "=== $(date) ALL DONE"
