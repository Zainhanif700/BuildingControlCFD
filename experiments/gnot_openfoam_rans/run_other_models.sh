#!/bin/bash
# Other models on the same data, split and error as the final GNOT model (v4_tr_tau09_long): linear, MLP, FNO.
# First a short check of the grid and a speed test, then the three trainings one after the other.
# Usage (in tmux, env cfd):  bash run_other_models.sh [iters 40000]
# Results: other_<model>.log
cd "$(dirname "$0")" || exit 1
ITERS="${1:-40000}"
COMMON="--data-dir data_transitions_v4 --transitions --scenarios scenarios_v4.txt"
echo "=== $(date) grid check"
python3 other_models.py --info $COMMON
for M in linear mlp fno; do
    echo "=== $(date) start $M"
    python3 other_models.py --model $M $COMMON --tag other_$M --iters "$ITERS" > other_$M.log 2>&1
    echo "=== $(date) $M finished (exit $?)"
done
echo "=== $(date) ALL DONE"
