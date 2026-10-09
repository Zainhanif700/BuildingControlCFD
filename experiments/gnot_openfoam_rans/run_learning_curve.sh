#!/bin/bash
# Learning curve: train the forecast ensemble with 5, 10 and 20 training cases (same 8 test cases, same
# settings as the full run rans_tr with 31 cases). Finished sizes are skipped. ~1.5 h per size on the GPU.
# Usage (from experiments/gnot_openfoam_rans, in tmux):  bash run_learning_curve.sh
set -e
cd "$(dirname "$0")"
PY="$HOME/anaconda3/envs/cfd/bin/python"
SIZES="5 10 20"
"$PY" make_lc_scenarios.py $SIZES
for n in $SIZES; do
    if [ -f "checkpoints/rans_tr_n$n/member4.pth" ]; then echo "n = $n: done -- skipped"; continue; fi
    rm -rf "checkpoints/rans_tr_n$n"
    "$PY" train_forecast.py --data-dir data_transitions --scenarios "scenarios_lc/scen_n$n.txt" --transitions \
        --tag "rans_tr_n$n" --members 5 --iters 20000 --z 1.6 --c-offset 400 2>&1 | tee "logs/rans_tr_n$n.log"
done
echo
echo "LEARNING CURVE (test, ensemble vs persistence)"
for f in logs/rans_tr_n5.log logs/rans_tr_n10.log logs/rans_tr_n20.log logs/rans_tr.log; do
    echo "== $f"; grep -E "^  test " "$f" || true
done
