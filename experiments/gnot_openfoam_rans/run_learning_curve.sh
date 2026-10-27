#!/bin/bash
# Learning curve: the same training with fewer training cases (same test cases); finished sizes are skipped.
# Usage: bash run_learning_curve.sh                                   (first setup: 5, 10, 20 cases)
#        bash run_learning_curve.sh scenarios_v2.txt data_transitions_v2 v2_tr 5 10 15
set -e
cd "$(dirname "$0")"
PY="$HOME/anaconda3/envs/cfd/bin/python"
SCEN="${1:-scenarios.txt}"; DATA="${2:-data_transitions}"; TAG="${3:-rans_tr}"
shift $(( $# < 3 ? $# : 3 ))
SIZES="${*:-5 10 20}"
mkdir -p logs
"$PY" make_lc_scenarios.py $SIZES --scenarios "$SCEN" --data-dir "$DATA" --name "$TAG"
for n in $SIZES; do
    if [ -f "checkpoints/${TAG}_n$n/member4.pth" ]; then echo "n = $n: done -- skipped"; continue; fi
    rm -rf "checkpoints/${TAG}_n$n"
    "$PY" train_forecast.py --data-dir "$DATA" --scenarios "scenarios_lc/${TAG}_n$n.txt" --transitions \
        --tag "${TAG}_n$n" --members 5 --iters 20000 --z 1.6 --c-offset 400 > "logs/${TAG}_n$n.log" 2>&1
done
echo
echo "LEARNING CURVE (test, ensemble vs persistence)"
for n in $SIZES; do
    echo "== n = $n"; grep -E "^  test " "logs/${TAG}_n$n.log" || true
done
