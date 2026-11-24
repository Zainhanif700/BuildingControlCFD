#!/usr/bin/env bash
# Updated 2026-11-24: moved to runs/; the script changes into the folder above, so it works from anywhere.
# Two tests of how the sensor model would behave with real sensors (quick settings, 5000 steps, 3 members, 8 real stations).
# 1. Direction: a speed sensor gives only the size of the speed, so the model does not get "in" or "out" for a window.
#    st_dir_f (fixed split), st_dir_1..3 (the three random splits), default settings.
#    Reference without this limit: fixed split 44.12 %, random splits 32.28 / 30.38 / 32.32 %.
# 2. Wrong window speed: the models of run_checks.sh (default, st_chk_d1..3, and st_lc52 on the fixed split) get window
#    speeds that are 0.5, 0.75, 1.5 and 2 times too large or too small (no retraining).
# Usage: bash runs/run_realism.sh <data dir> <scenario file>        Result: ~/realism_summary.txt
set -u
DATA="${1:?usage: bash run_realism.sh <data dir> <scenario file>}"
SCEN="${2:?usage: bash run_realism.sh <data dir> <scenario file>}"
cd "$(dirname "$0")/.." || exit 1
rm -f ~/realism_summary.txt
run() {
    tag=$1; shift
    python train_sensor.py --data-dir "$DATA" --scenarios "$SCEN" --transitions --iters 5000 --members 3 \
        --log-every 5000 --noise 10 --direction-blind --tag "$tag" "$@" > ~/"$tag".log 2>&1
    echo "direction-blind $tag: $(grep '^test:' ~/$tag.log)" >> ~/realism_summary.txt
}
run st_dir_f
for s in 1 2 3; do run st_dir_$s --split-seed $s; done
sens() {   # tag, split seed (0 = fixed), scale
    seed=$3; extra=""; [ "$seed" != 0 ] && extra="--split-seed $seed"
    python diagnose_sensor.py --data-dir "$DATA" --transitions --scenarios "$SCEN" --tag $1 $extra --v-scale $2 --quick \
        > ~/sens_$1_$2.txt 2>&1
    echo "speeds x$2, model $1: $(grep -A 20 '^TEST scenarios  (relative' ~/sens_$1_$2.txt | grep -E '^  mean' | awk '{print "strict " $2 " %, paper-style " $8 " %"}')" >> ~/realism_summary.txt
}
for sc in 1.0 0.5 0.75 1.5 2.0; do
    sens st_lc52 $sc 0
    for s in 1 2 3; do sens st_chk_d$s $sc $s; done
done
echo ALL_DONE >> ~/realism_summary.txt
