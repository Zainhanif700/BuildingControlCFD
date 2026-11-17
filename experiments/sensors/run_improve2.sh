#!/usr/bin/env bash
# Second round to bring the error of the sensor model down (5000 steps, 3 members, about 4-5 min per test).
# Reference: 8 real stations, default settings, st_lc52 = 44.12 %; one middle virtual sensor st_v1 = 28.81 %.
#   st_rel          relative loss: every sample is weighed by the size of its own signal (matches the relative error)
#   st_combo        all small tricks together: d 64, 2 layers, weight decay 1e-2, noise 30 ppm, station dropout 0.5
#   st_combo_rel    the combination plus the relative loss
#   st_combo_rel_v1 the same with one virtual sensor in the middle of the seating area (simulation only)
# Then the diagnosis (ppm, paper metric, warning test) of st_rel and st_combo_rel.
# Usage: bash run_improve2.sh <data dir> <scenario file>        Result: ~/improve2_summary.txt, ~/diag_<tag>.txt
set -u
DATA="${1:?usage: bash run_improve2.sh <data dir> <scenario file>}"
SCEN="${2:?usage: bash run_improve2.sh <data dir> <scenario file>}"
cd "$(dirname "$0")" || exit 1
COMBO="--d 64 --layers 2 --wd 1e-2 --noise 30 --station-dropout 0.5"
run() {
    tag=$1; shift
    python train_sensor.py --data-dir "$DATA" --scenarios "$SCEN" --transitions --iters 5000 --members 3 \
        --log-every 5000 --noise 10 --tag "$tag" "$@" > ~/"$tag".log 2>&1
}
run st_rel --relative-loss
run st_combo $COMBO
run st_combo_rel $COMBO --relative-loss
run st_combo_rel_v1 $COMBO --relative-loss --virtual 1
for t in st_rel st_combo st_combo_rel st_combo_rel_v1; do
    echo "$t: $(grep '^test:' ~/$t.log)"
done > ~/improve2_summary.txt
for t in st_rel st_combo_rel; do
    python diagnose_sensor.py --data-dir "$DATA" --transitions --scenarios "$SCEN" --tag $t > ~/diag_$t.txt 2>&1
done
echo ALL_DONE >> ~/improve2_summary.txt
