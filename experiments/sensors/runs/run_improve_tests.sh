#!/usr/bin/env bash
# Updated 2026-11-24: moved to runs/; the script changes into the folder above, so it works from anywhere.
# First round to bring the error of the sensor model down (5000 steps, 3 members each, about 4-5 min per test).
# Reference: 8 real stations with the default settings, st_lc52 = 44.12 %.
#   st_a_v16, st_a_v1   the anchored model with a grid of 16 / one virtual sensor in the seating area (simulation only)
#   st_small            a smaller network (d 64, 2 layers) against overfitting
#   st_wd               stronger weight decay (1e-2)
#   st_noise30          more noise on the station CO2 in training (30 ppm instead of 10)
#   st_sdrop            station dropout: some stations are hidden in half of the training steps
# Usage: bash runs/run_improve_tests.sh <data dir> <scenario file>        Result: ~/improve_summary.txt
set -u
DATA="${1:?usage: bash run_improve_tests.sh <data dir> <scenario file>}"
SCEN="${2:?usage: bash run_improve_tests.sh <data dir> <scenario file>}"
cd "$(dirname "$0")/.." || exit 1
run() {
    tag=$1; shift
    python train_sensor.py --data-dir "$DATA" --scenarios "$SCEN" --transitions --iters 5000 --members 3 \
        --log-every 5000 --noise 10 --tag "$tag" "$@" > ~/"$tag".log 2>&1
}
run st_a_v16 --anchor --virtual 16
run st_a_v1 --anchor --virtual 1
run st_small --d 64 --layers 2
run st_wd --wd 1e-2
run st_noise30 --noise 30
run st_sdrop --station-dropout 0.5
for t in st_a_v16 st_a_v1 st_small st_wd st_noise30 st_sdrop; do
    echo "$t: $(grep '^test:' ~/$t.log)"
done > ~/improve_summary.txt
echo ALL_DONE >> ~/improve_summary.txt
