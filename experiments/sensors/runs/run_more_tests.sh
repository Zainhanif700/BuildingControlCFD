#!/usr/bin/env bash
# Updated 2026-11-24: moved to runs/; the script changes into the folder above, so it works from anywhere.
# More cheap tests of the sensor model (5000 steps, 3 members each, about 4-5 min per test, 8 tests):
#   where should ONE extra sensor stand?   st_vfront, st_vback, st_vwindowside, st_vdoorside  (st_v1 = middle, done before)
#   what do many sensors give?             st_v16 (a grid of 16 virtual sensors in the seating area)
#   does a longer station history help?    st_h12, st_h24, st_h40 (6, 12 and 20 min), all on the same samples (t0 >= 39)
# The virtual sensors exist only in the simulation. Usage: bash runs/run_more_tests.sh <data dir> <scenario file>
# Result: ~/more_summary.txt
set -u
DATA="${1:?usage: bash run_more_tests.sh <data dir> <scenario file>}"
SCEN="${2:?usage: bash run_more_tests.sh <data dir> <scenario file>}"
cd "$(dirname "$0")/.." || exit 1
run() {
    tag=$1; shift
    python train_sensor.py --data-dir "$DATA" --scenarios "$SCEN" --transitions --iters 5000 --members 3 \
        --log-every 5000 --noise 10 --tag "$tag" "$@" > ~/"$tag".log 2>&1
}
for v in front back windowside doorside 16; do run st_v$v --virtual $v; done
for h in 12 24 40; do run st_h$h --hist $h --t0-min 39; done
for t in st_vfront st_vback st_vwindowside st_vdoorside st_v16 st_h12 st_h24 st_h40; do
    echo "$t: $(grep '^test:' ~/$t.log)"
done > ~/more_summary.txt
echo ALL_DONE >> ~/more_summary.txt
