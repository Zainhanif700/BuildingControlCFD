#!/usr/bin/env bash
# Third round (5000 steps, 3 members, about 4-5 min per test). Base: the relative loss, which was the best single change
# in round 2 (st_rel = 38.77 %, reference without it st_lc52 = 44.12 %). One extra change at a time:
#   st_rel_small    smaller network      st_rel_noise30  30 ppm noise       st_rel_sdrop  station dropout 0.5
#   st_rel_f2       relative-loss floor 2 ppm (default 5)                 st_rel_f15    floor 15 ppm
# and with virtual sensors in the seating area (simulation only), to compare with 28.81 % (1) and 21.29 % (16) without it:
#   st_rel_v1       st_rel_v16
# Usage: bash run_improve3.sh <data dir> <scenario file>        Result: ~/improve3_summary.txt
set -u
DATA="${1:?usage: bash run_improve3.sh <data dir> <scenario file>}"
SCEN="${2:?usage: bash run_improve3.sh <data dir> <scenario file>}"
cd "$(dirname "$0")" || exit 1
run() {
    tag=$1; shift
    python train_sensor.py --data-dir "$DATA" --scenarios "$SCEN" --transitions --iters 5000 --members 3 \
        --log-every 5000 --noise 10 --relative-loss --tag "$tag" "$@" > ~/"$tag".log 2>&1
}
run st_rel_small --d 64 --layers 2
run st_rel_noise30 --noise 30
run st_rel_sdrop --station-dropout 0.5
run st_rel_f2 --rel-floor 2
run st_rel_f15 --rel-floor 15
run st_rel_v1 --virtual 1
run st_rel_v16 --virtual 16
for t in st_rel_small st_rel_noise30 st_rel_sdrop st_rel_f2 st_rel_f15 st_rel_v1 st_rel_v16; do
    echo "$t: $(grep '^test:' ~/$t.log)"
done > ~/improve3_summary.txt
echo ALL_DONE >> ~/improve3_summary.txt
