#!/usr/bin/env bash
# Final proof runs with the full settings (5 members, 20000 steps) and the best setup found in the quick tests
# (relative loss, network width 64, 2 layers). Three cases, then the diagnosis of each (errors, ppm, warning test):
#   fin_8real   the 8 real stations
#   fin_v1      the 8 real stations + 1 virtual sensor in the middle of the seating area (simulation only)
#   fin_v16     the 8 real stations + a grid of 16 virtual sensors in the seating area (simulation only)
# Reference without tuning: sensors_v4_d2 (8 real stations, default settings, full settings) = 44.49 %.
# Usage: bash run_final.sh <data dir> <scenario file>        Result: ~/final_summary.txt, ~/diag_fin_*.txt
set -u
DATA="${1:?usage: bash run_final.sh <data dir> <scenario file>}"
SCEN="${2:?usage: bash run_final.sh <data dir> <scenario file>}"
cd "$(dirname "$0")" || exit 1
run() {
    tag=$1; shift
    python train_sensor.py --data-dir "$DATA" --scenarios "$SCEN" --transitions --members 5 --iters 20000 \
        --log-every 5000 --noise 10 --relative-loss --d 64 --layers 2 --tag "$tag" "$@" > ~/"$tag".log 2>&1
    python diagnose_sensor.py --data-dir "$DATA" --transitions --scenarios "$SCEN" --tag "$tag" > ~/diag_"$tag".txt 2>&1
    echo "$tag: $(grep '^test:' ~/$tag.log)" >> ~/final_summary.txt
}
rm -f ~/final_summary.txt
run fin_8real
run fin_v1 --virtual 1
run fin_v16 --virtual 16
echo ALL_DONE >> ~/final_summary.txt
