#!/usr/bin/env bash
# Fourth round (5000 steps, 3 members): smaller networks, all with the relative loss.
# Reference: relative loss + d 64 / 2 layers (st_rel_small) = 35.09 %; relative loss with the default network = 38.77 %;
# default settings = 44.12 %. With virtual sensors (simulation only): st_rel_v1 = 31.63 %, st_rel_v16 = 21.92 %.
#   st_r_d32l2  st_r_d64l1  st_r_d32l1  st_r_d16l1     smaller networks (d = width, l = layers)
#   st_rs_v1    st_rs_v16                               the d 64 / 2 layer network with 1 / 16 virtual sensors
# Usage: bash run_improve4.sh <data dir> <scenario file>        Result: ~/improve4_summary.txt
set -u
DATA="${1:?usage: bash run_improve4.sh <data dir> <scenario file>}"
SCEN="${2:?usage: bash run_improve4.sh <data dir> <scenario file>}"
cd "$(dirname "$0")" || exit 1
run() {
    tag=$1; shift
    python train_sensor.py --data-dir "$DATA" --scenarios "$SCEN" --transitions --iters 5000 --members 3 \
        --log-every 5000 --noise 10 --relative-loss --tag "$tag" "$@" > ~/"$tag".log 2>&1
}
run st_r_d32l2 --d 32 --layers 2
run st_r_d64l1 --d 64 --layers 1
run st_r_d32l1 --d 32 --layers 1
run st_r_d16l1 --d 16 --layers 1
run st_rs_v1 --d 64 --layers 2 --virtual 1
run st_rs_v16 --d 64 --layers 2 --virtual 16
for t in st_r_d32l2 st_r_d64l1 st_r_d32l1 st_r_d16l1 st_rs_v1 st_rs_v16; do
    echo "$t: $(grep '^test:' ~/$t.log)"
done > ~/improve4_summary.txt
echo ALL_DONE >> ~/improve4_summary.txt
