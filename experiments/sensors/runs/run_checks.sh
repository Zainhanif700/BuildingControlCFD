#!/usr/bin/env bash
# Updated 2026-11-24: moved to runs/; the script changes into the folder above, so it works from anywhere.
# Checks of how honest and how stable the sensor results are (5000 steps, 3 members each, about 4 min per run).
# Setup of all runs: the 8 real stations. "tuned" = relative loss, width 64, 2 layers (35.09 % on the fixed split);
# "default" = the first settings (44.12 % on the fixed split).
#   st_chk_people      tuned, the model is NOT told the number of people (the real system does not know it)
#   st_chk_seen        tuned, test = the training scenarios at later times (window setting seen in training)
#   st_chk_s1..s3      tuned, three other random splits of the 68 scenarios (8 N + 8 A for the test)
#   st_chk_d1..d3      default settings on the same three random splits
# Waits first for a tmux session called "fin" (the final proof runs), so that the GPU is not shared.
# Usage: bash runs/run_checks.sh <data dir> <scenario file>        Result: ~/checks_summary.txt
set -u
DATA="${1:?usage: bash run_checks.sh <data dir> <scenario file>}"
SCEN="${2:?usage: bash run_checks.sh <data dir> <scenario file>}"
cd "$(dirname "$0")/.." || exit 1
while tmux has-session -t fin 2>/dev/null; do sleep 30; done
TUNED="--relative-loss --d 64 --layers 2"
run() {
    tag=$1; shift
    python train_sensor.py --data-dir "$DATA" --scenarios "$SCEN" --transitions --iters 5000 --members 3 \
        --log-every 5000 --noise 10 --tag "$tag" "$@" > ~/"$tag".log 2>&1
}
run st_chk_people $TUNED --people-blind
run st_chk_seen $TUNED --seen-config
for s in 1 2 3; do run st_chk_s$s $TUNED --split-seed $s; done
for s in 1 2 3; do run st_chk_d$s --split-seed $s; done
# the physics token (fresh-air flow, final CO2 level, time constant of the room) with the tuned settings, on the fixed
# split and on the three random splits (compare with st_chk_s1..s3 and with 35.09 %)
run st_chk_phys $TUNED --physics
for s in 1 2 3; do run st_chk_ps$s $TUNED --physics --split-seed $s; done
for t in st_chk_people st_chk_seen st_chk_s1 st_chk_s2 st_chk_s3 st_chk_d1 st_chk_d2 st_chk_d3 st_chk_phys st_chk_ps1 st_chk_ps2 st_chk_ps3; do
    echo "$t: $(grep '^test:' ~/$t.log)"
done > ~/checks_summary.txt
echo ALL_DONE >> ~/checks_summary.txt
