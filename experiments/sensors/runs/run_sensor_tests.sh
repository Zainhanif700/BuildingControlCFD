#!/usr/bin/env bash
# Updated 2026-11-24: moved to runs/; the script changes into the folder above, so it works from anywhere.
# Two cheap tests of the sensor model (5000 steps and 3 members each, about 5 min per test):
#   st_blind   the model does not get the station CO2, only windows, doors and number of people
#   st_lc<n>   only n of the 52 training scenarios (does more data help?)
# Usage: bash runs/run_sensor_tests.sh <data dir> <scenario file>
# Waits first for a tmux session called "anch" (the anchored run), so that the GPU is not shared.
set -u
DATA="${1:?usage: bash run_sensor_tests.sh <data dir> <scenario file>}"
SCEN="${2:?usage: bash run_sensor_tests.sh <data dir> <scenario file>}"
cd "$(dirname "$0")/.." || exit 1
while tmux has-session -t anch 2>/dev/null; do sleep 20; done
run() {
    tag=$1; shift
    python train_sensor.py --data-dir "$DATA" --scenarios "$SCEN" --transitions --iters 5000 --members 3 \
        --log-every 5000 --noise 10 --tag "$tag" "$@" > ~/"$tag".log 2>&1
}
run st_blind --blind
for n in 13 26 39 52; do run st_lc$n --n-train $n; done
for t in st_blind st_lc13 st_lc26 st_lc39 st_lc52; do echo "$t: $(grep '^test:' ~/$t.log)"; done > ~/sensor_tests_summary.txt
echo ALL_DONE >> ~/sensor_tests_summary.txt
