#!/usr/bin/env bash
# Updated 2026-11-24: moved to runs/; the script changes into the folder above, so it works from anywhere.
# The synthetic lecture (see lecture_demo.py): CO2 starts at about 420 ppm, students come, some leave.
# 1. Trains a sensor model that fits the real sensors: direction-blind (a speed sensor gives only the size of the
#    speed) and people-blind (the number of people is not known), on the DIRECT data, 8 real stations,
#    5000 steps, 3 members.
# 2. Makes the lecture graph for two scenarios: the first test scenario that fits the plan and reaches 700 to 900 ppm,
#    and N24 (one window open, slow air exchange).
# Usage: bash runs/run_lecture_demo.sh <direct data dir, e.g. data_v4> <scenario file>
# Result: ~/lecture_auto.png, ~/lecture_N24.png and ~/lecture_demo.txt (lead times of the warning)
set -u
DATA="${1:?usage: bash run_lecture_demo.sh <direct data dir> <scenario file>}"
SCEN="${2:?usage: bash run_lecture_demo.sh <direct data dir> <scenario file>}"
cd "$(dirname "$0")/.." || exit 1
python train_sensor.py --data-dir "$DATA" --scenarios "$SCEN" --iters 5000 --members 3 --log-every 5000 \
    --noise 10 --direction-blind --people-blind --tag st_lecture > ~/st_lecture.log 2>&1
{
python lecture_demo.py --data-dir "$DATA" --scenarios "$SCEN" --tag st_lecture --out ~/lecture_auto.png --device cuda
echo
python lecture_demo.py --data-dir "$DATA" --scenarios "$SCEN" --tag st_lecture --scenario N24 --out ~/lecture_N24.png --device cuda
} > ~/lecture_demo.txt 2>&1
echo ALL_DONE >> ~/lecture_demo.txt
