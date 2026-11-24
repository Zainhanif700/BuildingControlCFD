#!/usr/bin/env bash
# Updated 2026-11-24: moved to runs/; the script changes into the folder above, so it works from anywhere.
# The whole chain with a trained model (simulation file -> sensor file in the server format -> reader -> mapper -> model).
# 1. Trains a sensor model that fits the real sensors: direction-blind (a speed sensor gives only the size of the speed),
#    on the DIRECT data (the room starts with clean air, like the synthetic sensor file), 8 real stations,
#    5000 steps, 3 members.
# 2. Runs demo_chain.py on the test scenarios (only those that fit the plan: W5 and W8 closed), 45 people.
# Usage: bash runs/run_chain_demo.sh <direct data dir, e.g. data_v4> <scenario file>        Result: ~/chain_demo.txt
set -u
DATA="${1:?usage: bash run_chain_demo.sh <direct data dir> <scenario file>}"
SCEN="${2:?usage: bash run_chain_demo.sh <direct data dir> <scenario file>}"
cd "$(dirname "$0")/.." || exit 1
python train_sensor.py --data-dir "$DATA" --scenarios "$SCEN" --iters 5000 --members 3 --log-every 5000 \
    --noise 10 --direction-blind --tag st_chain > ~/st_chain.log 2>&1
python demo_chain.py --data-dir "$DATA" --scenarios "$SCEN" --tag st_chain --people 45 --device cuda > ~/chain_demo.txt 2>&1
echo ALL_DONE >> ~/chain_demo.txt
