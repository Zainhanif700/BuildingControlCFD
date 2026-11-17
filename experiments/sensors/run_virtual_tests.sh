#!/usr/bin/env bash
# What-if test: how much would sensors in the middle of the room help? The 8 real stations plus 1, 2 or 4 virtual
# stations in the seating area (simulation only). 5000 steps, 3 members each, about 4-5 min per test.
# Then the diagnosis (errors, room average, warning test) for the 8-station reference and the three variants.
# Usage: bash run_virtual_tests.sh <data dir> <scenario file>
# Results: ~/virtual_summary.txt (the test errors) and ~/diag_<tag>.txt (the diagnosis of each)
set -u
DATA="${1:?usage: bash run_virtual_tests.sh <data dir> <scenario file>}"
SCEN="${2:?usage: bash run_virtual_tests.sh <data dir> <scenario file>}"
cd "$(dirname "$0")" || exit 1
for n in 1 2 4; do
    python train_sensor.py --data-dir "$DATA" --scenarios "$SCEN" --transitions --iters 5000 --members 3 \
        --log-every 5000 --noise 10 --virtual $n --tag st_v$n > ~/st_v$n.log 2>&1
done
{ for t in st_lc52 st_v1 st_v2 st_v4; do echo "$t: $(grep '^test:' ~/$t.log)"; done; } > ~/virtual_summary.txt
for t in sensors_v4_d2 st_v1 st_v2 st_v4; do
    python diagnose_sensor.py --data-dir "$DATA" --transitions --scenarios "$SCEN" --tag $t > ~/diag_$t.txt 2>&1
done
echo ALL_DONE >> ~/virtual_summary.txt
