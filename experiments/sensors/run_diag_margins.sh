#!/usr/bin/env bash
# Warning with a safety margin for the models of run_checks.sh on their random splits (default = st_chk_d1..d3,
# tuned + physics = st_chk_ps1..ps3), summed over the three splits.
# Usage: bash run_diag_margins.sh <data dir> <scenario file>        Result: ~/diag_margins_summary.txt
set -u
DATA="${1:?usage: bash run_diag_margins.sh <data dir> <scenario file>}"
SCEN="${2:?usage: bash run_diag_margins.sh <data dir> <scenario file>}"
cd "$(dirname "$0")" || exit 1
for s in 1 2 3; do
    for t in st_chk_d$s st_chk_ps$s; do
        python diagnose_sensor.py --data-dir "$DATA" --transitions --scenarios "$SCEN" --tag $t --split-seed $s \
            > ~/diagm_$t.txt 2>&1
    done
done
{
for kind in d ps; do
    echo "== $kind: summed over the three splits (real events, found, missed, false alarms)"
    for lim in 700 800; do
        for m in 0 25 50 75; do
            for s in 1 2 3; do grep -A 20 "^warning with a safety margin" ~/diagm_st_chk_$kind$s.txt | grep -E "^ +$lim\.? +$m +"; done \
                | awk -v l=$lim -v m=$m '{r+=$3; f+=$4; x+=$5; a+=$6} END {printf "  limit %s ppm, margin %2d ppm: real %d, found %d, missed %d, false alarms %d\n", l, m, r, f, x, a}'
        done
    done
done
} > ~/diag_margins_summary.txt
echo ALL_DONE >> ~/diag_margins_summary.txt
