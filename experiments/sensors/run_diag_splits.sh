#!/usr/bin/env bash
# Diagnosis (strict, paper-style, ppm, warning test) of the models from run_checks.sh on their own random splits:
#   st_chk_s1..s3 tuned, st_chk_d1..d3 default, st_chk_ps1..ps3 tuned + physics token
# Usage: bash run_diag_splits.sh <data dir> <scenario file>        Result: ~/diag_<tag>.txt and ~/diag_splits_summary.txt
set -u
DATA="${1:?usage: bash run_diag_splits.sh <data dir> <scenario file>}"
SCEN="${2:?usage: bash run_diag_splits.sh <data dir> <scenario file>}"
cd "$(dirname "$0")" || exit 1
for s in 1 2 3; do
    for t in st_chk_d$s st_chk_s$s st_chk_ps$s; do
        python diagnose_sensor.py --data-dir "$DATA" --transitions --scenarios "$SCEN" --tag $t --split-seed $s \
            > ~/diag_$t.txt 2>&1
    done
done
{
for kind in d s ps; do
    echo "== $kind: events found / missed / false alarms, summed over the three splits (700 ppm, 800 ppm)"
    for lim in 700 800; do
        for s in 1 2 3; do grep -A 6 "limit   real" ~/diag_st_chk_$kind$s.txt | grep -E "^ +$lim "; done \
            | awk -v l=$lim '{r+=$2; f+=$3; m+=$4; a+=$5} END {printf "  %s ppm: real %d, found %d, missed %d, false alarms %d\n", l, r, f, m, a}'
    done
done
} > ~/diag_splits_summary.txt
echo ALL_DONE >> ~/diag_splits_summary.txt
