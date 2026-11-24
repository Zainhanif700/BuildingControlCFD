#!/usr/bin/env bash
# Updated 2026-11-24: moved to runs/; the script changes into the folder above, so it works from anywhere.
# The results of the default sensor models on the scenarios that fit the real room (W5 and W8 closed), with the models
# from before (nothing is retrained): st_lc52 on the fixed split, st_chk_d1..d3 on the random splits.
# Prints, for every model, the strict and paper-style error, the room-average error and the warning test, and sums the
# warning events over the three random splits.
# Usage: bash runs/run_plan_only.sh <data dir> <scenario file>        Result: ~/plan_only_summary.txt, ~/diagp_<tag>.txt
set -u
DATA="${1:?usage: bash run_plan_only.sh <data dir> <scenario file>}"
SCEN="${2:?usage: bash run_plan_only.sh <data dir> <scenario file>}"
cd "$(dirname "$0")/.." || exit 1
run() {   # tag, split seed (0 = fixed)
    extra=""; [ "$2" != 0 ] && extra="--split-seed $2"
    python diagnose_sensor.py --data-dir "$DATA" --transitions --scenarios "$SCEN" --tag $1 $extra --plan-only \
        > ~/diagp_$1.txt 2>&1
}
run st_lc52 0
for s in 1 2 3; do run st_chk_d$s $s; done
{
for t in st_lc52 st_chk_d1 st_chk_d2 st_chk_d3; do
    echo "== $t"
    grep -E "^PLAN ONLY" ~/diagp_$t.txt | cut -c1-120
    grep -A 20 "^TEST scenarios  (relative" ~/diagp_$t.txt | grep -E "^  (mean|median)" | awk '{print "  " $1 ": strict " $2 " %, paper-style " $8 " %"}'
    grep -A 3 "the room average" ~/diagp_$t.txt | grep -E "seating|whole"
done
echo "== warning events summed over the three random splits (margin 0 / 25 / 50 ppm)"
for lim in 700 800; do
    for m in 0 25 50; do
        for s in 1 2 3; do grep -A 20 "^warning with a safety margin" ~/diagp_st_chk_d$s.txt | grep -E "^ +$lim\.? +$m +"; done \
            | awk -v l=$lim -v m=$m '{r+=$3; f+=$4; x+=$5; a+=$6} END {printf "  limit %s ppm, margin %2d ppm: real %d, found %d, missed %d, false alarms %d\n", l, m, r, f, x, a}'
    done
done
} > ~/plan_only_summary.txt
echo ALL_DONE >> ~/plan_only_summary.txt
