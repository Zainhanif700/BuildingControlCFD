#!/usr/bin/env bash
# Overnight parameter sweep for hfm_channel_flow_test.py
#
# Runs several parameter combinations ONE AFTER ANOTHER (not in parallel --
# safer if you're on a single GPU/CPU), saves each run's full log, and
# appends a one-line summary to results_summary.csv after each finishes.
# In the morning: check sweep_results/results_summary.csv for the table,
# or open any individual .log / .png for details on a specific run.
#
# USAGE (inside tmux, so it survives you closing the terminal):
#   tmux new -s sweep
#   bash run_overnight_sweep.sh
#   [press Ctrl+B then D to detach]
#   # next morning: tmux attach -t sweep   (or just read the results file)
#
# Make sure hfm_channel_flow_test.py is in the SAME directory as this script,
# or edit SCRIPT below to point at it.

set -uo pipefail  # (no -e: we want the sweep to continue even if one run fails)

SCRIPT="hfm_channel_flow_test.py"
OUTDIR="sweep_results"
SUMMARY="$OUTDIR/results_summary.csv"

mkdir -p "$OUTDIR"
if [ ! -f "$SUMMARY" ]; then
    echo "run_name,bc_sides,lambda_mom,lambda_bc,n_bc,mid_time_speed_err_pct,status" > "$SUMMARY"
fi

# --- Define the runs: name|extra args ------------------------------------
# Edit this list freely before launching. Each line is one full training run.
RUNS=(
    "left_baseline|--bc-sides left"
    "left_right|--bc-sides left,right"
    "left_top|--bc-sides left,top"
    "all_sides_upper_bound|--bc-sides all"
    "left_more_bc_points|--bc-sides left --n-bc 400"
    "left_higher_bc_weight|--bc-sides left --lambda-bc 50.0"
    "left_higher_mom_weight|--bc-sides left --lambda-mom 50.0"
)

echo "Starting sweep: ${#RUNS[@]} runs. Started at $(date)"
echo "Results will accumulate in $SUMMARY as each run finishes."
echo ""

for entry in "${RUNS[@]}"; do
    name="${entry%%|*}"
    args="${entry#*|}"
    log="$OUTDIR/${name}.log"
    png="$OUTDIR/${name}.png"

    echo "=== [$(date +%H:%M:%S)] Starting run: $name  (args: $args) ==="

    python3 -u "$SCRIPT" $args --out "$png" > "$log" 2>&1
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "  -> FAILED (exit code $exit_code), see $log"
        echo "$name,,,,,,FAILED" >> "$SUMMARY"
        continue
    fi

    # Pull out the fields we care about from the log.
    bc_sides=$(grep -oP '(?<=Boundary condition given on: )\[.*?\]' "$log" | tail -1)
    mid_err=$(grep -oP 'Primary result \(mid-time.*?speed error \K[0-9.]+(?=%)' "$log" | tail -1)
    lambda_mom=$(echo "$args" | grep -oP '(?<=--lambda-mom )[0-9.]+' || echo "10.0")
    lambda_bc=$(echo "$args" | grep -oP '(?<=--lambda-bc )[0-9.]+' || echo "10.0")
    n_bc=$(echo "$args" | grep -oP '(?<=--n-bc )[0-9]+' || echo "200")

    echo "  -> done. mid-time speed error: ${mid_err:-N/A}%"
    echo "$name,\"$bc_sides\",$lambda_mom,$lambda_bc,$n_bc,${mid_err:-N/A},OK" >> "$SUMMARY"
done

echo ""
echo "=== Sweep finished at $(date) ==="
echo "Summary table: $SUMMARY"
column -s, -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
