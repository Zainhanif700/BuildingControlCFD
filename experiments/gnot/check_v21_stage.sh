#!/usr/bin/env bash
# Checks one v21 checkpoint against the OpenFOAM case with the same viscosity.
# Usage: bash check_v21_stage.sh <checkpoint> [--gpu]
set -u
CKPT="${1:?usage: bash check_v21_stage.sh <checkpoint>}"
[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT"; exit 1; }
[ -f staged_smoke_test.py ] || { echo "run this from experiments/gnot"; exit 1; }
# default CPU (never disturbs a running training); 2nd argument --gpu when the GPU is idle
DEV=cpu; [ "${2:-}" = "--gpu" ] && DEV=cuda
[ "$DEV" = cpu ] && export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
NU=$(python3 -c "import torch,sys; print(float(torch.load(sys.argv[1], map_location='cpu').get('nu', 0.01)))" "$CKPT") \
    || { echo "could not read nu from $CKPT"; exit 1; }
CASE=$(python3 -c "import sys; nu=float(sys.argv[1]); print('openfoam/cases/W1_1ms_dx0.1' + ('' if nu == 0.01 else f'_nu{nu:g}'))" "$NU")
[ -f "$CASE/scenario.txt" ] || { echo "OpenFOAM case $CASE missing -- make and run it first (nu = $NU)"; exit 1; }
ls "$CASE" | grep -qx 120 || { echo "OpenFOAM case $CASE has no t = 120 s result -- run it first"; exit 1; }
CKPT_ABS=$(realpath "$CKPT"); CASE_ABS=$(realpath "$CASE")   # absolute or relative paths both work
mkdir -p logs/checks
LOG="logs/checks/v21_$(basename "$CKPT" .pth).log"
{
    echo "v21 check: $CKPT  (nu = $NU, OpenFOAM case $CASE, $(date))"
    echo; echo "== 1 compare with OpenFOAM"
    (cd openfoam && python3 compare_with_openfoam.py --case "$CASE_ABS" --checkpoint "$CKPT_ABS" --device $DEV) \
        || echo "-- compare_with_openfoam FAILED"
    echo; echo "== 2 flow-correction diagnosis"
    (cd openfoam && python3 diagnose_flow_correction.py --case "$CASE_ABS" --checkpoint "$CKPT_ABS" --device $DEV) \
        || echo "-- diagnose_flow_correction FAILED"
    echo; echo "Reference numbers (nu 0.01): v19 velocity error 74% (volume) / 67% (plane); B_p alone 71% / 65%;"
    echo "v19 alignment 0.14, achieved 0.05. v21 is a success if the error is clearly below B_p's and"
    echo "the alignment clearly above 0.14."
} 2>&1 | tee "$LOG"
echo "log: $LOG"
