#!/bin/bash
# Demo, version 2026-11-23. Two small tests that run alone or together:
#   bash run_demo.sh forecast   the final model of the thesis forecasts the CO2 of one test scenario (N14).
#                               Nothing is trained. It needs a GPU or a CPU and takes a short time (not timed).
#   bash run_demo.sh openfoam   OpenFOAM, scenario N01 for 20 s of flow on 4 cores (OpenFOAM v2412).
#                               About 10 minutes on our lab PC (measured: 10.5 min, solver 500 s).
#   bash run_demo.sh            both, one after the other.
# The two parts are independent: the 20 s OpenFOAM run does not feed the model.
# Before the first run, once: bash setup_env.sh   (it should print "python OK" and "OpenFOAM OK").
# Files for the forecast: demo/final_model/member0..4.pth and demo/N14.npz (CO2 of the scenario).
# Other Python: CFD_PY=/path/to/python bash run_demo.sh ...    Other number of cores: NP=8 bash run_demo.sh openfoam
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
PY="$(command -v "${CFD_PY:-python3}")"
NP="${NP:-4}"
WHAT="${1:-all}"
echo "=== demo $(date '+%F %T'), python $PY"

if [ "$WHAT" = forecast ] || [ "$WHAT" = all ]; then
    echo "=== forecast with the final model"
    [ -f demo/N14.npz ] && [ -f demo/final_model/member4.pth ] || { echo "missing demo/N14.npz or demo/final_model/member*.pth"; exit 1; }
    "$PY" demo_forecast.py --ckpt demo/final_model --file demo/N14.npz --plot demo_forecast.png || exit 1
fi

if [ "$WHAT" = openfoam ] || [ "$WHAT" = all ]; then
    echo "=== OpenFOAM, N01, 20 s, $NP cores"
    CFD_PY="$PY" T_END=20 bash run_batch.sh scenarios_v2.txt 1 "$NP" N01 || exit 1
    L=cases/N01_test_dx0.08/log.pimpleFoam
    [ -f "$L" ] || { echo "no solver log found: $L"; exit 1; }
    echo "last solver time: $(grep '^Time =' "$L" | tail -1)   (20 means finished)"
    grep '^ExecutionTime' "$L" | tail -1
    ls cases/N01_test_dx0.08/done > /dev/null 2>&1 && echo "OpenFOAM part OK (case in cases/N01_test_dx0.08, about 160 MB; delete it when you do not need it)"
fi
echo "=== demo finished $(date '+%F %T')"
