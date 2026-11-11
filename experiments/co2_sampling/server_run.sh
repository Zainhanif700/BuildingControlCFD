#!/usr/bin/env bash
# For an extra server: install OpenFOAM v2412 and Python with conda (in ~/anaconda3, nothing outside the home folder),
# then run OpenFOAM for a list of all-openings settings at low priority. Everything is removed again afterwards with
#   rm -rf ~/BuildingControlCFD ~/anaconda3 ~/.conda ~/cfdpy ~/v4_*.log
# Usage: bash server_run.sh setup
#        bash server_run.sh run <cores per setting> <name> [<name> ...]
set -eu
MODE="${1:?usage: bash server_run.sh setup | run <cores> <names...>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ "$MODE" = setup ]; then
    cd ~
    if [ ! -d ~/anaconda3 ]; then
        wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh
        bash miniforge.sh -b -p ~/anaconda3
        rm miniforge.sh
    fi
    source ~/anaconda3/etc/profile.d/conda.sh
    conda env list | grep -q "^foam " || conda create -y -n foam -c conda-forge openfoam=2412
    conda env list | grep -q "^cfd "  || conda create -y -n cfd -c conda-forge python=3.11 numpy scipy matplotlib
    ~/anaconda3/envs/cfd/bin/python -c "import torch" 2>/dev/null || \
        ~/anaconda3/envs/cfd/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    printf '#!/bin/bash\nexport LD_LIBRARY_PATH="$HOME/anaconda3/envs/cfd/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\nexec "$HOME/anaconda3/envs/cfd/bin/python" "$@"\n' > ~/cfdpy
    chmod +x ~/cfdpy
    ~/cfdpy -c "import torch, numpy; print('python OK')"
    (source "$HERE/../gnot_openfoam_rans/foam_env.sh" && echo "OpenFOAM $WM_PROJECT_VERSION OK")
    echo "[$(date '+%F %T')] setup done, free disk: $(df -h ~ | tail -1 | awk '{print $4}')"
    exit 0
fi
NP="${2:?cores}"; shift 2
for n in "$@"; do
    CFD_PY=$HOME/cfdpy nice -n 10 bash "$HERE/run_full_case.sh" "$n" "$NP" || echo "[$(date '+%F %T')] $n FAILED"
done
echo "[$(date '+%F %T')] ALL DONE"
