#!/usr/bin/env bash
# One-time setup on a new machine (no root needed): Miniforge in ~/anaconda3 with the envs foam (OpenFOAM v2412)
# and cfd (Python, PyTorch on CPU). Parts that already exist are skipped.
# Usage: bash setup_env.sh
set -eu
if [ ! -d "$HOME/anaconda3" ]; then
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$HOME/anaconda3"
    rm /tmp/miniforge.sh
fi
source "$HOME/anaconda3/etc/profile.d/conda.sh"
PY="$HOME/anaconda3/envs/cfd/bin/python"
conda env list | grep -q "^foam " || conda create -y -n foam -c conda-forge openfoam=2412
conda env list | grep -q "^cfd "  || conda create -y -n cfd -c conda-forge python=3.11 numpy scipy matplotlib
"$PY" -c "import torch" 2> /dev/null || "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"$PY" -c "import numpy, torch; print('python OK')"
( conda activate foam && command -v pimpleFoam > /dev/null && echo "OpenFOAM OK" )
