#!/usr/bin/env bash
# One-time setup of the Python packages with pip, inside the environment you want to use, e.g.
#   python3 -m venv ~/venv-cfd && source ~/venv-cfd/bin/activate && bash setup_env.sh
# OpenFOAM v2412 is not a Python package: install it separately and source its etc/bashrc
# (or set FOAM_BASHRC=/path/to/OpenFOAM/etc/bashrc). It is only needed for the simulations.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
python3 -m pip install -r "$HERE/../../requirements.txt"
python3 -c "import numpy, torch, matplotlib; print('python OK: numpy', numpy.__version__, '| torch', torch.__version__)"
( source "$HERE/foam_env.sh" && echo "OpenFOAM OK" ) || echo "OpenFOAM not found yet (only needed for the simulations)"
