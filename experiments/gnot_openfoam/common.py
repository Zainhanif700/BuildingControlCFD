"""
Settings of the first data track (laminar OpenFOAM data); geometry and solvers are imported from experiments/gnot.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GNOT_DIR = os.path.normpath(os.path.join(HERE, "..", "gnot"))
OPENFOAM_DIR = os.path.join(GNOT_DIR, "openfoam")
for p in (GNOT_DIR, OPENFOAM_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

DATA_DIR = os.path.join(HERE, "data")
CKPT_DIR = os.path.join(HERE, "checkpoints")
N_REF = 20.0
T_DATA = [float(t) for t in range(0, 121, 10)]
