"""
Track 2 (data-driven): GNOT trained on OpenFOAM data, in the spirit of Bian, Schmidt & Shi (2025),
arXiv:2504.21243. Everything physical (geometry, CO2 source, FV CO2 solver, OpenFOAM readers,
GNOT backbone) is IMPORTED from experiments/gnot -- one source of truth, no copies. Nothing in
experiments/gnot is modified by this track.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GNOT_DIR = os.path.normpath(os.path.join(HERE, "..", "gnot"))
OPENFOAM_DIR = os.path.join(GNOT_DIR, "openfoam")
for p in (GNOT_DIR, OPENFOAM_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

DATA_DIR = os.path.join(HERE, "data")            # extracted datasets (npz), not committed
CKPT_DIR = os.path.join(HERE, "checkpoints")     # not committed
N_REF = 20.0                                     # occupancy of the FV CO2 solve; C is exactly linear in N
T_DATA = [float(t) for t in range(0, 121, 10)]   # times stored in a dataset [s]
