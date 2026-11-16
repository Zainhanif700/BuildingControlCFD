"""
Compares the airflow of the same case run with different OpenFOAM versions (or builds).
Usage: python compare_versions.py A=<case folder> B=<case folder> C=<case folder>
       python compare_versions.py v2412apt=~/foam_compare/v2412apt/N01ws1_dx0.08 v2606=... v2412conda=...
For every pair of runs it prints
  - if the meshes are the same (cell centres in the same order), otherwise the field difference means nothing
  - the relative difference of the velocity field at some times and of the mean over the last 50 s
  - the room mean of the speed in both runs
"""
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pinn", "openfoam"))
from compare_with_openfoam import read_internal      # reads an OpenFOAM field file (plain or .gz)

TIMES = [30, 60, 90, 120, 150]
MEAN_FROM, MEAN_TO = 100, 150


def read_case(path):
    path = os.path.expanduser(path)
    centres = read_internal(os.path.join(path, "0", "C"), 3)
    return {"path": path, "C": centres}


def velocity(case, t):
    return read_internal(os.path.join(case["path"], f"{t:g}", "U"), 3, n_cells=len(case["C"]))


def mean_velocity(case, t0, t1):
    fields = [velocity(case, t) for t in range(t0, t1 + 1)]
    return np.mean(fields, axis=0)


def rel_diff(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)


def speed(u):
    return np.linalg.norm(u, axis=1)


def main():
    runs = {}
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        runs[name] = read_case(path)
    if len(runs) < 2:
        sys.exit(__doc__)

    for name, c in runs.items():
        print(f"{name}: {len(c['C'])} cells")
    means = {n: mean_velocity(c, MEAN_FROM, MEAN_TO) for n, c in runs.items()}
    for n, u in means.items():
        print(f"{n}: room mean of the speed, mean over {MEAN_FROM}-{MEAN_TO} s: {speed(u).mean():.4f} m/s")

    print()
    for a, b in itertools.combinations(runs, 2):
        ca, cb = runs[a], runs[b]
        if len(ca["C"]) != len(cb["C"]) or np.abs(ca["C"] - cb["C"]).max() > 1e-6:
            print(f"{a} vs {b}: the meshes are NOT the same (cell number or order) -- field difference not computed")
            continue
        row = [f"{t} s {100 * rel_diff(velocity(ca, t), velocity(cb, t)):5.1f} %" for t in TIMES]
        print(f"{a} vs {b}: same mesh. Difference of the velocity field: " + ", ".join(row))
        print(f"{'':{len(a) + len(b) + 5}}mean over {MEAN_FROM}-{MEAN_TO} s: {100 * rel_diff(means[a], means[b]):.1f} %")
    print("\n(difference = norm of the velocity difference / norm of the first run's velocity, all cells equally weighted)")


if __name__ == "__main__":
    main()
