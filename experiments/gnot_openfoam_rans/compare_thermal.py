"""
Compares a normal OpenFOAM case with the thermal case of the same scenario at the same time (last saved time):
airflow U and CO2 (s) in the whole room and on the plane at 1.58 m (the plane of the model), and the temperature.
Usage: python3 compare_thermal.py cases_thermal/N01_ref_dx0.08 cases_thermal/N01_thermal_dx0.08
"""
import os
import sys

import numpy as np

import common  # noqa: F401  (adds experiments/pinn to the path)
from compare_with_openfoam import read_internal, time_dirs


def have(case, d, f):
    return any(os.path.isfile(os.path.join(case, d, f + e)) for e in ("", ".gz"))


def load(case):
    t, d = time_dirs(case)[-1]
    C = read_internal(os.path.join(case, "0", "C"), 3)
    order = np.lexsort(np.round(C, 5).T[::-1])          # same cell order in both cases
    out = {"t": t, "C": C[order]}
    for f, k in (("U", 3), ("s", 1), ("T", 1)):
        out[f] = read_internal(os.path.join(case, d, f), k, len(C))[order] if have(case, d, f) else None
    return out


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ref_case, th_case = sys.argv[1], sys.argv[2]
    A, B = load(ref_case), load(th_case)                 # A = normal (reference), B = with temperature
    assert A["t"] == B["t"], f"different last times: {A['t']} and {B['t']}"
    assert A["C"].shape == B["C"].shape and np.allclose(A["C"], B["C"], atol=1e-4), "the two meshes are not the same"
    zs = np.unique(np.round(A["C"][:, 2], 3))
    z = zs[np.argmin(np.abs(zs - 1.58))]
    plane = np.abs(A["C"][:, 2] - z) < 1e-3
    print(f"time {A['t']:g} s, {len(A['C'])} cells, plane z = {z:.3f} m with {int(plane.sum())} cells")
    print("difference of the case with temperature to the normal case (relative l2):")
    for name, f in (("airflow U", "U"), ("CO2 s", "s")):
        a, b = A[f], B[f]
        print(f"  {name:10s} whole room {rel(b, a):7.2%}   plane {rel(b[plane], a[plane]):7.2%}")
    uz_a, uz_b = A["U"][plane, 2], B["U"][plane, 2]
    print(f"  mean vertical speed Uz on the plane [m/s]: normal {uz_a.mean():+.4f}   thermal {uz_b.mean():+.4f}")
    sa, sb = A["s"][plane].reshape(-1), B["s"][plane].reshape(-1)
    print(f"  mean CO2 s on the plane: normal {sa.mean():.4g}   thermal {sb.mean():.4g}   (ratio {sb.mean() / sa.mean():.3f})")
    if B["T"] is not None:
        T = B["T"].reshape(-1)
        print(f"  temperature [K] thermal case: room min {T.min():.2f}  mean {T.mean():.2f}  max {T.max():.2f};"
              f"  plane mean {T[plane].mean():.2f}")
    print("note: only one scenario and a short flow time; this says nothing about later times.")


if __name__ == "__main__":
    main()
