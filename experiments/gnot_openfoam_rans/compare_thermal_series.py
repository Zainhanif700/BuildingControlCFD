"""
Thermal case against the normal case of the same scenario at every saved time: does the difference grow, stay, or fade?
Per time: CO2 s on the plane at 1.58 m (mean, normal and thermal), relative l2 difference of U and s on the plane,
mean vertical speed on the plane, and the temperature minimum / mean (thermal). At the end the largest Courant number
of both runs.
Usage: python3 compare_thermal_series.py cases_thermal/N01_ref_dx0.08 cases_thermal/N01_thermal_dx0.08
"""
import os
import re
import sys

import numpy as np

import common  # noqa: F401  (adds experiments/pinn to the path)
from compare_with_openfoam import read_internal, time_dirs


def have(case, d, f):
    return any(os.path.isfile(os.path.join(case, d, f + e)) for e in ("", ".gz"))


def courant_max(case):
    best = 0.0
    for f in os.listdir(case):
        if f.startswith("log.") and f.split(".", 1)[1] in ("pimpleFoam", "buoyantBoussinesqPimpleFoam"):
            for m in re.finditer(r"Courant Number mean: \S+ max: (\S+)", open(os.path.join(case, f)).read()):
                best = max(best, float(m.group(1)))
    return best


def main():
    ref, th = sys.argv[1], sys.argv[2]
    C = read_internal(os.path.join(ref, "0", "C"), 3)
    C2 = read_internal(os.path.join(th, "0", "C"), 3)
    o1, o2 = (np.lexsort(np.round(c, 5).T[::-1]) for c in (C, C2))
    assert C.shape == C2.shape and np.allclose(C[o1], C2[o2], atol=1e-4), "the two meshes are not the same"
    zs = np.unique(np.round(C[:, 2], 3))
    z = zs[np.argmin(np.abs(zs - 1.58))]
    plane = (np.abs(C[o1][:, 2] - z) < 1e-3)
    ta = {t: d for t, d in time_dirs(ref)}
    tb = {t: d for t, d in time_dirs(th)}
    times = sorted(t for t in ta if t in tb and t > 0)
    rd = lambda case, d, f, k, o: read_internal(os.path.join(case, d, f), k, len(C))[o]
    print(f"plane z = {z:.3f} m, {int(plane.sum())} cells; difference = thermal to normal (relative l2 on the plane)")
    print(f"{'t [s]':>6s} {'s normal':>10s} {'s thermal':>10s} {'diff s':>8s} {'diff U':>8s} "
          f"{'Uz normal':>10s} {'Uz thermal':>10s} {'T min [K]':>10s} {'T mean [K]':>10s}")
    for t in times:
        if not (have(ref, ta[t], "U") and have(th, tb[t], "U") and have(ref, ta[t], "s") and have(th, tb[t], "s")):
            continue
        Ua, Ub = rd(ref, ta[t], "U", 3, o1)[plane], rd(th, tb[t], "U", 3, o2)[plane]
        sa, sb = rd(ref, ta[t], "s", 1, o1).reshape(-1)[plane], rd(th, tb[t], "s", 1, o2).reshape(-1)[plane]
        rel = lambda a, b: np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
        if have(th, tb[t], "T"):
            T = rd(th, tb[t], "T", 1, o2).reshape(-1)
            tmin, tmean = T.min(), T.mean()
        else:
            tmin = tmean = float("nan")
        print(f"{t:6g} {sa.mean():10.4g} {sb.mean():10.4g} {rel(sb, sa):8.1%} {rel(Ub, Ua):8.1%} "
              f"{Ua[:, 2].mean():+10.4f} {Ub[:, 2].mean():+10.4f} {tmin:10.2f} {tmean:10.2f}")
    print(f"largest Courant number: normal {courant_max(ref):.2f}, thermal {courant_max(th):.2f}")
    print("note: one scenario (N01); the thermal case assumes 75 W per person and adiabatic walls.")


if __name__ == "__main__":
    main()
