"""
Checks of the dataset CO2: negative values, and that the CO2 can be recomputed from the stored flow.
Usage: python3 check_rans_data.py [--pilot <case> | --recompute-test data/S03.npz]
"""
import argparse
import glob
import os
import shutil
import tempfile

import numpy as np

import common

HERE = os.path.dirname(os.path.abspath(__file__))
Y_WIN = 9.16


def where_negative(P, neg, seat_box, band=0.3):
    (x0, x1), (y0, y1), (z0, z1) = seat_box
    x, y, z = P[neg, 0], P[neg, 1], P[neg, 2]
    win = y > Y_WIN - band
    inside = (x > x0 - band) & (x < x1 + band) & (y > y0 - band) & (y < y1 + band) & (z > z0 - band) & (z < z1 + band)
    edge = inside & ~win
    return int(win.sum()), int(edge.sum()), int((~win & ~edge).sum())


def stats(c, P, seat_box, label):
    neg = c < -1.0
    negmass = -c[c < 0].sum() / max(c[c > 0].sum(), 1e-30)
    w, e, o = where_negative(P, neg, seat_box)
    zs = np.unique(P[:, 2])
    pl = lambda zz: np.abs(P[:, 2] - zs[np.argmin(np.abs(zs - zz))]) < 1e-5
    p11, p16 = c[pl(1.1)], c[pl(1.6)]
    print(f"  {label:>8s} | min {c.min():7.2f} max {c.max():8.1f} | cells < -1 ppm {neg.sum():6d} ({100 * neg.mean():5.2f} %) "
          f"neg. mass {100 * negmass:5.2f} % | where: window {w}, seat-box edge {e}, other {o} | "
          f"plane 1.1 min {p11.min():6.2f} ({100 * (p11 < -1).mean():4.1f} %), 1.6 min {p16.min():6.2f} ({100 * (p16 < -1).mean():4.1f} %)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default=None)
    ap.add_argument("--recompute-test", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.recompute_test:
        from extract_rans import recompute
        src = os.path.abspath(args.recompute_test)
        tmp = os.path.join(tempfile.mkdtemp(), os.path.basename(src))
        shutil.copy(src, tmp)
        d0 = np.load(src)
        solver = str(d0["co2_solver"]) if "co2_solver" in d0.files else "old"
        sct = float(d0["sc_t"]) if "sc_t" in d0.files else common.SC_T
        recompute(tmp, args.device, solver, sct)
        a, b = np.load(src)["C"], np.load(tmp)["C"]
        rel = float(np.abs(a - b).max() / np.abs(a).max())
        print(f"recompute test {src}: max |C_new - C_stored| / max C = {rel:.2e}  -> "
              f"{'OK (flow is stored completely; CO2 can be redone later)' if rel < 1e-3 else 'DIFFERENT -- tell me'}")
        os.remove(tmp)
        return

    if args.pilot:
        from compare_with_openfoam import read_internal, time_dirs
        from compare_rans_pilot import load_case, to_grid
        import check_co2_with_model_flow as L2
        case = os.path.abspath(args.pilot)
        meta, V, g, Cc, idx, td = load_case(case, read_internal, time_dirs, L2)
        fl = g.fluid
        P = np.stack([g.X[fl], g.Y[fl], g.Z[fl]], 1)
        print(f"OpenFOAM CO2 (s) in the pilot {case}:")
        for t in (300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0):
            fs = os.path.join(case, td[t], "s") if t in td else ""
            if not (t in td and (os.path.exists(fs) or os.path.exists(fs + ".gz"))):
                print(f"  {t:>6g} s: no saved s field")
                continue
            if True:
                s = np.nan_to_num(to_grid(g, idx, read_internal(os.path.join(case, td[t], "s"), 1, len(Cc)).reshape(-1)))
                stats(s[fl], P, common.SEAT_BOX, f"{t:g} s")
        return

    for f in sorted(glob.glob(os.path.join(HERE, "data", "S*.npz"))):
        d = np.load(f)
        print(f"{os.path.basename(f)}: V = {d['V'].tolist()}")
        for i in (10, 20, 40, 60):
            stats(d["C"][i], d["P"], d["seat_box"], f"{30 * i} s")
    print("\nReading: negative mass well below 1 % and only at sharp edges (window inflow, seat-box edge) ->"
          "\n         harmless numerical undershoot (absolute CO2 = 400 + C stays ~385-400 ppm there);"
          "\n         clipping C at 0 before training changes almost nothing. Widespread or large -> fix the scheme"
          "\n         and recompute all files from the stored flow (~30 min GPU per file, no OpenFOAM).")


if __name__ == "__main__":
    main()
