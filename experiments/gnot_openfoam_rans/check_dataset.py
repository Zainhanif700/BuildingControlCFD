"""
Quick health check of the extracted RANS dataset files (data/S*.npz), no OpenFOAM needed:
  - all keys there, C has 61 frames 0..1800 s every 30 s, finite, not negative (small undershoot ok)
  - CO2 grows over time and levels off (room mean excess at 5/10/20/30 min, plane 1.1 m and 1.6 m)
  - flow: mean speed, fluctuation; check numbers from extraction (B2 vs OpenFOAM at 600 s)
Usage (training env, from experiments/gnot_openfoam_rans):  python3 check_dataset.py
"""
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ["P", "plane", "t_c", "C", "V", "N_ref", "U_mean", "nut_mean", "fluctuation", "err_real_vs_OF", "err_B2_vs_OF",
        "mass_B2_vs_OF"]


def main():
    files = sorted(glob.glob(os.path.join(HERE, "data", "S*.npz")))
    print(f"{len(files)} files\n")
    print(f"{'case':5s} {'ok':3s} {'min C':>7s} | mean excess ppm room @5/10/20/30 min | plane 1.1 / 1.6 m @30 | "
          f"speed  fluct | B2 vs OF @600 (excess) | issues")
    for f in files:
        d = np.load(f)
        name = os.path.basename(f)[:-4]
        issues = [k for k in KEYS if k not in d.files]
        if issues:
            print(f"{name:5s} NO  missing keys {issues}")
            continue
        t, C, P = d["t_c"], d["C"], d["P"]
        if C.shape[0] != 61 or not np.allclose(t, np.arange(0, 1801, 30)):
            issues.append(f"time grid {C.shape[0]} frames to {t[-1]:g}")
        if not np.isfinite(C).all():
            issues.append("NaN/inf in C")
        mx = float(C.max())
        if C.min() < -0.02 * mx:
            issues.append(f"negative C {C.min():.2f}")
        m = C.mean(1)
        at = lambda s: m[int(round(s / 30))]
        if not (at(300) < at(600) < at(1200) <= at(1800) * 1.02):
            issues.append("CO2 not rising")
        zs = np.unique(P[:, 2])
        pl = lambda z: np.abs(P[:, 2] - zs[np.argmin(np.abs(zs - z))]) < 1e-5
        sp = float(np.linalg.norm(d["U_mean"], axis=1).mean())
        print(f"{name:5s} {'OK' if not issues else 'NO':3s} {C.min():7.2f} | {at(300):6.1f} {at(600):6.1f} {at(1200):6.1f} "
              f"{at(1800):6.1f}          | {C[-1][pl(1.1)].mean():6.1f} / {C[-1][pl(1.6)].mean():6.1f}      | "
              f"{sp:.3f} {100 * float(d['fluctuation']):4.0f}% | {100 * float(d['err_B2_vs_OF']):5.1f}% mass "
              f"{float(d['mass_B2_vs_OF']):.3f}     | {'; '.join(issues)}")
    print("\nOK = file complete and physically plausible (CO2 positive, rising, levelling off).")


if __name__ == "__main__":
    main()
