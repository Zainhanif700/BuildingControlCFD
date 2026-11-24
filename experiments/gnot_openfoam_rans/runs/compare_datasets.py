"""
Compares two dataset files of the same window setting (e.g. the same case with two OpenFOAM settings):
mean flow, CO2 over the 30 min and on the breathing plane, as relative l2 differences.
Usage: python3 runs/compare_datasets.py data_v2/N12.npz data_check/N12mc2.npz
Updated 2026-11-24: moved to runs/ (it was used for the comparison of the first and the corrected dataset).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("other")
    args = ap.parse_args()
    a, b = np.load(args.other), np.load(args.reference)
    assert np.allclose(a["P"], b["P"]) and np.allclose(a["V"], b["V"]), "not the same mesh / window setting"
    t, m = b["t_c"], b["plane"]
    print(f"{args.other} vs {args.reference} (relative l2 difference)")
    print(f"  mean flow (150-600 s): U {100 * rel(a['U_mean'], b['U_mean']):.2f} %, nut {100 * rel(a['nut_mean'], b['nut_mean']):.2f} %")
    print(f"  CO2 excess, all points and times: {100 * rel(a['C'][1:], b['C'][1:]):.2f} %")
    for tt in (60, 300, 600, 1200, 1800):
        k = int(np.argmin(np.abs(t - tt)))
        print(f"  t = {t[k]:5.0f} s: room {100 * rel(a['C'][k], b['C'][k]):6.2f} %, plane {100 * rel(a['C'][k][m], b['C'][k][m]):6.2f} %, "
              f"paper metric (400 + excess, plane) {100 * rel(a['C'][k][m] + common.PPM_OUTDOOR, b['C'][k][m] + common.PPM_OUTDOOR):5.2f} %, "
              f"room mean {a['C'][k].mean():7.1f} vs {b['C'][k].mean():7.1f} ppm")


if __name__ == "__main__":
    main()
