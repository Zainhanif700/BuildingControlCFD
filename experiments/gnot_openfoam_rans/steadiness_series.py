"""
Is the flow still developing or does it oscillate? For every saved time of an OpenFOAM case:
mean speed, change of U relative to the previous save (per 10 s) and relative to the save 30 s
earlier, the door split, and the mean nut. A steadily shrinking change = still settling (will become
steady); a change that stays large while the mean speed goes up and down = unsteady (oscillating).
Usage (training env, from experiments/gnot_openfoam_rans):
  python3 steadiness_series.py --case cases/W1_1ms_rans_dx0.1
"""
import argparse
import os

import numpy as np

import common  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    args = ap.parse_args()
    from compare_with_openfoam import read_internal, time_dirs, read_patch_sum
    case = os.path.abspath(args.case)
    C = read_internal(os.path.join(case, "0", "C"), 3)
    td = [(t, d) for t, d in time_dirs(case) if t > 0]
    U, out = {}, []
    print(f"{'t [s]':>6s} | {'mean |u|':>8s} | {'change vs t-10':>14s} {'vs t-30':>8s} | {'door split':>10s} | {'mean nut':>9s}")
    for t, d in td:
        U[t] = read_internal(os.path.join(case, d, "U"), 3, len(C))
        nutp = os.path.join(case, d, "nut")
        nut = read_internal(nutp, 1, len(C)).mean() if os.path.isfile(nutp) or os.path.isfile(nutp + ".gz") else float("nan")
        ch = lambda s: (np.linalg.norm(U[t] - U[s]) / np.linalg.norm(U[t])) if s in U else float("nan")
        phi = os.path.join(case, d, "phi")
        if os.path.isfile(phi) or os.path.isfile(phi + ".gz"):
            d1, d2 = read_patch_sum(phi, "door1"), read_patch_sum(phi, "door2")
            split = f"{d1 / (d1 + d2):10.3f}"
        else:
            split = f"{'-':>10s}"
        print(f"{t:6.0f} | {np.mean(np.linalg.norm(U[t], axis=1)):8.4f} | {100 * ch(t - 10.0):13.2f}% "
              f"{100 * ch(t - 30.0):7.2f}% | {split} | {nut:9.2e}")


if __name__ == "__main__":
    main()
