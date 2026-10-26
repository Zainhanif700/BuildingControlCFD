"""
Compares the runs of speed_test.sh: wall time and the difference of U, nut and CO2 (s) to a reference run.
Usage: python3 compare_runs.py speedtest/N12 current_1
"""
import argparse
import os
import re

import numpy as np

import common  # noqa: F401  (adds experiments/pinn to the path)
from compare_with_openfoam import read_internal, time_dirs


def wall_time(case):
    t = re.findall(r"^ExecutionTime = .* ClockTime = ([0-9.]+) s", open(os.path.join(case, "log.pimpleFoam")).read(), re.M)
    return float(t[-1]) if t else float("nan")


def last_fields(case):
    t, d = time_dirs(case)[-1]
    n = len(read_internal(os.path.join(case, "0", "C"), 3))
    return t, {f: read_internal(os.path.join(case, d, f), k, n) for f, k in (("U", 3), ("nut", 1), ("s", 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("reference")
    args = ap.parse_args()
    runs = sorted(r for r in os.listdir(args.folder) if os.path.isfile(os.path.join(args.folder, r, "log.pimpleFoam")))
    t_ref, ref = last_fields(os.path.join(args.folder, args.reference))
    w_ref = wall_time(os.path.join(args.folder, args.reference))
    print(f"t = {t_ref:g} s, difference to {args.reference} (relative l2):")
    print(f"  {'run':12s} {'wall time':>10s} {'speed-up':>9s} {'U':>8s} {'nut':>8s} {'CO2':>8s}")
    for r in runs:
        case = os.path.join(args.folder, r)
        t, f = last_fields(case)
        assert abs(t - t_ref) < 1e-6, f"{r} ended at {t} s"
        err = [np.linalg.norm(f[k] - ref[k]) / np.linalg.norm(ref[k]) for k in ("U", "nut", "s")]
        w = wall_time(case)
        print(f"  {r:12s} {w:9.0f}s {w_ref / w:8.1f}x " + " ".join(f"{100 * e:7.2f}%" for e in err))


if __name__ == "__main__":
    main()
