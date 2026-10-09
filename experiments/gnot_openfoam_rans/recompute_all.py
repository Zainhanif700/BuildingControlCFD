"""
Recomputes the CO2 of all dataset files from the stored flow (skips files that are already up to date).
Usage: python3 recompute_all.py
"""
import argparse
import glob
import os

import numpy as np

import common

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default=common.DATA_CO2_SOLVER, choices=["old", "cons"])
    ap.add_argument("--sct", type=float, default=common.SC_T_DATA)
    ap.add_argument("--pattern", default="S*.npz")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    from extract_rans import recompute
    files = sorted(glob.glob(os.path.join(HERE, "data", args.pattern)))
    todo = []
    for f in files:
        try:
            d = np.load(f)
            d.files
        except Exception as e:
            print(f"  {os.path.basename(f)}: not readable now ({e.__class__.__name__}) -- skipped, run again later")
            continue
        done = ("co2_solver" in d.files and str(d["co2_solver"]) == args.solver
                and "sc_t" in d.files and abs(float(d["sc_t"]) - args.sct) < 1e-6)
        if not done:
            todo.append(f)
    print(f"{len(files)} files, {len(todo)} to recompute (solver {args.solver}, Sc_t {args.sct:g})", flush=True)
    for i, f in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {os.path.basename(f)}", flush=True)
        recompute(f, args.device, args.solver, args.sct)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
