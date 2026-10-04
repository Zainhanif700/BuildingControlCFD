"""
Recompute the CO2 of all extracted files with the chosen solver and Sc_t (from the stored flow, no
OpenFOAM). Files already done with these settings are skipped, so it can be re-run any time (e.g. again
when the dataset has produced more files). One file at a time, ~30 min each on the GPU.
Usage (training env, from experiments/gnot_openfoam_rans):
  python3 recompute_all.py --solver cons --sct 0.3
"""
import argparse
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="cons", choices=["old", "cons"])
    ap.add_argument("--sct", type=float, required=True)
    ap.add_argument("--pattern", default="S*.npz")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    from extract_rans import recompute
    files = sorted(glob.glob(os.path.join(HERE, "data", args.pattern)))
    todo = []
    for f in files:
        d = np.load(f)
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
