"""
Our CO2 solver (the one used for the dataset) on the OpenFOAM flow saved every 1, 2, 5 or 10 s,
compared with OpenFOAM's own CO2 at the same times. Only reads the case.
Usage: python3 compare_sampling.py --case cases/N28ws1_dx0.08
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "gnot_openfoam_rans")))
import common
from common import N_REF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--every", default="1,2,5,10", help="save intervals to test [s]")
    ap.add_argument("--check", default="30,60,90,120,150", help="times to compare with OpenFOAM [s]")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import torch
    import check_co2_with_model_flow as L2
    from compare_with_openfoam import read_internal, time_dirs
    from compare_rans_pilot import load_case, to_grid
    from fv_turb import seat_source
    from fv_cons import ConsFV

    case = os.path.abspath(args.case)
    meta, V, g, Cc, idx, td = load_case(case, read_internal, time_dirs, L2)
    fl = g.fluid
    times = sorted(t for t in td if t > 0)
    doors = [int(x) for x in meta.get("doors", "1 1").split()]
    check = [float(t) for t in args.check.split(",") if float(t) in td]
    rd = lambda t, f, n: np.nan_to_num(to_grid(g, idx, read_internal(os.path.join(case, td[t], f), n, len(Cc)).reshape(len(Cc), -1)))
    t0 = time.time()
    z = np.zeros(g.X.shape, np.float32)
    snaps = {0.0: [z, z, z, z]}
    for t in times:
        u = (rd(t, "U", 3).reshape(g.X.shape + (3,)) * fl[..., None]).astype(np.float32)
        nut = (rd(t, "nut", 1).reshape(g.X.shape) * fl).astype(np.float32)
        snaps[t] = [u[..., 0], u[..., 1], u[..., 2], nut]
    ref = {t: rd(t, "s", 1).reshape(g.X.shape)[fl] for t in check}
    print(f"{os.path.basename(case)}: V = {V}, doors {doors}; {len(times)} OpenFOAM saves up to {times[-1]:g} s "
          f"(read in {time.time() - t0:.0f} s)")
    print("CO2 error vs OpenFOAM (relative l2, all cells) / total CO2 ours:OpenFOAM")
    print("save every | " + " | ".join(f"{t:5g} s        " for t in check))
    src = seat_source(g, N_REF)
    for every in [float(e) for e in args.every.split(",")]:
        ts = [0.0] + [t for t in times if abs(t / every - round(t / every)) < 1e-6]
        t1 = time.time()
        out, _, _ = ConsFV(g, src, V, args.device, torch.float32, doors=doors).solve(ts, [snaps[t] for t in ts], check)
        cells = []
        for t in check:
            c = out[t][fl]
            cells.append(f"{100 * np.linalg.norm(c - ref[t]) / np.linalg.norm(ref[t]):5.1f} % {c.sum() / ref[t].sum():5.3f}")
        print(f"{every:6g} s   | " + " | ".join(cells) + f"   ({len(ts) - 1} saves, {time.time() - t1:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
