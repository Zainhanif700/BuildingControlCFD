"""
Old CO2 solver vs the mass-conserving one, both against OpenFOAM's own CO2 on a kept case.
Usage: python3 compare_solvers.py --case cases/V04_dx0.1
"""
import argparse
import os
import time

import numpy as np

import common
from common import N_REF

T_AVG = 150.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="cases/V04_dx0.1")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import torch
    import check_co2_with_model_flow as L2
    from compare_with_openfoam import read_internal, time_dirs
    from compare_rans_pilot import load_case, to_grid
    from fv_turb import TurbFV, seat_source
    from fv_cons import ConsFV
    case = os.path.abspath(args.case)
    meta, V, g, Cc, idx, td = load_case(case, read_internal, time_dirs, L2)
    fl = g.fluid
    rd = lambda t, f, n: np.nan_to_num(to_grid(g, idx, read_internal(os.path.join(case, td[t], f), n, len(Cc)) if n == 3
                                               else read_internal(os.path.join(case, td[t], f), 1, len(Cc)).reshape(-1)))
    times = sorted(t for t in td if t > 0)
    z = np.zeros(g.X.shape)
    snaps = {0.0: [z, z, z, z]}
    for t in times:
        u = rd(t, "U", 3) * fl[..., None]
        snaps[t] = [u[..., 0], u[..., 1], u[..., 2], rd(t, "nut", 1) * fl]
    mean = [np.mean([snaps[t][k] for t in times if t >= T_AVG], axis=0) for k in range(4)]
    T_real = [0.0] + times
    T_b2 = [0.0] + [t for t in times if t <= T_AVG]
    runs = {
        "real": (T_real, [snaps[t] for t in T_real]),
        "B2": (T_b2 + [T_b2[-1] + 10.0], [snaps[t] for t in T_b2] + [mean]),
    }
    t_chk = [t for t in (120.0, 300.0, 600.0) if t in td]
    src = seat_source(g, N_REF)
    res = {}
    for name, (T, S) in runs.items():
        for solver in ("old", "new"):
            t0 = time.time()
            if solver == "old":
                out, dt, n = TurbFV(g, src, args.device, torch.float32).solve(T, S, t_chk)
            else:
                fv = ConsFV(g, src, V, args.device, torch.float32)
                out, dt, n = fv.solve(T, S, t_chk, verbose=True)
            res[(name, solver)] = out
            print(f"{name:4s} {solver}: {n} steps, dt {dt:.4f} s, {time.time() - t0:.0f} s", flush=True)

    k11 = int(np.argmin(np.abs(g.c1d[2] - 1.1)))
    k16 = int(np.argmin(np.abs(g.c1d[2] - 1.6)))
    print(f"\n{'run':10s} {'t':>5s} | {'vol':>6s} {'plane':>6s} {'mass':>6s} | paper 1.1 / 1.6 m | min C")
    for name in runs:
        for solver in ("old", "new"):
            for t in t_chk:
                ref = rd(t, "s", 1) * fl
                c = res[(name, solver)][t]
                ev = np.linalg.norm(c[fl] - ref[fl]) / np.linalg.norm(ref[fl])
                pl = fl[:, :, k11]
                ep = np.linalg.norm(c[:, :, k11][pl] - ref[:, :, k11][pl]) / np.linalg.norm(ref[:, :, k11][pl])
                pap = []
                for k in (k11, k16):
                    m = fl[:, :, k]
                    r = ref[:, :, k][m] + common.PPM_OUTDOOR
                    pap.append(100 * np.linalg.norm(c[:, :, k][m] + common.PPM_OUTDOOR - r) / np.linalg.norm(r))
                print(f"{name + ' ' + solver:10s} {t:5.0f} | {100 * ev:5.1f}% {100 * ep:5.1f}% {c[fl].sum() / ref[fl].sum():6.3f} | "
                      f"{pap[0]:5.2f}% / {pap[1]:5.2f}% | {c[fl].min():7.2f}")
    print("\nReading: 'new' should have mass ~1.00 and smaller errors than 'old' (real and B2), min C >= 0.")


if __name__ == "__main__":
    main()
