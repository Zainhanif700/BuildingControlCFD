"""
The RANS flow is statistically steady but keeps fluctuating (pilot: mean speed constant from ~150 s,
instantaneous field changes 25-35 % per 10-30 s). Can the CO2 be computed on the TIME-AVERAGED flow?
Measured against OpenFOAM's own CO2 transport on the real fluctuating flow (scalarTransport, field s):

  run 1  fv_turb on ALL saved snapshots (every 10 s, linear in between)   -> solver on the true flow
  run 2  fv_turb on the snapshots up to T_AVG_START, then the AVERAGE of U and nut over
         [T_AVG_START, last save] frozen                                   -> error of the averaged flow

Also prints how far each instantaneous field is from the average (the size of the fluctuations).
Usage (training env, from experiments/gnot_openfoam_rans):
  python3 check_mean_flow_co2.py --case cases/W1_1ms_rans_dx0.1 [--avg-start 150]
"""
import argparse
import os

import numpy as np

import common
from common import N_REF, BREATHING_Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--avg-start", type=float, default=150.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import torch
    import check_co2_with_model_flow as L2
    from compare_with_openfoam import read_internal, time_dirs
    from compare_rans_pilot import load_case, to_grid
    from fv_turb import TurbFV, seat_source
    case = os.path.abspath(args.case)
    meta, V, g, Cc, idx, td = load_case(case, read_internal, time_dirs, L2)
    fl = g.fluid
    kz = int(np.argmin(np.abs(g.c1d[2] - BREATHING_Z)))
    pl = fl[:, :, kz]
    times = sorted(t for t in td if t > 0)
    z = np.zeros(g.X.shape)
    snaps = {0.0: [z, z, z, z]}
    for t in times:
        u = np.nan_to_num(to_grid(g, idx, read_internal(os.path.join(case, td[t], "U"), 3, len(Cc)))) * fl[..., None]
        nut = np.nan_to_num(to_grid(g, idx, read_internal(os.path.join(case, td[t], "nut"), 1, len(Cc)).reshape(-1))) * fl
        snaps[t] = [u[..., 0], u[..., 1], u[..., 2], nut]
    avg_t = [t for t in times if t >= args.avg_start]
    mean = [np.mean([snaps[t][k] for t in avg_t], axis=0) for k in range(4)]
    um = np.stack(mean[:3], -1)[fl]
    dev = [np.linalg.norm(np.stack(snaps[t][:3], -1)[fl] - um) / np.linalg.norm(um) for t in avg_t]
    print(f"case {case}: {len(times)} saves to t = {times[-1]:g} s; average over {len(avg_t)} saves from {args.avg_start:g} s")
    print(f"  instantaneous field vs average: {100 * np.mean(dev):.1f} % (min {100 * min(dev):.1f}, max {100 * max(dev):.1f}); "
          f"mean speed of the average flow {np.mean(np.linalg.norm(um, axis=1)):.4f} m/s")
    src = seat_source(g, N_REF)
    t_chk = [t for t in (120.0, 300.0, 600.0) if t in td]
    T_all = [0.0] + times
    run1, _, n1 = TurbFV(g, src, args.device, torch.float32).solve(T_all, [snaps[t] for t in T_all], t_chk)
    T2 = [0.0] + [t for t in times if t < args.avg_start] + [args.avg_start]
    S2 = [snaps[t] for t in T2[:-1]] + [mean]          # average flow from avg_start on (frozen afterwards)
    run2, _, n2 = TurbFV(g, src, args.device, torch.float32).solve(T2, S2, t_chk)
    print(f"\n{'t':>5s} | {'run 1 (all snapshots)':>27s} | {'run 2 (average flow)':>27s} | mean excess CO2 OF / 1 / 2 [ppm]")
    print(f"{'':>5s} | {'vol':>7s} {'plane':>7s} {'mass':>9s} | {'vol':>7s} {'plane':>7s} {'mass':>9s} |")
    for t in t_chk:
        ref = to_grid(g, idx, read_internal(os.path.join(case, td[t], "s"), 1, len(Cc)).reshape(-1))
        row = []
        for out in (run1, run2):
            c = out[t]
            row += [np.linalg.norm(c[fl] - ref[fl]) / np.linalg.norm(ref[fl]),
                    np.linalg.norm(c[:, :, kz][pl] - ref[:, :, kz][pl]) / np.linalg.norm(ref[:, :, kz][pl]),
                    c[fl].sum() / ref[fl].sum()]
        print(f"{t:5.0f} | {100 * row[0]:6.1f}% {100 * row[1]:6.1f}% {row[2]:9.3f} | {100 * row[3]:6.1f}% "
              f"{100 * row[4]:6.1f}% {row[5]:9.3f} | {ref[fl].mean():6.1f} / {run1[t][fl].mean():6.1f} / {run2[t][fl].mean():6.1f}")
    print("\nReading: run 2 close to run 1 (and to OpenFOAM) -> the averaged flow is good enough for CO2 (cheap dataset);"
          "\n         run 2 much worse -> the fluctuations matter, CO2 must follow the unsteady flow.")


if __name__ == "__main__":
    main()
