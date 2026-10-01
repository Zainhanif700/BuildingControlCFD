"""
Option a': run OpenFOAM only until the flow is statistically steady, then continue the CO2 with our
solver on a RECORDED stretch of the fluctuating flow, replayed in a loop. Tested on the pilot without a
new OpenFOAM run:

  reference   OpenFOAM's own CO2 (s) on the real flow, 0-600 s
  run real    fv_turb on all real snapshots 0-600 s                  (solver + 10-s snapshot error)
  run replay  fv_turb on the real snapshots up to --real-end, then the stretch
              [--loop-start, --real-end) replayed in a loop until 600 s

If 'replay' is about as close to OpenFOAM as 'real', the replay is justified for the 30-min dataset
(OpenFOAM to ~400-600 s per case instead of 1800 s).
Usage (training env, from experiments/gnot_openfoam_rans):
  python3 check_replay_co2.py --case cases/W1_1ms_rans_dx0.1 [--loop-start 150 --real-end 380]
"""
import argparse
import os

import numpy as np

import common
from common import N_REF, BREATHING_Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--loop-start", type=float, default=150.0)
    ap.add_argument("--real-end", type=float, default=380.0)
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
    t_last = times[-1]
    assert args.loop_start in td and args.real_end in td and args.real_end < t_last, "loop bounds must be saved times"
    z = np.zeros(g.X.shape)
    snaps = {0.0: [z, z, z, z]}
    for t in times:
        u = np.nan_to_num(to_grid(g, idx, read_internal(os.path.join(case, td[t], "U"), 3, len(Cc)))) * fl[..., None]
        nut = np.nan_to_num(to_grid(g, idx, read_internal(os.path.join(case, td[t], "nut"), 1, len(Cc)).reshape(-1))) * fl
        snaps[t] = [u[..., 0], u[..., 1], u[..., 2], nut]
    loop = [t for t in times if args.loop_start <= t < args.real_end]      # recorded stretch
    period = args.real_end - args.loop_start
    T_real = [0.0] + times
    T_rep = [0.0] + [t for t in times if t <= args.real_end]
    S_rep = [snaps[t] for t in T_rep]
    t = args.real_end
    while t < t_last - 1e-9:
        t += 10.0
        src_t = args.loop_start + ((t - args.real_end - 10.0) % period)    # 390 -> loop_start, ...
        T_rep.append(t)
        S_rep.append(snaps[min(loop, key=lambda s: abs(s - src_t))])
    src = seat_source(g, N_REF)
    t_chk = [t for t in (args.real_end, 450.0, 500.0, 600.0) if t in td and t <= t_last]
    real, _, _ = TurbFV(g, src, args.device, torch.float32).solve(T_real, [snaps[t] for t in T_real], t_chk)
    rep, _, _ = TurbFV(g, src, args.device, torch.float32).solve(T_rep, S_rep, t_chk)
    print(f"case {case}: real flow to {args.real_end:g} s, then the stretch {args.loop_start:g}-{args.real_end:g} s "
          f"({len(loop)} snapshots, {period:g} s) replayed to {t_last:g} s")
    print(f"\n{'t':>5s} | {'real flow vs OpenFOAM':>24s} | {'replayed flow vs OpenFOAM':>26s} | mean excess CO2 OF / real / replay [ppm]")
    print(f"{'':>5s} | {'vol':>7s} {'plane':>7s} {'mass':>8s} | {'vol':>7s} {'plane':>7s} {'mass':>8s}   |")
    for t in t_chk:
        ref = to_grid(g, idx, read_internal(os.path.join(case, td[t], "s"), 1, len(Cc)).reshape(-1))
        row = []
        for out in (real, rep):
            c = out[t]
            row += [np.linalg.norm(c[fl] - ref[fl]) / np.linalg.norm(ref[fl]),
                    np.linalg.norm(c[:, :, kz][pl] - ref[:, :, kz][pl]) / np.linalg.norm(ref[:, :, kz][pl]),
                    c[fl].sum() / ref[fl].sum()]
        print(f"{t:5.0f} | {100 * row[0]:6.1f}% {100 * row[1]:6.1f}% {row[2]:8.3f} | {100 * row[3]:6.1f}% "
              f"{100 * row[4]:6.1f}% {row[5]:8.3f}   | {ref[fl].mean():6.1f} / {real[t][fl].mean():6.1f} / {rep[t][fl].mean():6.1f}")
    print("\nReading: 'replayed' about as close to OpenFOAM as 'real' -> option a' is justified (cheap dataset);"
          "\n         clearly worse -> CO2 must be computed with the full OpenFOAM flow (option a).")


if __name__ == "__main__":
    main()
