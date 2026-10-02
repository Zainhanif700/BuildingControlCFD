"""
Minutes 10-30 check on the pilot extended to 1800 s (W1 at 1 m/s). Works on a partly finished run too
(uses the saves that exist).

 1. flow: time average over [150, 600] (what the dataset uses) vs over [600, end] -> is the mean flow
    the same in minutes 10-30?
 2. CO2 against OpenFOAM's own CO2 (field s, real fluctuating flow) at 5, 10, 15, 20, 25, 30 min:
      real  fv_turb on all saved snapshots                 (solver error on the true flow)
      B2    snapshots to 150 s, then the [150, 600] average frozen  (= the dataset)
      B2L   snapshots to 150 s, then the [150, end] average frozen  (longer average)
    volume and plane errors and mass ratio; does the B2 error stay ~constant or grow after 10 min?
Usage (training env, from experiments/gnot_openfoam_rans):
  python3 check_pilot30.py --case cases/W1_1ms_rans_dx0.1
"""
import argparse
import os

import numpy as np

import common  # noqa: F401
from common import N_REF, BREATHING_Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="cases/W1_1ms_rans_dx0.1")
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
    has = lambda t, f: os.path.isfile(os.path.join(case, td[t], f)) or os.path.isfile(os.path.join(case, td[t], f + ".gz"))
    times = sorted(t for t in td if t > 0 and has(t, "U") and has(t, "nut") and has(t, "s"))
    t_end = times[-1]
    if t_end < 700:
        raise SystemExit(f"saved only to {t_end:g} s -- run it when the pilot is past ~700 s")
    rd = lambda t, f, n: np.nan_to_num(to_grid(g, idx, read_internal(os.path.join(case, td[t], f), n, len(Cc)) if n == 3
                                               else read_internal(os.path.join(case, td[t], f), 1, len(Cc)).reshape(-1)))
    z = np.zeros(g.X.shape, np.float32)
    snaps = {0.0: [z, z, z, z]}
    for t in times:
        u = (rd(t, "U", 3) * fl[..., None]).astype(np.float32)
        snaps[t] = [u[..., 0], u[..., 1], u[..., 2], (rd(t, "nut", 1) * fl).astype(np.float32)]
    avg = lambda a, b: [np.mean([snaps[t][k] for t in times if a <= t <= b], axis=0) for k in range(4)]
    m1, m2, mL = avg(150, 600), avg(600.0001, t_end), avg(150, t_end)
    U = lambda m: np.stack(m[:3], -1)[fl]
    d12 = np.linalg.norm(U(m1) - U(m2)) / np.linalg.norm(U(m1))
    sp = lambda m: np.mean(np.linalg.norm(U(m), axis=1))
    print(f"case {case}: {len(times)} saves to {t_end:g} s")
    print(f"1. mean flow [150,600] vs [600,{t_end:g}]: velocity difference {100 * d12:.1f} %; mean speed "
          f"{sp(m1):.4f} / {sp(m2):.4f} m/s; mean nut {m1[3][fl].mean():.2e} / {m2[3][fl].mean():.2e}")

    src = seat_source(g, N_REF)
    t_chk = [t for t in (300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0) if t <= t_end and t in td and has(t, "s")]
    T_all = [0.0] + times
    real, _, _ = TurbFV(g, src, args.device, torch.float32).solve(T_all, [snaps[t] for t in T_all], t_chk)
    T0 = [0.0] + [t for t in times if t <= 150.0]
    runs = {"real": real}
    for name, m in (("B2", m1), ("B2L", mL)):
        runs[name], _, _ = TurbFV(g, src, args.device, torch.float32).solve(T0 + [T0[-1] + 10.0], [snaps[t] for t in T0] + [m], t_chk)

    print(f"\n2. CO2 vs OpenFOAM (vol % / plane % / mass ratio)")
    print(f"{'t':>5s} | {'real flow':>22s} | {'B2 (dataset)':>22s} | {'B2L (long avg)':>22s} | mean excess OF [ppm]")
    for t in t_chk:
        ref = rd(t, "s", 1) * fl
        cells = []
        for name in ("real", "B2", "B2L"):
            c = runs[name][t]
            ev = np.linalg.norm(c[fl] - ref[fl]) / np.linalg.norm(ref[fl])
            ep = np.linalg.norm(c[:, :, kz][pl] - ref[:, :, kz][pl]) / np.linalg.norm(ref[:, :, kz][pl])
            cells.append(f"{100 * ev:5.1f} {100 * ep:5.1f} {c[fl].sum() / ref[fl].sum():6.3f}")
        print(f"{t:5.0f} | " + " | ".join(f"{s:>22s}" for s in cells) + f" | {ref[fl].mean():7.1f}")
    print("\nReading: 1. small difference -> the mean flow is the same in minutes 10-30."
          "\n         2. B2 error roughly constant after 10 min -> the 30-min data are as good as at 10 min;"
          "\n            growing -> B2 drifts away from the real run in minutes 10-30.")


if __name__ == "__main__":
    main()
