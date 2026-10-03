"""
Can extra turbulent mixing make up for the flow fluctuations that the averaged flow (B2) leaves out?
B2 with the new solver (fv_cons) for several turbulent Schmidt numbers Sc_t (smaller = more mixing,
D = D_CO2 + nut / Sc_t), against OpenFOAM's own CO2 on a kept case (0-600 s).
Sc_t = 0.7 is the standard value used so far. A value is only adopted if it also wins on a SECOND case.
Usage (training env, from experiments/gnot_openfoam_rans):
  python3 scan_sct.py --case cases/V04_dx0.1          (~8 min per value)
"""
import argparse
import os

import numpy as np

import common
from common import N_REF

T_AVG = 150.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="cases/V04_dx0.1")
    ap.add_argument("--sct", default="0.7,0.5,0.35,0.25,0.15")
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
    rd = lambda t, f, n: np.nan_to_num(to_grid(g, idx, read_internal(os.path.join(case, td[t], f), n, len(Cc)) if n == 3
                                               else read_internal(os.path.join(case, td[t], f), 1, len(Cc)).reshape(-1)))
    times = sorted(t for t in td if t > 0)
    z = np.zeros(g.X.shape)
    snaps = {0.0: [z, z, z, z]}
    for t in times:
        u = rd(t, "U", 3) * fl[..., None]
        snaps[t] = [u[..., 0], u[..., 1], u[..., 2], rd(t, "nut", 1) * fl]
    mean = [np.mean([snaps[t][k] for t in times if t >= T_AVG], axis=0) for k in range(4)]
    T = [0.0] + [t for t in times if t <= T_AVG]
    S = [snaps[t] for t in T] + [mean]
    T = T + [T[-1] + 10.0]
    t_chk = [t for t in (300.0, 600.0) if t in td]
    refs = {t: rd(t, "s", 1) * fl for t in t_chk}
    src = seat_source(g, N_REF)
    k11 = int(np.argmin(np.abs(g.c1d[2] - 1.1)))
    k16 = int(np.argmin(np.abs(g.c1d[2] - 1.6)))
    print(f"case {case}: B2 + new solver vs OpenFOAM CO2\n{'Sc_t':>5s} {'t':>5s} | {'vol':>6s} {'mass':>6s} | paper 1.1 / 1.6 m", flush=True)
    for sct in [float(s) for s in args.sct.split(",")]:
        out, _, _ = ConsFV(g, src, V, args.device, torch.float32, sc_t=sct).solve(T, S, t_chk)
        for t in t_chk:
            ref, c = refs[t], out[t]
            ev = np.linalg.norm(c[fl] - ref[fl]) / np.linalg.norm(ref[fl])
            pap = []
            for k in (k11, k16):
                m = fl[:, :, k]
                r = ref[:, :, k][m] + common.PPM_OUTDOOR
                pap.append(100 * np.linalg.norm(c[:, :, k][m] + common.PPM_OUTDOOR - r) / np.linalg.norm(r))
            print(f"{sct:5.2f} {t:5.0f} | {100 * ev:5.1f}% {c[fl].sum() / ref[fl].sum():6.3f} | {pap[0]:5.2f}% / {pap[1]:5.2f}%", flush=True)


if __name__ == "__main__":
    main()
