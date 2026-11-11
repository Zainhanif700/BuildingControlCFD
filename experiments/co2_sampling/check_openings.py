"""
Checks a case with outlet windows or closed doors: air in through each inlet window and out through each door and
outlet window, in OpenFOAM's flow (cells next to the wall) and after our solver's divergence-free correction.
Only reads the case.
Usage: python3 check_openings.py --case <case> [--time <s>]
"""
import argparse

import numpy as np

import stream_co2 as S
from point_sampler import WINDOWS, DOORS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--time", type=float, default=None, help="default: last save")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import torch
    from fv_cons import ConsFV

    meta, V, g, doors, times, field, td = S.read_case(args.case)
    t = args.time or times[-1]
    u = field(t, "U")
    fv = ConsFV(g, np.zeros(g.X.shape), V, args.device, torch.float64, doors=doors)
    F, rd, it = fv.project(u[..., 0], u[..., 1], u[..., 2], t)
    xc, a = g.c1d[0], g.h[0] * g.h[2]
    top_raw, bot_raw = u[:, -1, :, 1], u[:, 0, :, 1]
    top, bot = F[1][:, -1, :].cpu().numpy(), F[1][:, 0, :].cpu().numpy()
    print(f"{meta.get('name', args.case)}: V = {V}, doors {doors}, t = {t:g} s; projection divergence {rd:.1e} ({it} CG iterations)")
    print("opening      state      OpenFOAM [m3/s]  corrected [m3/s]   (+ = into the room)")
    tot_in = tot_out = 0.0
    for k, (x0, x1, _, _) in enumerate(WINDOWS):
        m = ((xc >= x0) & (xc <= x1))[:, None] & g.fluid[:, -1, :]
        state = "inlet" if V[k] > 0 else ("outlet" if V[k] < 0 else "closed")
        qr, qc = -float(top_raw[m].sum() * a), -float(top[m].sum() * a)
        if V[k] != 0:
            print(f"  window {k + 1}   {state:8s} {qr:12.3f} {qc:15.3f}" + (f"   (set {V[k] * (x1 - x0) * 3.15:.3f})" if V[k] > 0 else ""))
        tot_in += max(qc, 0.0)
        tot_out += max(-qc, 0.0)
    for j, ((x0, x1, _, z1), op) in enumerate(zip(DOORS, doors)):
        e = 0.25 * min(g.h)
        m = ((xc >= x0) & (xc <= x1))[:, None] & (g.c1d[2] <= z1 + e)[None, :] & g.fluid[:, 0, :]
        qr, qc = float(bot_raw[m].sum() * a), float(bot[m].sum() * a)
        print(f"  door {j + 1}     {'open' if op else 'closed':8s} {qr:12.3f} {qc:15.3f}")
        tot_in += max(qc, 0.0)
        tot_out += max(-qc, 0.0)
    print(f"total in {tot_in:.3f} m3/s, total out {tot_out:.3f} m3/s (must be equal after the correction)")


if __name__ == "__main__":
    main()
