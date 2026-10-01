"""
RANS pilot evaluation (window 1 at 1 m/s), two questions:

 A. How different is the k-omega SST flow from the laminar effective-viscosity flow used so far?
    relative L2 difference of the velocity (volume / breathing plane), mean speeds, door split,
    steadiness (U 150 s vs 180 s), size of the turbulent viscosity nut, yPlus (from the log).
 B. Is our CO2 solver with turbulent mixing (fv_turb.py) right? It is run on the OpenFOAM U and nut
    and compared with OpenFOAM's own CO2 transport of the same problem (scalarTransport function
    object: same diffusivity D_CO2 + nut/Sc_t, same seating source, same boundary conditions).
    The two use different discretisations (cell-centred 2nd-order upwind vs OpenFOAM linearUpwind,
    our flux form vs OpenFOAM's), so agreement within a few % is the target, not round-off.

Usage (training env, from experiments/gnot_openfoam_rans):
  python3 compare_rans_pilot.py --rans cases/W1_1ms_rans_dx0.1 --laminar ../gnot/openfoam/cases/W1_1ms_dx0.1
"""
import argparse
import os
import re

import numpy as np

import common
from common import NU_AIR, N_REF, BREATHING_Z


def load_case(case, read_internal, time_dirs, L2):
    from point_sampler import ROOM_X, ROOM_Y
    meta = dict(l.split(None, 1) for l in open(os.path.join(case, "scenario.txt")).read().splitlines() if l.strip())
    V = [float(v) for v in meta["V"].split()]
    dx = float(meta["dx"])
    g = L2.Grid(dx, V)
    Cc = read_internal(os.path.join(case, "0", "C"), 3)
    idx = [np.clip(np.floor((Cc[:, a] - (ROOM_X[0], ROOM_Y[0], 0.0)[a]) / g.h[a]).astype(int), 0, g.n[a] - 1)
           for a in range(3)]
    return meta, V, g, Cc, idx, dict(time_dirs(case))


def to_grid(g, idx, vals):
    out = np.full(g.X.shape + vals.shape[1:], np.nan)
    out[idx[0], idx[1], idx[2]] = vals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rans", required=True)
    ap.add_argument("--laminar", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import check_co2_with_model_flow as L2
    from compare_with_openfoam import read_internal, time_dirs, read_patch_sum
    rans = os.path.abspath(args.rans)
    lam = os.path.abspath(args.laminar)
    mr, Vr, g, Cr, ir, tr = load_case(rans, read_internal, time_dirs, L2)
    ml, Vl, gl, Cl, il, tl = load_case(lam, read_internal, time_dirs, L2)
    assert Vr == Vl and g.n == gl.n, "the two cases must have the same windows and grid"
    if max(tr) < 60.0:
        raise SystemExit(f"the RANS case has results only up to t = {max(tr):g} s -- the OpenFOAM run did not "
                         f"finish. See: tail -40 {rans}/log.pimpleFoam (and the newest log.* file there)")
    fl = g.fluid
    kz = int(np.argmin(np.abs(g.c1d[2] - BREATHING_Z)))
    pl = fl[:, :, kz]
    U = lambda case, td, ii, CC, t: to_grid(g, ii, read_internal(os.path.join(case, td[t], "U"), 3, len(CC)))
    S = lambda case, td, ii, CC, t, f: to_grid(g, ii, read_internal(os.path.join(case, td[t], f), 1, len(CC)).reshape(-1))

    print(f"RANS case {rans} (V = {Vr}); laminar case {lam}")
    print("\nA. FLOW: k-omega SST (nu = 1.5e-5) vs laminar (nu = 0.01)")
    print(f"{'t':>5s} | {'diff vol':>8s} {'diff plane':>10s} | {'|u| RANS':>8s} {'|u| lam':>8s} | split RANS / lam | nut mean / max [m2/s], nut/nu")
    for t in (60.0, 120.0):
        if t not in tr or t not in tl:
            print(f"{t:5.0f} | missing"); continue
        a, b = U(rans, tr, ir, Cr, t), U(lam, tl, il, Cl, t)
        dv = np.linalg.norm(a[fl] - b[fl]) / np.linalg.norm(b[fl])
        dp = np.linalg.norm(a[:, :, kz][pl] - b[:, :, kz][pl]) / np.linalg.norm(b[:, :, kz][pl])
        sp = []
        for case, td in ((rans, tr), (lam, tl)):
            d1 = read_patch_sum(os.path.join(case, td[t], "phi"), "door1")
            d2 = read_patch_sum(os.path.join(case, td[t], "phi"), "door2")
            sp.append(d1 / (d1 + d2))
        nut = S(rans, tr, ir, Cr, t, "nut")[fl]
        print(f"{t:5.0f} | {100 * dv:7.1f}% {100 * dp:9.1f}% | {np.mean(np.linalg.norm(a[fl], axis=1)):8.3f} "
              f"{np.mean(np.linalg.norm(b[fl], axis=1)):8.3f} | {sp[0]:.3f} / {sp[1]:.3f}      | "
              f"{nut.mean():.2e} / {nut.max():.2e}, {nut.mean() / NU_AIR:.0f}x")
    t_last = max(tr)
    t_prev = min(tr, key=lambda s: abs(s - (t_last - 30.0)))
    a, b = U(rans, tr, ir, Cr, t_last)[fl], U(rans, tr, ir, Cr, t_prev)[fl]
    print(f"steadiness RANS |U({t_last:g}) - U({t_prev:g})| / |U({t_last:g})| = {100 * np.linalg.norm(a - b) / np.linalg.norm(a):.2f}%")
    logp = os.path.join(rans, "log.pimpleFoam")
    if os.path.isfile(logp):
        txt = open(logp).read()
        yp = re.findall(r"patch (\w+) y\+ : min = ([0-9.eE+-]+), max = ([0-9.eE+-]+), average = ([0-9.eE+-]+)", txt)
        if yp:
            last = {}
            for p, mn, mx, av in yp:
                last[p] = (float(mn), float(mx), float(av))
            print("yPlus at the last write (patch: min / max / average):")
            for p, (mn, mx, av) in last.items():
                print(f"    {p:10s} {mn:8.2f} / {mx:8.2f} / {av:8.2f}")

    print("\nB. CO2: fv_turb.py vs OpenFOAM scalarTransport (same D_CO2 + nut/Sc_t, same seating source)")
    from fv_turb import TurbFV, seat_source
    times = sorted(tr)
    snaps = []
    for t in times:
        if t == 0:
            z = np.zeros(g.X.shape)
            snaps.append([z, z, z, z])
            continue
        u = np.nan_to_num(U(rans, tr, ir, Cr, t)) * fl[..., None]
        nut = np.nan_to_num(S(rans, tr, ir, Cr, t, "nut")) * fl
        snaps.append([u[..., 0], u[..., 1], u[..., 2], nut])
    if times[0] > 0:
        z = np.zeros(g.X.shape); snaps.insert(0, [z, z, z, z]); times.insert(0, 0.0)
    src = seat_source(g, N_REF)
    t_chk = [t for t in (60.0, 120.0, 180.0) if t in tr]
    import torch
    got, dt, n = TurbFV(g, src, args.device, torch.float64).solve(times, snaps, t_chk)
    print(f"  fv_turb: {n} steps, dt = {dt:.4f} s; source {src.max():.4g} ppm/s in {int((src > 0).sum())} cells")
    print(f"{'t':>5s} | {'diff vol':>8s} {'diff plane':>10s} | {'mass fv / OF':>12s} | mean excess CO2 fv / OF [ppm]")
    for t in t_chk:
        ref = S(rans, tr, ir, Cr, t, "s")
        c = got[t]
        dv = np.linalg.norm(c[fl] - ref[fl]) / np.linalg.norm(ref[fl])
        dp = np.linalg.norm(c[:, :, kz][pl] - ref[:, :, kz][pl]) / np.linalg.norm(ref[:, :, kz][pl])
        print(f"{t:5.0f} | {100 * dv:7.1f}% {100 * dp:9.1f}% | {c[fl].sum() / ref[fl].sum():12.3f} | "
              f"{c[fl].mean():8.1f} / {ref[fl].mean():8.1f}")
    print("\nReading: B within a few % = fv_turb.py is right (different discretisations explain small gaps);"
          "\n         A shows how much the turbulence model changes the flow compared with the laminar model.")


if __name__ == "__main__":
    main()
