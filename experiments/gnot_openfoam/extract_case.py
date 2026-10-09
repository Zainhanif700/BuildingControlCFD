"""
Turns one laminar OpenFOAM case into a training file (velocity snapshots and CO2 on the shared grid).
"""
import argparse
import os
import time

import numpy as np

import common
from common import DATA_DIR, N_REF


def co2_times(t_co2):
    base = [float(t) for t in range(0, 121, 10)]
    return base + [float(t) for t in range(150, int(t_co2) + 1, 30)] if t_co2 > 120 else \
        [t for t in base if t <= t_co2]


def extract(case=None, out_dir=DATA_DIR, t_co2=120.0, fv="numpy", name=None, closed_dx=0.1, verify=False,
            device="cuda", fv_dtype="float32"):
    import check_co2_with_model_flow as L2
    from compare_with_openfoam import read_internal, time_dirs
    from point_sampler import ROOM_X, ROOM_Y, BREATHING_HEIGHT, NUM_WINDOWS
    assert abs(L2.N_PEOPLE - N_REF) < 1e-12, "FV solver occupancy must equal common.N_REF"

    if case is None:
        V, dx, nu = [0.0] * NUM_WINDOWS, closed_dx, 0.01
        g = L2.Grid(dx, V)
        zero = [np.zeros(g.X.shape)] * 3
        snaps, stimes = [zero, zero], [0.0, 10.0]
        steady = 0.0
        src = "closed room (no flow)"
    else:
        case = os.path.abspath(os.path.expanduser(case))
        meta = dict(line.split(None, 1) for line in open(os.path.join(case, "scenario.txt")).read().splitlines())
        V = [float(v) for v in meta["V"].split()]
        dx = float(meta["dx"])
        nu = float(meta.get("nu", "0.01"))
        g = L2.Grid(dx, V)
        Cc = read_internal(os.path.join(case, "0", "C"), 3)
        idx = [np.clip(np.floor((Cc[:, a] - (ROOM_X[0], ROOM_Y[0], 0.0)[a]) / g.h[a]).astype(int), 0, g.n[a] - 1)
               for a in range(3)]
        occupied = np.zeros(g.X.shape, bool)
        occupied[idx[0], idx[1], idx[2]] = True
        mism = int(np.sum(occupied != g.fluid))
        assert mism < 0.01 * g.fluid.sum(), f"cell layouts do not match ({mism} cells) -- case made with another dx?"
        snaps, stimes = [], []
        for t, d in time_dirs(case):
            Ug = np.full(g.X.shape + (3,), np.nan)
            Ug[idx[0], idx[1], idx[2]] = read_internal(os.path.join(case, d, "U"), 3, len(Cc))
            Ug = np.nan_to_num(Ug) * g.fluid[..., None]
            snaps.append([Ug[..., a] for a in range(3)])
            stimes.append(t)
        if stimes[0] > 0:
            snaps.insert(0, [np.zeros(g.X.shape)] * 3)
            stimes.insert(0, 0.0)
        t_last = stimes[-1]
        j = int(np.argmin(np.abs(np.array(stimes) - (t_last - 30.0))))
        a, b = np.stack(snaps[-1], -1)[g.fluid], np.stack(snaps[j], -1)[g.fluid]
        steady = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))
        src = f"OpenFOAM {case} (saved to t = {t_last:g} s)"
    name = name or os.path.basename(case.rstrip("/"))
    print(f"{name}: {src}; V = {V}; dx = {dx}; nu = {nu:g}")
    print(f"  steadiness |U(t_last) - U(t_last - 30 s)| / |U(t_last)| = {100 * steady:.2f}%"
          + ("   <-- WARNING: flow not steady, frozen-flow CO2 beyond the OpenFOAM horizon is questionable"
             if steady > 0.05 else ""))

    if verify:
        import fv_torch
        dev = device
        t_chk = (60.0, stimes[-1] + 20.0)
        r = fv_torch.verify(g, stimes, snaps, t_check=t_chk, device=dev)
        print("  torch FV vs numpy FV (max relative difference; seconds per step):")
        for k, v in r.items():
            print(f"    {k:24s} {v:.3e}")
        return None

    t_c = co2_times(t_co2)
    t0 = time.time()
    if fv == "torch":
        import fv_torch
        import torch
        ref, dt_used, nsteps = fv_torch.TorchFV(g, device, getattr(torch, fv_dtype)).solve(
            stimes, snaps, [t for t in t_c if t > 0], log_every=20000)
    else:
        L2.T_SNAP = stimes
        L2.T_OUT = tuple(t for t in t_c if t > 0)
        ref, dt_used, nsteps = L2.solve_co2(g, snaps)
    print(f"  FV CO2 ({fv}{', ' + fv_dtype if fv == 'torch' else ''}): {nsteps} steps, dt = {dt_used:.4f} s, {time.time() - t0:.0f} s wall, to t = {t_c[-1]:g} s")
    nf = int(g.fluid.sum())
    C = np.stack([np.zeros(nf, np.float32) if t == 0 else ref[t][g.fluid].astype(np.float32) for t in t_c])
    t_u = [t for t in stimes if abs(t / 10.0 - round(t / 10.0)) < 1e-6]
    U = np.stack([np.stack(snaps[stimes.index(t)], -1)[g.fluid].astype(np.float32) for t in t_u])
    P = np.stack([g.X[g.fluid], g.Y[g.fluid], g.Z[g.fluid]], 1).astype(np.float32)
    kz = int(np.argmin(np.abs(g.c1d[2] - BREATHING_HEIGHT)))
    plane = np.abs(P[:, 2] - g.c1d[2][kz]) < 1e-5
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, name + ".npz")
    np.savez_compressed(out, P=P, t_u=np.array(t_u, np.float32), U=U, t_c=np.array(t_c, np.float32), C=C,
                        V=np.array(V, np.float32), N_ref=np.float32(N_REF), nu=np.float32(nu), dx=np.float32(dx),
                        plane=plane, z_plane=np.float32(g.c1d[2][kz]), steadiness=np.float32(steady))
    print(f"  saved {out}: {P.shape[0]} points, U at {len(t_u)} times (to {t_u[-1]:g} s, frozen after), "
          f"C at {len(t_c)} times (to {t_c[-1]:g} s); plane z = {g.c1d[2][kz]:.3f} m; {os.path.getsize(out) / 1e6:.1f} MB")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    ap.add_argument("--closed", action="store_true", help="all windows closed (no OpenFOAM case needed)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=DATA_DIR)
    ap.add_argument("--t-co2", type=float, default=120.0, help="CO2 transport horizon [s] (Phase 2: 1800)")
    ap.add_argument("--fv", choices=["numpy", "torch"], default="numpy")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fv-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--verify-fv", action="store_true", help="compare torch FV with numpy FV (to 60 s), no output")
    args = ap.parse_args()
    if args.closed == (args.case is not None):
        raise SystemExit("give exactly one of --case or --closed")
    if args.closed and not args.name:
        raise SystemExit("--closed needs --name")
    extract(None if args.closed else args.case, args.out, args.t_co2, args.fv, args.name,
            verify=args.verify_fv, device=args.device, fv_dtype=args.fv_dtype)


if __name__ == "__main__":
    main()
