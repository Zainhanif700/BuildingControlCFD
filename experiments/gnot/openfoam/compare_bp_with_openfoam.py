"""
NO-TRAINING test of the through-flow field B_p (throughflow.py) against OpenFOAM.

The model's velocity is curl(B_p + s*phi*A_net). This script evaluates curl(B_p) ALONE (network
correction = 0) for several B_p variants and compares each with the OpenFOAM flow:
  linear       -- v19/v20 construction (psi blended linearly across the room depth)
  jet_<d>      -- v21 candidate: straight jets from the windows, turning layer of depth d at the
                  door wall (d = 1, 2, 3 m)
each with the door split alpha from potential flow (as v20 starts) and with OpenFOAM's measured
split (isolates the path effect from the split effect). If a B_p variant ALONE is already much
closer to OpenFOAM than the trained v19 model (74% volume / 67% plane velocity error for
W1 1 m/s), it is a better starting point for training -- decided before spending GPU hours.
Also run on a second scenario (e.g. all 1 m/s) before choosing, to avoid tuning to one case.

Usage (training env with torch):  python3 compare_bp_with_openfoam.py --case cases/W1_1ms_dx0.1
Writes openfoam/results/<case>__bp_variants/.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--times", type=float, nargs="*", default=[60.0, 120.0])
    ap.add_argument("--depths", type=float, nargs="*", default=[1.0, 2.0, 3.0])
    ap.add_argument("--batch", type=int, default=20000)
    args = ap.parse_args()
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import throughflow as tfl
    from point_sampler import BREATHING_HEIGHT, ROOM_X, ROOM_Y
    import check_co2_with_model_flow as L2
    from compare_with_openfoam import read_internal, read_patch_sum, time_dirs

    case = os.path.abspath(os.path.expanduser(args.case))
    meta = dict(line.split(None, 1) for line in open(os.path.join(case, "scenario.txt")).read().splitlines())
    V = [float(v) for v in meta["V"].split()]
    dx = float(meta["dx"])
    out_dir = os.path.join(HERE, "results", f"{os.path.basename(case)}__bp_variants")
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "bp_variants.log"), "w")

    def say(s=""):
        print(s, flush=True)
        log.write(s + "\n")

    g = L2.Grid(dx, V)
    C = read_internal(os.path.join(case, "0", "C"), 3)
    idx = [np.clip(np.floor((C[:, a] - (ROOM_X[0], ROOM_Y[0], 0.0)[a]) / g.h[a]).astype(int), 0, g.n[a] - 1)
           for a in range(3)]
    tdirs = dict((t, d) for t, d in time_dirs(case))
    kz = int(np.argmin(np.abs(g.c1d[2] - BREATHING_HEIGHT)))
    fl = g.fluid
    Pf = np.stack([g.X[fl], g.Y[fl], g.Z[fl]], 1)

    def bp_velocity(t, alpha, blend, depth):
        out = []
        Vt = torch.tensor([V], dtype=torch.float64)
        for i in range(0, len(Pf), args.batch):
            Q = torch.tensor(Pf[i:i + args.batch], dtype=torch.float64)
            x, y, z = (Q[:, j:j + 1].clone().requires_grad_(True) for j in range(3))
            n = x.shape[0]
            chi, psi = tfl.through_flow_potential(x, y, z, torch.full((n, 1), t, dtype=torch.float64),
                                                  Vt.expand(n, -1), torch.full((n, 1), alpha, dtype=torch.float64),
                                                  blend=blend, turn_depth=depth)
            gp = torch.autograd.grad(psi.sum(), (x, y), retain_graph=True)
            gc = torch.autograd.grad(chi.sum(), (y, z))
            u, v, w = gp[1], gc[1] - gp[0], -gc[0]
            out.append(torch.cat([u, v, w], 1).numpy())
        U = np.full(g.X.shape + (3,), np.nan)
        U[fl] = np.concatenate(out)
        return U

    with torch.no_grad():
        a_pot = tfl.alpha_potential(torch.tensor([V], dtype=torch.float64)).item()
    say(f"case {case}: V = {V}, dx = {dx}; potential-flow alpha = {a_pot:.3f}")
    say("reference: trained v19 (W1 1 m/s, t=60/120 s): velocity error 74% volume / 67% plane\n")
    variants = [("linear", "linear", None)] + [(f"jet_{d:g}m", "jet", d) for d in args.depths]
    say(f"{'t':>5s} {'variant':>9s} {'alpha':>11s} | {'volume':>7s} {'plane':>7s} | {'|u| B_p':>8s} {'|u| OF':>7s}")
    best = None
    maps = {}
    for t in args.times:
        d = tdirs.get(t)
        if d is None:
            say(f"{t:5.0f}  not saved by OpenFOAM -- skipped")
            continue
        U_of = np.full(g.X.shape + (3,), np.nan)
        U_of[idx[0], idx[1], idx[2]] = read_internal(os.path.join(case, d, "U"), 3, len(C))
        d1 = read_patch_sum(os.path.join(case, d, "phi"), "door1")
        d2 = read_patch_sum(os.path.join(case, d, "phi"), "door2")
        a_of = d1 / (d1 + d2)
        for name, blend, depth in variants:
            for a_lab, alpha in (("pot", a_pot), ("OpenFOAM", a_of)):
                U = bp_velocity(t, alpha, blend, depth)
                e_v = np.sqrt(np.nansum((U[fl] - U_of[fl]) ** 2) / np.nansum(U_of[fl] ** 2))
                pl = fl[:, :, kz]
                e_p = np.sqrt(np.nansum((U[:, :, kz][pl] - U_of[:, :, kz][pl]) ** 2) / np.nansum(U_of[:, :, kz][pl] ** 2))
                say(f"{t:5.0f} {name:>9s} {a_lab + f'={alpha:.2f}':>11s} | {100 * e_v:6.1f}% {100 * e_p:6.1f}% | "
                    f"{np.nanmean(np.linalg.norm(U[fl], axis=1)):8.3f} {np.nanmean(np.linalg.norm(U_of[fl], axis=1)):7.3f}")
                if t == max(args.times) and a_lab == "pot":
                    maps[name] = np.linalg.norm(U[:, :, kz], axis=-1)
                    if best is None or e_v < best[1]:
                        best = (name, e_v)
        if t == max(args.times):
            maps["OpenFOAM"] = np.linalg.norm(U_of[:, :, kz], axis=-1)
    if maps:
        names = ["OpenFOAM"] + [n for n, _, _ in variants]
        fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 3.6))
        vmax = np.nanmax(maps["OpenFOAM"])
        ext = [ROOM_X[0], ROOM_X[1], ROOM_Y[0], ROOM_Y[1]]
        for ax, n in zip(axes, names):
            im = ax.imshow(maps[n].T, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax)
            ax.set_title(n if n == "OpenFOAM" else f"B_p only: {n}")
            ax.set_xlabel("x [m]")
        axes[0].set_ylabel("y [m]")
        fig.colorbar(im, ax=axes, fraction=0.015, label="speed [m/s]")
        fig.suptitle(f"{meta['name']}: speed at z = {BREATHING_HEIGHT} m, t = {max(args.times):g} s -- "
                     f"through-flow field alone (no network) vs OpenFOAM")
        fig.savefig(os.path.join(out_dir, "bp_variants_speed.png"), dpi=110, bbox_inches="tight")
    if best:
        say(f"\nclosest B_p variant alone (potential alpha, t = {max(args.times):g} s): {best[0]} "
            f"({100 * best[1]:.1f}% volume error)")
    say(f"results in {out_dir}")


if __name__ == "__main__":
    main()
