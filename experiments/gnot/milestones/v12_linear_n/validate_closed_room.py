# ============================================================================
# MILESTONE SNAPSHOT: v12_linear_n (2026-09-26) -- frozen copy, DO NOT EDIT.
# Code from git commit 0340b21 (exactly what the v12 run trained with).
# See README.md. Run scripts from INSIDE this folder.
# ============================================================================

"""
Closed-window ROBUSTNESS validation of a trained PINN against the
finite-difference reference (fd_reference_closed_room.py), across several
occupancies, heights and times -- plus two figures for presentations.

The closed-room problem (u = 0, pure diffusion + source, c(t=0) = 0) is LINEAR
in the source, and the source is proportional to N_people. So ONE reference
solve at N = 1 gives the exact reference for every N by scaling: c_N = N * c_1.
The network is evaluated separately at each N. Before v12 this tested its
ability to generalize across occupancy (learned only from physics). From v12 on
the model is EXACTLY linear in N by construction, so all relative errors are
identical across N -- compare the common error level against earlier versions'
best case instead.

Cases: N_people in {5, 20, 50} x height z in {0.5, 1.10, 2.0} m x t in
{30, 60, 120} s (27 cases). For each: CO2 at the source column (x, y of the
source, at that height) and the relative L2 error over the full 40x40
horizontal plane at that height.

Outputs (in figures/<checkpoint version>/):
  closed_room_validation.csv          -- the full table
  closed_room_maps_N20_z1.10_t60.png  -- reference | PINN | PINN - reference
  closed_room_timeseries_source.png   -- CO2 at the source vs time, N = 5/20/50

Usage:
    python3 validate_closed_room.py <checkpoint> [--dx 0.1]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from fd_reference_closed_room import solve, interp, breathing_grid, pinn_on, SX, SY
from gnot_model import GNOTOperator, check_checkpoint_compat
from point_sampler import ROOM_X, ROOM_Y, COLUMNS, WINDOWS, DOORS, BREATHING_HEIGHT

N_LIST = [5.0, 20.0, 50.0]
Z_LIST = [0.5, BREATHING_HEIGHT, 2.0]
T_LIST = [30.0, 60.0, 120.0]
HERE = os.path.dirname(os.path.abspath(__file__))


def room_outline(ax):
    """Room walls, windows (y = max wall, blue), doors (y = 0 wall, orange), columns."""
    ax.add_patch(patches.Rectangle((ROOM_X[0], ROOM_Y[0]), ROOM_X[1] - ROOM_X[0], ROOM_Y[1] - ROOM_Y[0],
                                   fill=False, lw=1.2, color="k"))
    for xlo, xhi, _, _ in WINDOWS:
        ax.plot([xlo, xhi], [ROOM_Y[1], ROOM_Y[1]], color="tab:blue", lw=4, solid_capstyle="butt")
    for xlo, xhi, _, _ in DOORS:
        ax.plot([xlo, xhi], [ROOM_Y[0], ROOM_Y[0]], color="tab:orange", lw=4, solid_capstyle="butt")
    for cx, cy, r, _, _ in COLUMNS:
        ax.add_patch(patches.Circle((cx, cy), r, color="0.5"))
    ax.plot(SX, SY, marker="+", color="k", ms=10, mew=1.5)
    ax.set_xlim(ROOM_X[0] - 0.3, ROOM_X[1] + 0.3)
    ax.set_ylim(ROOM_Y[0] - 0.3, ROOM_Y[1] + 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--dx", type=float, default=0.1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    check_checkpoint_compat(ckpt, args.checkpoint)
    model = GNOTOperator().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    version = ckpt.get("version", "unknown")
    fig_dir = os.path.join(HERE, "figures", version)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Reference: one FD solve at N=1 (dx~{args.dx} m, no-flux walls), scaled by N.")
    out1, grid, (nx, ny, nz, dt) = solve(args.dx, "noflux", n_people=1.0)
    xs, ys, xg, yg, inside = breathing_grid()
    m = ~inside

    rows = []
    print(f"\nPINN {args.checkpoint} (version={version}, iter={ckpt.get('iter', '?')})")
    print(f"{'N':>4s} {'z[m]':>5s} {'t[s]':>5s} | {'FD source':>9s} {'PINN source':>11s} {'err':>7s} | {'plane L2':>8s}")
    for N in N_LIST:
        for z in Z_LIST:
            zb = np.full_like(xg, z)
            for t in T_LIST:
                ref = N * interp(out1[t], grid, xg, yg, zb).astype(float)
                p = pinn_on(model, device, xg, yg, t, z=z, n_people=N)
                rel = np.linalg.norm(p[m] - ref[m]) / np.linalg.norm(ref[m])
                ref_s = N * float(interp(out1[t], grid, SX, SY, z))
                p_s = float(pinn_on(model, device, [SX], [SY], t, z=z, n_people=N)[0])
                err_s = (p_s - ref_s) / ref_s
                rows.append((N, z, t, ref_s, p_s, err_s, rel))
                print(f"{N:4.0f} {z:5.2f} {t:5.0f} | {ref_s:9.4f} {p_s:11.4f} {err_s * 100:+6.1f}% | {rel * 100:7.1f}%")

    arr = np.array(rows)
    print("\nSUMMARY (plane relative L2 error):")
    print(f"  all 27 cases: mean {arr[:, 6].mean() * 100:.1f}%, worst {arr[:, 6].max() * 100:.1f}%")
    for N in N_LIST:
        sel = arr[arr[:, 0] == N]
        print(f"  N={N:4.0f}: mean {sel[:, 6].mean() * 100:5.1f}%   source error mean |{np.abs(sel[:, 5]).mean() * 100:4.1f}|%")
    for z in Z_LIST:
        sel = arr[np.isclose(arr[:, 1], z)]
        print(f"  z={z:4.2f} m: mean {sel[:, 6].mean() * 100:5.1f}%   source error mean |{np.abs(sel[:, 5]).mean() * 100:4.1f}|%")

    csv = os.path.join(fig_dir, "closed_room_validation.csv")
    np.savetxt(csv, arr, delimiter=",", fmt="%.6g",
               header="N_people,z_m,t_s,fd_source,pinn_source,source_rel_err,plane_rel_L2", comments="")
    print(f"\nTable written to {csv}")

    # ---- figure 1: reference | PINN | difference, N=20, breathing height, t=60 s
    N, z, t = 20.0, BREATHING_HEIGHT, 60.0
    ref = (N * interp(out1[t], grid, xg, yg, np.full_like(xg, z))).astype(float)
    p = pinn_on(model, device, xg, yg, t, z=z, n_people=N)
    ref[inside] = np.nan
    p = p.astype(float)
    p[inside] = np.nan
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    vmax = np.nanmax(ref)
    dmax = np.nanmax(np.abs(p - ref))
    rel = np.sqrt(np.nansum((p - ref) ** 2) / np.nansum(ref ** 2))
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    panels = [(ref, "Reference (finite difference)", "viridis", 0, vmax),
              (p, f"physics-informed GNOT ({version})", "viridis", 0, vmax),
              (p - ref, f"GNOT - reference (rel. L2 {rel * 100:.1f}%)", "RdBu_r", -dmax, dmax)]
    for ax, (F, title, cmap, lo, hi) in zip(axes, panels):
        im = ax.imshow(F.reshape(40, 40).T, origin="lower", extent=extent, cmap=cmap, vmin=lo, vmax=hi,
                       interpolation="bilinear")
        room_outline(ax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="excess CO2 [model units]")
    fig.suptitle(f"Closed windows, {N:.0f} people, breathing height z = {z:.2f} m, t = {t:.0f} s "
                 f"(blue = windows, orange = doors, grey = columns, + = CO2 source)")
    fig.tight_layout()
    f1 = os.path.join(fig_dir, "closed_room_maps_N20_z1.10_t60.png")
    fig.savefig(f1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- figure 2: CO2 at the source vs time for N = 5, 20, 50
    t_fine = np.arange(0.0, 120.0 + 1e-9, 2.0)
    t_ref = sorted(out1.keys())
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for N, col in zip(N_LIST, ["tab:green", "tab:blue", "tab:red"]):
        pinn_t = [float(pinn_on(model, device, [SX], [SY], tt, z=BREATHING_HEIGHT, n_people=N)[0]) for tt in t_fine]
        ref_t = [N * float(interp(out1[tt], grid, SX, SY, BREATHING_HEIGHT)) for tt in t_ref]
        ax.plot(t_fine, pinn_t, color=col, lw=2, label=f"GNOT, {N:.0f} people")
        ax.plot(t_ref, ref_t, "o", color=col, mfc="white", mew=1.5, label=f"reference, {N:.0f} people")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("excess CO2 at the source [model units]")
    ax.set_title("Closed windows: CO2 build-up at the source (breathing height)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    f2 = os.path.join(fig_dir, "closed_room_timeseries_source.png")
    fig.savefig(f2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figures written to\n  {f1}\n  {f2}")


if __name__ == "__main__":
    main()
