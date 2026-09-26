"""
DIAGNOSTIC D1: is the remaining closed-room error driven by the PDE residual,
and WHERE does that residual sit relative to where the training loss looks?

Physics behind it: with all windows closed the flow is exactly zero (hard
constraint), so the CO2 equation is linear: c_t - D lap(c) = S. If the model
satisfies c_t - D lap(c) - S = r (its residual), the error e = c_model - c_true
obeys e_t - D lap(e) = r with e(0) = 0. The diffusion length over 60 s is only
sqrt(2*D*60) ~ 0.8 m (source width sigma = 2.5 m), so locally
    e(x, 60 s)  ~  integral_0^60 r(x, t) dt.
This script evaluates the model's residual on the 40x40 breathing-height grid
at 13 times, integrates it, and compares the PREDICTED error map with the
ACTUAL error against the finite-difference reference.

It also splits both the error and the training loss into a NEAR-source region
(S >= 20% of its peak, i.e. within ~3.2 m of the source) and the FAR field.
The training loss "sees" each point in proportion to how often it is sampled
(40% uniform + 60% Gaussian around the source, std sigma/2). If the far field
holds most of the error but only a small share of the loss, the remaining error
is a weighting/sampling mismatch that no optimizer will fix.

Usage:
    python3 diagnose_residual_map.py <checkpoint> [--dx 0.1]
Writes figures/<version>/residual_diagnosis_N20_z1.10_t60.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from fd_reference_closed_room import solve, interp, breathing_grid, pinn_on, SX, SY
from gnot_model import GNOTOperator, check_checkpoint_compat
from point_sampler import (NUM_WINDOWS, BREATHING_HEIGHT, ROOM_X, ROOM_Y, S_REF,
                           SOURCE_SAMPLE_FRAC, SOURCE_SAMPLE_XY_STD)
from train_gnot import grad, get_velocity_and_derivs, DIFFUSIVITY, EMISSION_PER_PERSON, SIGMA
from validate_closed_room import room_outline

HERE = os.path.dirname(os.path.abspath(__file__))
N_PEOPLE = 20.0
T_END = 60.0
TIMES = np.linspace(0.0, T_END, 13)   # 5 s steps for the time integral
NEAR_FRAC = 0.2                       # "near" = S >= 20% of its peak


def residual_on(model, device, xg, yg, z, t, n_people, batch=400):
    """CO2 residual r = dc/dt + u.grad(c) - D lap(c) - S (physical units, per s),
    closed windows, evaluated in batches (second-order autograd is memory-heavy)."""
    out = []
    for i in range(0, len(xg), batch):
        n = len(xg[i:i + batch])
        x = torch.tensor(xg[i:i + batch], dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
        y = torch.tensor(yg[i:i + batch], dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
        zz = torch.full((n, 1), float(z), device=device).requires_grad_(True)
        tt = torch.full((n, 1), float(t), device=device).requires_grad_(True)
        V = torch.zeros(n, NUM_WINDOWS, device=device)
        Np = torch.full((n, 1), float(n_people), device=device)
        u, v, w, c, _ = get_velocity_and_derivs(model, x, y, zz, tt, V, Np)
        dc_dx, dc_dy, dc_dz, dc_dt = grad(c, x), grad(c, y), grad(c, zz), grad(c, tt)
        lap = grad(dc_dx, x) + grad(dc_dy, y) + grad(dc_dz, zz)
        dist2 = (x - SX) ** 2 + (y - SY) ** 2 + (zz - BREATHING_HEIGHT) ** 2
        S = Np * EMISSION_PER_PERSON * torch.exp(-dist2 / SIGMA ** 2)
        r = dc_dt + u * dc_dx + v * dc_dy + w * dc_dz - DIFFUSIVITY * lap - S
        out.append(r.detach().cpu().numpy().ravel())
        del x, y, zz, tt, u, v, w, c, dc_dx, dc_dy, dc_dz, dc_dt, lap, r
    return np.concatenate(out)


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

    xs, ys, xg, yg, inside = breathing_grid()
    m = ~inside
    z = BREATHING_HEIGHT

    # residual at each time; integrate over time (trapezoid)
    R = np.stack([residual_on(model, device, xg, yg, z, t, N_PEOPLE) for t in TIMES])  # (T, P)
    dt = np.diff(TIMES)[:, None]                     # explicit trapezoid rule (np.trapz is
    e_pred = np.sum(0.5 * (R[1:] + R[:-1]) * dt, 0)  # deprecated in newer NumPy)

    out1, grid, _ = solve(args.dx, "noflux", n_people=1.0)
    ref = N_PEOPLE * interp(out1[T_END], grid, xg, yg, np.full_like(xg, z)).astype(float)
    pinn = pinn_on(model, device, xg, yg, T_END, z=z, n_people=N_PEOPLE).astype(float)
    e_act = pinn - ref

    # region split
    S0 = N_PEOPLE * EMISSION_PER_PERSON
    S_plane = S0 * np.exp(-((xg - SX) ** 2 + (yg - SY) ** 2) / SIGMA ** 2)   # z = source height
    near = (S_plane >= NEAR_FRAC * S0) & m
    far = (~near) & m
    # training sampling density on this plane (xy only): 40% uniform + 60% Gaussian
    area = (ROOM_X[1] - ROOM_X[0]) * (ROOM_Y[1] - ROOM_Y[0])
    s2 = SOURCE_SAMPLE_XY_STD ** 2
    dens = (1 - SOURCE_SAMPLE_FRAC) / area + SOURCE_SAMPLE_FRAC * np.exp(
        -((xg - SX) ** 2 + (yg - SY) ** 2) / (2 * s2)) / (2 * np.pi * s2)
    r_loss = (R / S_REF) ** 2 * dens          # what the training loss 'sees', per point and time
    loss_near = r_loss[:, near].sum()
    loss_far = r_loss[:, far].sum()
    err2_near = np.sum(e_act[near] ** 2)
    err2_far = np.sum(e_act[far] ** 2)

    def rel(a, b):
        return np.linalg.norm(a[m] - b[m]) / np.linalg.norm(b[m])

    corr = np.corrcoef(e_pred[m], e_act[m])[0, 1]
    print(f"Checkpoint {args.checkpoint} (version={version}, iter={ckpt.get('iter', '?')}), "
          f"closed windows, N={N_PEOPLE:.0f}, z={z:.2f} m, t={T_END:.0f} s")
    print(f"\n1) Is the error residual-driven?")
    print(f"   actual plane error ||e||/||ref||             : {np.linalg.norm(e_act[m]) / np.linalg.norm(ref[m]) * 100:.1f}%")
    print(f"   predicted from residual ||int r dt||/||ref|| : {np.linalg.norm(e_pred[m]) / np.linalg.norm(ref[m]) * 100:.1f}%")
    print(f"   correlation(predicted error map, actual)     : {corr:.3f}")
    print(f"   mismatch ||e_pred - e_act|| / ||e_act||      : {rel(e_pred, e_act) * 100:.1f}%")
    print("   -> correlation near 1 and similar magnitudes = the error IS the accumulated residual,")
    print("      so reducing the residual (in the right places) reduces the error.")
    print(f"\n2) Where is the error, and where does the loss look?  (near = S >= {NEAR_FRAC:.0%} of peak)")
    print(f"   share of plane points           : near {near.sum() / m.sum() * 100:5.1f}%   far {far.sum() / m.sum() * 100:5.1f}%")
    print(f"   share of squared ERROR           : near {err2_near / (err2_near + err2_far) * 100:5.1f}%   "
          f"far {err2_far / (err2_near + err2_far) * 100:5.1f}%")
    print(f"   share of training CO2 LOSS (r^2) : near {loss_near / (loss_near + loss_far) * 100:5.1f}%   "
          f"far {loss_far / (loss_near + loss_far) * 100:5.1f}%")
    print("   -> far field holding much more of the ERROR than of the LOSS = a weighting/sampling")
    print("      mismatch (fix: redistribute points or weights); similar shares = the loss already")
    print("      looks where the error is (then an optimizer/conditioning limit is more likely).")
    print(f"\n3) Residual at the source over time (fraction of S there):")
    src = np.argmin((xg - SX) ** 2 + (yg - SY) ** 2)
    print("   " + "  ".join(f"t={TIMES[i]:3.0f}s {R[i, src] / S_plane[src] * 100:+5.1f}%" for i in range(0, len(TIMES), 3)))

    fig_dir = os.path.join(HERE, "figures", version)
    os.makedirs(fig_dir, exist_ok=True)
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    r60 = R[-1].copy() / S_REF
    emax = np.nanmax(np.abs(np.concatenate([e_pred[m], e_act[m]])))
    panels = [(r60, f"residual r/S_REF at t={T_END:.0f} s", np.nanmax(np.abs(r60[m]))),
              (e_pred, "predicted error = integral of r dt", emax),
              (e_act, "actual error (GNOT - reference)", emax)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    for ax, (F, title, lim) in zip(axes, panels):
        F = F.astype(float).copy()
        F[inside] = np.nan
        im = ax.imshow(F.reshape(40, 40).T, origin="lower", extent=extent, cmap="RdBu_r",
                       vmin=-lim, vmax=lim, interpolation="bilinear")
        room_outline(ax)
        ax.contour(xs, ys, near.reshape(40, 40).T.astype(float), levels=[0.5], colors="k", linewidths=0.8,
                   linestyles="--")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle(f"{version}: residual-driven error check (closed windows, N={N_PEOPLE:.0f}, z={z:.2f} m; "
                 f"dashed = near-source region, correlation {corr:.2f})")
    fig.tight_layout()
    f = os.path.join(fig_dir, "residual_diagnosis_N20_z1.10_t60.png")
    fig.savefig(f, dpi=150, bbox_inches="tight")
    print(f"\nFigure written to {f}")


if __name__ == "__main__":
    main()
