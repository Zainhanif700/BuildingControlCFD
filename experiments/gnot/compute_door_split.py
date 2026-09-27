"""
v20: potential-flow door split per window (used by throughflow.py as the base door split).

For each window k, solve the potential-flow problem
    lap(theta) = 0 in the air (columns are solid),
    d theta/dn = -1 on window k (unit inflow), 0 on all other surfaces (walls, floor,
    ceiling, columns, other windows), theta = 0 on both doors (outlets at equal pressure),
u = grad(theta), and record r_k = (outflow through door 1) / (total outflow).
The through-flow of a scenario then splits as
    alpha_pot(V) = sum_k V_k A_k r_k / sum_k V_k A_k        (share leaving by door 1).

Why: in v19 the split alpha was a free learned scalar and barely moved in 5000 iterations
(0.47 / 0.50 / 0.53 for W1-only / W8-only / all open, although W1 sits right across from
door 1). Potential flow (inviscid, irrotational) gives a physically sensible split from the
geometry; the model keeps a learned correction on top (gnot_model.door_split).

Numerics: cell-centred finite volumes on a uniform grid, matrix-free 7-point operator,
Dirichlet doors via the half-cell ghost value, Jacobi-preconditioned conjugate gradients
(pure numpy). Openings are represented by the cells whose centre lies inside them, so their
discrete areas depend on dx (at dx = 0.1 door 2 is 10% larger than door 1); use a spacing that
resolves both doors equally -- dx = 0.12 (used in throughflow.py). Across dx = 0.075-0.2 the
shares scatter by about +-0.006.

Usage: python3 compute_door_split.py [--dx 0.2] [--tol 1e-9]
"""
import argparse
import time

import numpy as np

from point_sampler import ROOM_X, ROOM_Y, ROOM_Z, WINDOWS, DOORS, COLUMNS


def build(dx):
    L = [ROOM_X[1] - ROOM_X[0], ROOM_Y[1] - ROOM_Y[0], ROOM_Z[1] - ROOM_Z[0]]
    n = [max(4, round(l / dx)) for l in L]
    h = [l / m for l, m in zip(L, n)]
    c = [ROOM_X[0] + (np.arange(n[0]) + 0.5) * h[0], ROOM_Y[0] + (np.arange(n[1]) + 0.5) * h[1],
         ROOM_Z[0] + (np.arange(n[2]) + 0.5) * h[2]]
    X, Y = np.meshgrid(c[0], c[1], indexing="ij")
    fluid2 = np.ones(X.shape, bool)
    for cx, cy, r, _, _ in COLUMNS:
        fluid2 &= (X - cx) ** 2 + (Y - cy) ** 2 > r ** 2
    fluid = np.repeat(fluid2[:, :, None], n[2], axis=2)                  # columns floor-to-ceiling
    # door faces: y = 0 face of the j = 0 cells inside a door rectangle
    door = np.zeros((len(DOORS), n[0], n[2]), bool)
    for d, (a, b, z0, z1) in enumerate(DOORS):
        door[d] = ((c[0] >= a) & (c[0] <= b))[:, None] & ((c[2] >= z0) & (c[2] <= z1))[None, :]
    win = np.zeros((len(WINDOWS), n[0], n[2]), bool)
    for k, (a, b, z0, z1) in enumerate(WINDOWS):
        win[k] = ((c[0] >= a) & (c[0] <= b))[:, None] & ((c[2] >= z0) & (c[2] <= z1))[None, :]
    door &= fluid[:, 0, :][None]
    win &= fluid[:, -1, :][None]
    return n, h, fluid, door, win


def make_operator(n, h, fluid, door_any):
    """A theta = -(sum of face fluxes)/volume (SPD thanks to the Dirichlet doors)."""
    open_ = [fluid[:-1] & fluid[1:], fluid[:, :-1] & fluid[:, 1:], fluid[:, :, :-1] & fluid[:, :, 1:]]
    dir_coef = np.zeros(fluid.shape)
    dir_coef[:, 0, :] = 2.0 * door_any / h[1] ** 2                     # (0 - theta)/(h/2) / h

    def A(th):
        out = dir_coef * th
        for ax in range(3):
            sl_lo = [slice(None)] * 3; sl_hi = [slice(None)] * 3
            sl_lo[ax] = slice(None, -1); sl_hi[ax] = slice(1, None)
            f = (th[tuple(sl_hi)] - th[tuple(sl_lo)]) * open_[ax] / h[ax] ** 2
            out[tuple(sl_lo)] -= f
            out[tuple(sl_hi)] += f
        return out * fluid
    diag = dir_coef.copy()
    for ax in range(3):
        cnt = np.zeros(fluid.shape)
        sl_lo = [slice(None)] * 3; sl_hi = [slice(None)] * 3
        sl_lo[ax] = slice(None, -1); sl_hi[ax] = slice(1, None)
        cnt[tuple(sl_lo)] += open_[ax]
        cnt[tuple(sl_hi)] += open_[ax]
        diag += cnt / h[ax] ** 2
    diag = np.where(fluid, diag, 1.0)
    return A, diag


def pcg(A, b, diag, tol, maxit=200000):
    x = np.zeros_like(b)
    r = b - A(x)
    z = r / diag
    p = z.copy()
    rz = np.sum(r * z)
    bn = np.sqrt(np.sum(b * b))
    for it in range(maxit):
        Ap = A(p)
        alpha = rz / np.sum(p * Ap)
        x += alpha * p
        r -= alpha * Ap
        if np.sqrt(np.sum(r * r)) < tol * bn:
            return x, it + 1
        z = r / diag
        rz_new = np.sum(r * z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    raise RuntimeError("PCG did not converge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.2)
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()
    n, h, fluid, door, win = build(args.dx)
    A, diag = make_operator(n, h, fluid, door.any(0))
    print(f"grid {n[0]}x{n[1]}x{n[2]} (h = {h[0]:.3f}, {h[1]:.3f}, {h[2]:.3f} m)")
    ratios = []
    for k in range(len(WINDOWS)):
        t0 = time.time()
        # FV balance for a window cell: sum_interior (th_nb - th)/h^2 + (d th/dn)/h = 0 with
        # d th/dn = -1 (unit inflow)  ->  -A th - 1/h = 0  ->  A th = -1/h
        b = np.zeros(fluid.shape)
        b[:, -1, :] = -win[k].astype(float) / h[1]
        th, its = pcg(A, b, diag, args.tol)
        # outward flux through each door: d th/dn (n = -y) = -(th_cell - 0)/(h/2)
        fd = [np.sum(-2.0 * th[:, 0, :][door[d]] / h[1]) * h[0] * h[2] for d in range(len(DOORS))]
        fin = np.sum(win[k]) * h[0] * h[2]              # inflow (unit speed x discrete window area)
        ratios.append(fd[0] / (fd[0] + fd[1]))
        print(f"window {k + 1}: door outflow {fd[0]:.4f} + {fd[1]:.4f} = {fd[0] + fd[1]:.4f} "
              f"(inflow {fin:.4f}, balance {100 * (fd[0] + fd[1] - fin) / fin:+.2f}%) -> r_{k + 1} = {ratios[-1]:.4f} "
              f"[{its} PCG its, {time.time() - t0:.0f} s]", flush=True)
    print("\nDOOR1_SHARE_PER_WINDOW = [" + ", ".join(f"{r:.4f}" for r in ratios) + "]")


if __name__ == "__main__":
    main()
