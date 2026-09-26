# ============================================================================
# MILESTONE SNAPSHOT: v10_hardic (2026-09-26) -- frozen copy, DO NOT EDIT.
# Model/training code from git commit 6529b15 (exactly what the v10 run trained
# with); fd_reference_closed_room.py from a0b2992. See README.md. Run scripts
# from INSIDE this folder so they import this frozen model code.
# ============================================================================

"""
Independent FINITE-DIFFERENCE reference solution for the CLOSED-WINDOW case,
to validate the PINN against something that is not a neural network.

Why this is a valid reference: with all windows closed the exact velocity is
zero (no forcing), so CO2 obeys pure diffusion with a source,
    dc/dt = D lap(c) + S,   S = N * E * exp(-dist^2 / sigma^2),   c(t=0) = 0,
with no-flux on walls, floor, ceiling and columns, and zero-gradient at the
doors (= no-flux when u = 0). That is solved here with a standard
second-order finite-volume/finite-difference scheme and explicit Euler time
stepping on a cell-centred grid -- same physical constants as training.

Two treatments of the (closed) windows:
  noflux    -- physically correct: a closed window is a wall.
  dirichlet -- c = 0 at the window openings: what the PINN is currently
               trained with (windows_loss applies c=0 regardless of V).
Comparing the two measures how much that modelling choice matters.

Usage:
    python3 fd_reference_closed_room.py [--dx 0.1] [--ckpt <pinn checkpoint> ...]
Run once with --dx 0.1 and once with --dx 0.05: if the two agree, the
reference is grid-converged. Pure numpy (the optional PINN comparison needs
torch and a checkpoint in the LIVE model format).
"""
import argparse
import math
import numpy as np

from point_sampler import (ROOM_X, ROOM_Y, ROOM_Z, WINDOWS, COLUMNS, CO2_SOURCE_SIGMA,
                           BREATHING_HEIGHT, EMISSION_PER_PERSON, NUM_WINDOWS)
from train_gnot import DIFFUSIVITY

N_PEOPLE = 20.0
TIMES = [0.0, 10.0, 30.0, 60.0, 90.0, 120.0]
SX, SY = (ROOM_X[0] + ROOM_X[1]) / 2, (ROOM_Y[0] + ROOM_Y[1]) / 2
PROBES = {"source (7.76,4.58)": (SX, SY), "2.5 m east of source": (SX + 2.5, SY),
          "far corner (1.0,1.0)": (1.0, 1.0)}


def solve(dx_target, windows):
    Lx, Ly, Lz = ROOM_X[1] - ROOM_X[0], ROOM_Y[1] - ROOM_Y[0], ROOM_Z[1] - ROOM_Z[0]
    nx, ny, nz = (max(4, round(L / dx_target)) for L in (Lx, Ly, Lz))
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    xc = ROOM_X[0] + (np.arange(nx) + 0.5) * dx
    yc = ROOM_Y[0] + (np.arange(ny) + 0.5) * dy
    zc = ROOM_Z[0] + (np.arange(nz) + 0.5) * dz
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")

    fluid = np.ones((nx, ny, nz), dtype=bool)
    for cx, cy, r, _, _ in COLUMNS:          # floor-to-ceiling solid columns
        fluid &= (X - cx) ** 2 + (Y - cy) ** 2 > r ** 2
    S = N_PEOPLE * EMISSION_PER_PERSON * np.exp(
        -((X - SX) ** 2 + (Y - SY) ** 2 + (Z - BREATHING_HEIGHT) ** 2) / CO2_SOURCE_SIGMA ** 2) * fluid

    # open faces between neighbouring FLUID cells (no flux into solids / through walls)
    ox = fluid[:-1] & fluid[1:]
    oy = fluid[:, :-1] & fluid[:, 1:]
    oz = fluid[:, :, :-1] & fluid[:, :, 1:]
    # window openings on the y = ROOM_Y[1] wall (windows span floor to ceiling)
    win = np.zeros(nx, dtype=bool)
    for xlo, xhi, _, _ in WINDOWS:
        win |= (xc >= xlo) & (xc <= xhi)
    win_cells = win[:, None] & fluid[:, -1, :]            # (nx, nz)

    dt_max = 0.9 / (2 * DIFFUSIVITY * (1 / dx ** 2 + 1 / dy ** 2 + 1 / dz ** 2))
    n_sub = math.ceil(10.0 / dt_max)
    dt = 10.0 / n_sub                                     # divides every output time exactly
    c = np.zeros((nx, ny, nz))
    out, t = {0.0: c.copy()}, 0.0
    D = DIFFUSIVITY
    while t < TIMES[-1] - 1e-9:
        for _ in range(n_sub):
            lap = np.zeros_like(c)
            fx = (c[1:] - c[:-1]) * ox / dx ** 2
            lap[:-1] += fx; lap[1:] -= fx
            fy = (c[:, 1:] - c[:, :-1]) * oy / dy ** 2
            lap[:, :-1] += fy; lap[:, 1:] -= fy
            fz = (c[:, :, 1:] - c[:, :, :-1]) * oz / dz ** 2
            lap[:, :, :-1] += fz; lap[:, :, 1:] -= fz
            if windows == "dirichlet":                    # c = 0 on the window face (distance dy/2)
                lap[:, -1, :] -= 2.0 * c[:, -1, :] * win_cells / dy ** 2
            c = (c + dt * (D * lap + S)) * fluid
        t = round(t + 10.0, 6)
        out[t] = c.copy()
    grid = (ROOM_X[0], ROOM_Y[0], ROOM_Z[0], dx, dy, dz, nx, ny, nz)
    return out, grid, (nx, ny, nz, dt)


def interp(c, grid, px, py, pz):
    """Trilinear interpolation of the cell-centred field c at points (px, py, pz)."""
    x0, y0, z0, dx, dy, dz, nx, ny, nz = grid
    res = 1.0
    idx, wts = [], []
    for p, o, d, n in ((px, x0, dx, nx), (py, y0, dy, ny), (pz, z0, dz, nz)):
        f = np.clip((np.asarray(p, dtype=float) - o) / d - 0.5, 0, n - 1)
        i0 = np.minimum(np.floor(f).astype(int), n - 2)
        idx.append(i0); wts.append(f - i0)
    (i, j, k), (wx, wy, wz) = idx, wts
    res = 0.0
    for di, a in ((0, 1 - wx), (1, wx)):
        for dj, b in ((0, 1 - wy), (1, wy)):
            for dk, g in ((0, 1 - wz), (1, wz)):
                res = res + a * b * g * c[i + di, j + dj, k + dk]
    return res


def breathing_grid():
    xs = np.linspace(ROOM_X[0] + 0.1, ROOM_X[1] - 0.1, 40)
    ys = np.linspace(ROOM_Y[0] + 0.1, ROOM_Y[1] - 0.1, 40)
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    xg, yg = Xg.ravel(), Yg.ravel()
    inside = np.zeros_like(xg, dtype=bool)
    for cx, cy, r, _, _ in COLUMNS:
        inside |= (xg - cx) ** 2 + (yg - cy) ** 2 <= r ** 2
    return xs, ys, xg, yg, inside


def pinn_on(model, device, xg, yg, t):
    import torch
    n = len(xg)
    with torch.no_grad():
        _, _, _, C, _ = model(
            torch.tensor(xg, dtype=torch.float32, device=device).view(-1, 1),
            torch.tensor(yg, dtype=torch.float32, device=device).view(-1, 1),
            torch.full((n, 1), BREATHING_HEIGHT, device=device),
            torch.full((n, 1), float(t), device=device),
            torch.zeros(n, NUM_WINDOWS, device=device),
            torch.full((n, 1), N_PEOPLE, device=device))
    return C.cpu().numpy().ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.1, help="target grid spacing in metres")
    ap.add_argument("--ckpt", nargs="*", default=[], help="PINN checkpoint(s) to compare")
    args = ap.parse_args()

    xs, ys, xg, yg, inside = breathing_grid()
    zb = np.full_like(xg, BREATHING_HEIGHT)
    refs = {}
    for windows in ("noflux", "dirichlet"):
        out, grid, (nx, ny, nz, dt) = solve(args.dx, windows)
        refs[windows] = (out, grid)
        print(f"\n=== FD reference, closed windows treated as {windows.upper()} "
              f"(grid {nx}x{ny}x{nz}, dx~{args.dx} m, dt={dt:.3f} s) -- N=20, z=1.10 m ===")
        print(f"{'point':24s} " + " ".join(f"t={t:>4.0f}s" for t in TIMES))
        for name, (px, py) in PROBES.items():
            vals = [float(interp(out[t], grid, px, py, BREATHING_HEIGHT)) for t in TIMES]
            print(f"{name:24s} " + " ".join(f"{v:+7.4f}" for v in vals))
        f60 = interp(out[60.0], grid, xg, yg, zb).astype(float)
        f60[inside] = np.nan
        G = f60.reshape(40, 40)
        pi, pj = np.unravel_index(np.nanargmax(G), G.shape)
        half = np.nanmin(G) + (np.nanmax(G) - np.nanmin(G)) / 2
        print(f"t=60 s field: peak {np.nanmax(G):.4f} at ({xs[pi]:.2f}, {ys[pj]:.2f}); "
              f"above half-max along y {np.sum(G[pi, :] > half)}/40, along x {np.sum(G[:, pj] > half)}/40")

    if args.ckpt:
        import torch
        from gnot_model import GNOTOperator, check_checkpoint_compat
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for path in args.ckpt:
            ckpt = torch.load(path, map_location=device)
            check_checkpoint_compat(ckpt, path)
            model = GNOTOperator().to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            print(f"\n=== PINN {path} (iter={ckpt.get('iter', '?')}) vs FD reference, breathing height ===")
            for t in (30.0, 60.0, 120.0):
                p = pinn_on(model, device, xg, yg, t)
                for windows, (out, grid) in refs.items():
                    r = interp(out[t], grid, xg, yg, zb).astype(float)
                    m = ~inside
                    rel = np.linalg.norm(p[m] - r[m]) / np.linalg.norm(r[m])
                    print(f"  t={t:5.0f}s vs FD[{windows:9s}]: relative L2 error {rel * 100:6.2f}%, "
                          f"max |error| {np.max(np.abs(p[m] - r[m])):.4f} (field max {r[m].max():.4f})")


if __name__ == "__main__":
    main()
