"""
HFM feasibility test, take 3: extending the through-flow ("window") test
from 2D to 3D.

--------------------------------------------------------------------------
WHY THIS FILE EXISTS
--------------------------------------------------------------------------
Everything tested so far (hfm_synthetic_test.py: closed swirl,
hfm_channel_flow_test.py: 2D channel/window flow) was deliberately 2D --
a simplified proof-of-concept, the same way the original HFM paper
(Raissi, Yazdani, Karniadakis, Science 2020) itself validates its method
in 2D before moving to 3D. A real room is obviously 3D, so this file is
the next honest step: does the same method still work once we add the
third dimension?

--------------------------------------------------------------------------
WHAT CHANGES GOING FROM 2D TO 3D (read this before trusting any number
this script prints)
--------------------------------------------------------------------------
1. Divergence-free velocity trick. In 2D we used a single scalar
   "streamfunction" psi, with u=d(psi)/dy, v=-d(psi)/dx -- this
   automatically guarantees incompressibility (no made-up mass
   appearing/disappearing). In 3D that trick becomes a VECTOR potential
   A=(A1,A2,A3), with velocity = curl(A):
       u = dA3/dy - dA2/dz
       v = dA1/dz - dA3/dx
       w = dA2/dx - dA1/dy
   This is exactly what the real HFM paper does for its own 3D example
   (an aneurysm). It is the standard, paper-supported way to do this --
   not a new trick invented for this thesis.
2. More outputs, more derivatives, slower training. The network now
   predicts 3 potential components instead of 1, plus concentration,
   the auxiliary variable, and pressure -- 6 outputs instead of 4. The
   momentum equation also now has 3 components (u, v, AND w) instead of
   2, each needing second derivatives in x, y, AND z. Expect this to run
   noticeably slower per epoch than the 2D version, and the domain now
   needs 3D collocation points instead of 2D ones.
3. A real practical risk: sensor height coverage. If the real CO2
   sensors in a room all sit at roughly the same height, we will have
   almost no information about how concentration changes vertically --
   which means the vertical (z) velocity component may simply not be
   identifiable, the same "gradient-rich region" problem discussed in
   hfm_synthetic_test.py, but now in a new direction. This script's
   "sensors" are placed at MULTIPLE heights on purpose, specifically so
   we can later re-run it restricted to a single height and see how much
   that hurts -- mirroring how the 2D file compared all-sides vs.
   one-side boundary knowledge.

--------------------------------------------------------------------------
THE TEST SCENARIO
--------------------------------------------------------------------------
Deliberately the SIMPLEST possible 3D flow, chosen to validate the new
3D/vector-potential machinery first, before adding more realistic
complexity:
    u(x,y,z) = U0 * 4*z*(1-z),   v = 0,   w = 0
This is the same channel (Poiseuille) profile as the 2D test, just
carried through unchanged in the y-direction (no side-wall effect yet).
It is still an EXACT solution of the full viscous 3D Navier-Stokes
equations for any viscosity (same derivation as the 2D case, just with
z playing the role y played before) -- so, like the 2D channel test, no
manufactured forcing hack is needed.
This is intentionally a smaller step than a true rectangular duct flow
(which has side walls slowing the flow near y=0/y=1 too, and needs a
more complicated exact solution). If this simpler case works, a true
duct flow with side-wall effects is the natural next increment.

--------------------------------------------------------------------------
WHAT WAS AND WASN'T TESTED BY ME BEFORE HANDING THIS OVER
--------------------------------------------------------------------------
- The forward solver (pure NumPy) can be run in my sandbox and was
  checked for stability/boundedness.
- The curl-based velocity identity and the viscous-solution claim follow
  directly from the already-verified 2D derivation (z takes the role of
  y); not re-verified numerically in 3D specifically due to time, but
  the underlying math is the same.
- The PyTorch/PINN part could not be run in my environment (no PyTorch
  install available there). Please run it and report exactly what
  happens, same as the previous two files.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python hfm_channel_flow_3d_test.py                      # left face only (the "window")
    python hfm_channel_flow_3d_test.py --bc-sides all        # easier version, for comparison
    python hfm_channel_flow_3d_test.py --nx 20 --ny 20 --nz 20 --n-collocation 15000   # if too slow
"""

import argparse
import numpy as np


U0 = 0.6  # made-up peak channel-flow speed (matches the 2D test)


def true_velocity(x, y, z):
    """Channel (Poiseuille) flow, now embedded in 3D: parabolic in z,
    uniform in x and y, zero cross-flow. Exact solution of the full
    viscous 3D Navier-Stokes equations for any nu (same derivation as
    the 2D case: d^2u/dz^2 = -8*U0, matched by dp/dx = -8*nu*U0)."""
    u = U0 * 4.0 * z * (1.0 - z)
    zero = np.zeros_like(u) if hasattr(u, "shape") else 0.0
    return u, zero, zero


def solve_forward(nx=24, ny=24, nz=24, T=1.0, D=0.01, K=0.2,
                   source_xyz=(0.2, 0.5, 0.5), source_sigma=0.06, source_strength=5.0,
                   n_snapshots=16, verbose=True):
    """3D forward finite-difference solve of advection-diffusion.
    Boundary treatment mirrors the 2D through-flow case: fresh air at
    the inlet (x=0), free outflow at x=1, and no-flux (zero-gradient)
    at all four remaining walls (y=0, y=1, z=0, z=1)."""
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    dz = 1.0 / (nz - 1)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    U, V, W = true_velocity(X, Y, Z)

    h_min = min(dx, dy, dz)
    dt_diff = h_min * h_min / (6.0 * D)
    dt_adv = h_min / (np.max(np.abs(U)) + np.max(np.abs(V)) + np.max(np.abs(W)) + 1e-8)
    dt = 0.4 * min(dt_diff, dt_adv)
    n_steps = int(np.ceil(T / dt))
    dt = T / n_steps

    S = source_strength * np.exp(
        -((X - source_xyz[0]) ** 2 + (Y - source_xyz[1]) ** 2 + (Z - source_xyz[2]) ** 2)
        / (2 * source_sigma ** 2)
    )

    C = np.zeros((nx, ny, nz))
    snap_every = max(1, n_steps // (n_snapshots - 1))
    times, snaps = [0.0], [C.copy()]

    for step in range(1, n_steps + 1):
        dCdx = np.where(U > 0,
                         (C - np.roll(C, 1, axis=0)) / dx,
                         (np.roll(C, -1, axis=0) - C) / dx)
        dCdy = np.where(V > 0,
                         (C - np.roll(C, 1, axis=1)) / dy,
                         (np.roll(C, -1, axis=1) - C) / dy)
        dCdz = np.where(W > 0,
                         (C - np.roll(C, 1, axis=2)) / dz,
                         (np.roll(C, -1, axis=2) - C) / dz)
        laplacian = (
            (np.roll(C, -1, axis=0) - 2 * C + np.roll(C, 1, axis=0)) / dx ** 2
            + (np.roll(C, -1, axis=1) - 2 * C + np.roll(C, 1, axis=1)) / dy ** 2
            + (np.roll(C, -1, axis=2) - 2 * C + np.roll(C, 1, axis=2)) / dz ** 2
        )
        C = C + dt * (-U * dCdx - V * dCdy - W * dCdz + D * laplacian + S - K * C)

        C[0, :, :] = 0.0             # inlet (the "window"): fresh air
        C[-1, :, :] = C[-2, :, :]    # outlet: zero-gradient
        C[:, 0, :] = C[:, 1, :]      # side wall: zero-gradient
        C[:, -1, :] = C[:, -2, :]    # side wall: zero-gradient
        C[:, :, 0] = C[:, :, 1]      # floor: zero-gradient
        C[:, :, -1] = C[:, :, -2]    # ceiling: zero-gradient

        if step % snap_every == 0 or step == n_steps:
            times.append(step * dt)
            snaps.append(C.copy())

    snaps = np.stack(snaps, axis=0)
    times = np.array(times)

    if verbose:
        print(f"[forward solve] grid {nx}x{ny}x{nz}, dt={dt:.5f}, steps={n_steps}, "
              f"snapshots={len(times)}")
        print(f"[forward solve] concentration range: "
              f"[{snaps.min():.4f}, {snaps.max():.4f}]")
        if not np.isfinite(snaps).all():
            raise RuntimeError("Forward solve produced NaN/Inf -- solver is unstable, "
                                "reduce dt or check parameters before continuing.")

    return x, y, z, times, snaps, S


def find_gradient_rich_box(x, y, z, snaps, threshold_frac=0.05, pad_cells=2):
    """Same idea as the 2D scripts, extended to 3D."""
    c_max_over_time = snaps.max(axis=0)
    threshold = threshold_frac * c_max_over_time.max()
    mask = c_max_over_time > threshold
    ii, jj, kk = np.where(mask)
    nx, ny, nz = len(x), len(y), len(z)
    i_lo, i_hi = max(0, ii.min() - pad_cells), min(nx - 1, ii.max() + pad_cells)
    j_lo, j_hi = max(0, jj.min() - pad_cells), min(ny - 1, jj.max() + pad_cells)
    k_lo, k_hi = max(0, kk.min() - pad_cells), min(nz - 1, kk.max() + pad_cells)
    return (x[i_lo], x[i_hi], y[j_lo], y[j_hi], z[k_lo], z[k_hi],
            (i_lo, i_hi, j_lo, j_hi, k_lo, k_hi))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nx", type=int, default=24)
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--nz", type=int, default=24)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--D", type=float, default=0.01)
    parser.add_argument("--K", type=float, default=0.2)
    parser.add_argument("--nu", type=float, default=None,
                         help="viscosity for the momentum residual. Default: same as --D.")
    parser.add_argument("--n-sensors", type=int, default=30,
                         help="sensors are spread across multiple heights (z) on purpose -- "
                              "see docstring on the sensor-height identifiability risk.")
    parser.add_argument("--n-bc", type=int, default=300)
    parser.add_argument("--bc-sides", type=str, default="left",
                         help="which faces get known true velocity: comma list from "
                              "{left,right,front,back,top,bottom} or 'all'. "
                              "left/right = x=0/x=1 (window/outlet), "
                              "front/back = y=0/y=1 (side walls), "
                              "top/bottom = z=0/z=1 (ceiling/floor). Default 'left' only.")
    parser.add_argument("--n-collocation", type=int, default=20000)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-pde", type=float, default=10.0)
    parser.add_argument("--lambda-aux", type=float, default=10.0)
    parser.add_argument("--lambda-mom", type=float, default=10.0)
    parser.add_argument("--lambda-bc", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--single-height", action="store_true",
                         help="Restrict all sensors to ONE z-layer (the middle of the "
                              "gradient-rich box), instead of spreading them across "
                              "multiple heights. This mimics a real deployment where all "
                              "CO2 sensors sit at roughly the same height on a wall/ceiling "
                              "-- the realistic case flagged as a risk in this file's "
                              "docstring. Compare the result against the default "
                              "multi-height run to see how much vertical sensor coverage "
                              "actually matters.")
    parser.add_argument("--out", type=str, default="hfm_channel_3d_result.png")
    args = parser.parse_args()

    if args.nu is None:
        args.nu = args.D

    np.random.seed(args.seed)

    x, y, z, times, snaps, S = solve_forward(nx=args.nx, ny=args.ny, nz=args.nz,
                                              T=args.T, D=args.D, K=args.K)
    nx, ny, nz = len(x), len(y), len(z)

    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, (i_lo, i_hi, j_lo, j_hi, k_lo, k_hi) = \
        find_gradient_rich_box(x, y, z, snaps)
    print(f"[domain] gradient-rich box: x in [{x_lo:.3f}, {x_hi:.3f}], "
          f"y in [{y_lo:.3f}, {y_hi:.3f}], z in [{z_lo:.3f}, {z_hi:.3f}]")

    if args.single_height:
        k_mid = (k_lo + k_hi) // 2
        interior = [(i, j, k_mid)
                    for i in range(max(1, i_lo), min(nx - 1, i_hi) + 1)
                    for j in range(max(1, j_lo), min(ny - 1, j_hi) + 1)]
        print(f"[data] --single-height set: all sensors pinned to z={z[k_mid]:.3f} "
              f"(realistic single-sensor-height deployment)")
    else:
        interior = [(i, j, k)
                    for i in range(max(1, i_lo), min(nx - 1, i_hi) + 1)
                    for j in range(max(1, j_lo), min(ny - 1, j_hi) + 1)
                    for k in range(max(1, k_lo), min(nz - 1, k_hi) + 1)]
    idx = np.random.choice(len(interior), size=min(args.n_sensors, len(interior)), replace=False)
    sensor_ijk = [interior[m] for m in idx]
    sensor_xyz = np.array([[x[i], y[j], z[k]] for i, j, k in sensor_ijk])
    sensor_readings = np.stack([snaps[:, i, j, k] for i, j, k in sensor_ijk], axis=1)

    print(f"[data] {len(sensor_ijk)} fixed sensors (spread across "
          f"{len(set(round(v, 3) for v in sensor_xyz[:, 2]))} distinct heights) "
          f"x {len(times)} time snapshots")

    sides = [s.strip().lower() for s in args.bc_sides.split(",")] if args.bc_sides != "all" \
        else ["left", "right", "front", "back", "top", "bottom"]
    n_side = max(1, args.n_bc // max(1, len(sides)))

    def z_sample(n):
        # Keep the comparison clean: if sensors are restricted to one
        # height, the known boundary edge (the window) should be too --
        # otherwise the boundary leaks z-resolved velocity info the
        # sensors themselves don't have, which is what happened in the
        # first --single-height run (result barely changed because the
        # boundary still saw all heights).
        if args.single_height:
            return np.full(n, z[k_mid])
        return np.random.uniform(z_lo, z_hi, n)

    bx, by, bz = [], [], []
    if "left" in sides:
        bx.append(np.full(n_side, x_lo)); by.append(np.random.uniform(y_lo, y_hi, n_side)); bz.append(z_sample(n_side))
    if "right" in sides:
        bx.append(np.full(n_side, x_hi)); by.append(np.random.uniform(y_lo, y_hi, n_side)); bz.append(z_sample(n_side))
    if "front" in sides:
        bx.append(np.random.uniform(x_lo, x_hi, n_side)); by.append(np.full(n_side, y_lo)); bz.append(z_sample(n_side))
    if "back" in sides:
        bx.append(np.random.uniform(x_lo, x_hi, n_side)); by.append(np.full(n_side, y_hi)); bz.append(z_sample(n_side))
    if "top" in sides:
        bx.append(np.random.uniform(x_lo, x_hi, n_side)); by.append(np.random.uniform(y_lo, y_hi, n_side)); bz.append(np.full(n_side, z_hi))
    if "bottom" in sides:
        bx.append(np.random.uniform(x_lo, x_hi, n_side)); by.append(np.random.uniform(y_lo, y_hi, n_side)); bz.append(np.full(n_side, z_lo))
    if not bx:
        raise ValueError(f"--bc-sides '{args.bc_sides}' matched none of left/right/front/back/top/bottom")
    bc_x = np.concatenate(bx); bc_y = np.concatenate(by); bc_z = np.concatenate(bz)
    bc_t = np.random.uniform(0, args.T, len(bc_x))
    bc_u_true, bc_v_true, bc_w_true = true_velocity(bc_x, bc_y, bc_z)
    print(f"[data] known-velocity faces: {sides}")
    print(f"[data] {len(bc_x)} boundary points with known true velocity given")

    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[train] using device: {device}")

    class HFMNet3D(nn.Module):
        def __init__(self, hidden, n_layers):
            super().__init__()
            dims = [4] + [hidden] * n_layers + [6]  # (x,y,z,t) -> (A1,A2,A3,C,daux,p)
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.Tanh())
            self.net = nn.Sequential(*layers)

        def forward(self, x, y, z, t):
            out = self.net(torch.cat([x, y, z, t], dim=1))
            A1, A2, A3, C, daux, p = (out[:, 0:1], out[:, 1:2], out[:, 2:3],
                                       out[:, 3:4], out[:, 4:5], out[:, 5:6])
            return A1, A2, A3, C, daux, p

    def _d(f, wrt):
        return torch.autograd.grad(f, wrt, grad_outputs=torch.ones_like(f), create_graph=True)[0]

    def derive_fields(model, x, y, z, t):
        A1, A2, A3, C, daux, p = model(x, y, z, t)

        A1_y, A1_z = _d(A1, y), _d(A1, z)
        A2_x, A2_z = _d(A2, x), _d(A2, z)
        A3_x, A3_y = _d(A3, x), _d(A3, y)

        # velocity = curl(A) -- guarantees div(u)=0 by construction (same
        # role the 2D streamfunction played, generalized to 3D).
        u = A3_y - A2_z
        v = A1_z - A3_x
        w = A2_x - A1_y

        u_x, u_y, u_z, u_t = _d(u, x), _d(u, y), _d(u, z), _d(u, t)
        v_x, v_y, v_z, v_t = _d(v, x), _d(v, y), _d(v, z), _d(v, t)
        w_x, w_y, w_z, w_t = _d(w, x), _d(w, y), _d(w, z), _d(w, t)
        u_xx, u_yy, u_zz = _d(u_x, x), _d(u_y, y), _d(u_z, z)
        v_xx, v_yy, v_zz = _d(v_x, x), _d(v_y, y), _d(v_z, z)
        w_xx, w_yy, w_zz = _d(w_x, x), _d(w_y, y), _d(w_z, z)

        p_x, p_y, p_z = _d(p, x), _d(p, y), _d(p, z)

        C_x, C_y, C_z, C_t = _d(C, x), _d(C, y), _d(C, z), _d(C, t)
        C_xx, C_yy, C_zz = _d(C_x, x), _d(C_y, y), _d(C_z, z)

        d_x, d_y, d_z, d_t = _d(daux, x), _d(daux, y), _d(daux, z), _d(daux, t)
        d_xx, d_yy, d_zz = _d(d_x, x), _d(d_y, y), _d(d_z, z)

        return dict(
            u=u, v=v, w=w, C=C, daux=daux, p=p,
            u_x=u_x, u_y=u_y, u_z=u_z, u_t=u_t, u_xx=u_xx, u_yy=u_yy, u_zz=u_zz,
            v_x=v_x, v_y=v_y, v_z=v_z, v_t=v_t, v_xx=v_xx, v_yy=v_yy, v_zz=v_zz,
            w_x=w_x, w_y=w_y, w_z=w_z, w_t=w_t, w_xx=w_xx, w_yy=w_yy, w_zz=w_zz,
            p_x=p_x, p_y=p_y, p_z=p_z,
            C_x=C_x, C_y=C_y, C_z=C_z, C_t=C_t, C_xx=C_xx, C_yy=C_yy, C_zz=C_zz,
            d_x=d_x, d_y=d_y, d_z=d_z, d_t=d_t, d_xx=d_xx, d_yy=d_yy, d_zz=d_zz,
        )

    def source_term(x, y, z):
        return 5.0 * torch.exp(
            -((x - 0.2) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) / (2 * 0.06 ** 2)
        )

    def physics_losses(model, cx, cy, cz, ct, D, K, nu):
        f = derive_fields(model, cx, cy, cz, ct)
        S_val = source_term(cx, cy, cz)

        transport = (f["C_t"] + f["u"] * f["C_x"] + f["v"] * f["C_y"] + f["w"] * f["C_z"]
                     - D * (f["C_xx"] + f["C_yy"] + f["C_zz"]) - S_val + K * f["C"])
        aux = (f["d_t"] + f["u"] * f["d_x"] + f["v"] * f["d_y"] + f["w"] * f["d_z"]
               - D * (f["d_xx"] + f["d_yy"] + f["d_zz"]) + S_val - K + K * f["daux"])

        # No manufactured forcing needed -- this 3D channel flow is an
        # exact viscous solution for any nu, same as the 2D case.
        mom_u = (f["u_t"] + f["u"] * f["u_x"] + f["v"] * f["u_y"] + f["w"] * f["u_z"]
                 + f["p_x"] - nu * (f["u_xx"] + f["u_yy"] + f["u_zz"]))
        mom_v = (f["v_t"] + f["u"] * f["v_x"] + f["v"] * f["v_y"] + f["w"] * f["v_z"]
                 + f["p_y"] - nu * (f["v_xx"] + f["v_yy"] + f["v_zz"]))
        mom_w = (f["w_t"] + f["u"] * f["w_x"] + f["v"] * f["w_y"] + f["w"] * f["w_z"]
                 + f["p_z"] - nu * (f["w_xx"] + f["w_yy"] + f["w_zz"]))

        transport_loss = torch.mean(transport ** 2)
        aux_loss = torch.mean(aux ** 2)
        momentum_loss = torch.mean(mom_u ** 2) + torch.mean(mom_v ** 2) + torch.mean(mom_w ** 2)
        return transport_loss, aux_loss, momentum_loss

    model = HFMNet3D(args.hidden, args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)], gamma=0.1
    )
    mse = nn.MSELoss()

    n_t = len(times)
    n_sensors = len(sensor_ijk)
    sx = np.repeat(sensor_xyz[:, 0], n_t)
    sy = np.repeat(sensor_xyz[:, 1], n_t)
    sz = np.repeat(sensor_xyz[:, 2], n_t)
    st = np.tile(times, n_sensors)
    sC = sensor_readings.T.reshape(-1)

    data_x = torch.tensor(sx, dtype=torch.float32, device=device).view(-1, 1)
    data_y = torch.tensor(sy, dtype=torch.float32, device=device).view(-1, 1)
    data_z = torch.tensor(sz, dtype=torch.float32, device=device).view(-1, 1)
    data_t = torch.tensor(st, dtype=torch.float32, device=device).view(-1, 1)
    data_C = torch.tensor(sC, dtype=torch.float32, device=device).view(-1, 1)
    data_daux = torch.tensor(1.0 - sC, dtype=torch.float32, device=device).view(-1, 1)

    bc_x_t = torch.tensor(bc_x, dtype=torch.float32, device=device).view(-1, 1)
    bc_y_t = torch.tensor(bc_y, dtype=torch.float32, device=device).view(-1, 1)
    bc_z_t = torch.tensor(bc_z, dtype=torch.float32, device=device).view(-1, 1)
    bc_t_t = torch.tensor(bc_t, dtype=torch.float32, device=device).view(-1, 1)
    bc_u_t = torch.tensor(bc_u_true, dtype=torch.float32, device=device).view(-1, 1)
    bc_v_t = torch.tensor(bc_v_true, dtype=torch.float32, device=device).view(-1, 1)
    bc_w_t = torch.tensor(bc_w_true, dtype=torch.float32, device=device).view(-1, 1)

    def sample_collocation(n):
        cx = (x_lo + torch.rand(n, 1, device=device) * (x_hi - x_lo)).requires_grad_(True)
        cy = (y_lo + torch.rand(n, 1, device=device) * (y_hi - y_lo)).requires_grad_(True)
        cz = (z_lo + torch.rand(n, 1, device=device) * (z_hi - z_lo)).requires_grad_(True)
        ct = (torch.rand(n, 1, device=device) * args.T).requires_grad_(True)
        return cx, cy, cz, ct

    print(f"[train] training for {args.epochs} epochs, {args.n_collocation} collocation "
          f"points/epoch (3D -- expect slower per-epoch time than the 2D scripts)")

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        data_fields = derive_fields(
            model,
            data_x.clone().requires_grad_(True),
            data_y.clone().requires_grad_(True),
            data_z.clone().requires_grad_(True),
            data_t.clone().requires_grad_(True),
        )
        data_loss = mse(data_fields["C"], data_C) + mse(data_fields["daux"], data_daux)

        bc_fields = derive_fields(
            model,
            bc_x_t.clone().requires_grad_(True),
            bc_y_t.clone().requires_grad_(True),
            bc_z_t.clone().requires_grad_(True),
            bc_t_t.clone().requires_grad_(True),
        )
        bc_loss = mse(bc_fields["u"], bc_u_t) + mse(bc_fields["v"], bc_v_t) + mse(bc_fields["w"], bc_w_t)

        cx, cy, cz, ct = sample_collocation(args.n_collocation)
        transport_loss, aux_loss, momentum_loss = physics_losses(model, cx, cy, cz, ct, args.D, args.K, args.nu)

        loss = (data_loss + args.lambda_bc * bc_loss
                + args.lambda_pde * transport_loss + args.lambda_aux * aux_loss
                + args.lambda_mom * momentum_loss)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(f"  epoch {epoch:6d}  lr={scheduler.get_last_lr()[0]:.1e}  "
                  f"data={data_loss.item():.6e}  bc={bc_loss.item():.6e}  "
                  f"transport={transport_loss.item():.6e}  aux={aux_loss.item():.6e}  "
                  f"momentum={momentum_loss.item():.6e}  total={loss.item():.6e}")

    model.eval()
    gx_full, gy_full, gz_full = np.meshgrid(x, y, z, indexing="ij")
    box_mask = ((gx_full >= x_lo) & (gx_full <= x_hi) &
                (gy_full >= y_lo) & (gy_full <= y_hi) &
                (gz_full >= z_lo) & (gz_full <= z_hi))
    gx, gy, gz = gx_full[box_mask], gy_full[box_mask], gz_full[box_mask]
    u_true, v_true, w_true = true_velocity(gx, gy, gz)
    speed_true = np.sqrt(u_true ** 2 + v_true ** 2 + w_true ** 2)

    def eval_at(t_val):
        ex = torch.tensor(gx.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
        ey = torch.tensor(gy.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
        ez = torch.tensor(gz.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
        et = torch.full_like(ex, t_val, requires_grad=True)
        f = derive_fields(model, ex, ey, ez, et)
        u_pred = f["u"].detach().cpu().numpy().reshape(-1)
        v_pred = f["v"].detach().cpu().numpy().reshape(-1)
        w_pred = f["w"].detach().cpu().numpy().reshape(-1)
        # Reference scale for v/w relative error: use the overall speed
        # scale (norm of u_true), NOT norm(v_true)/norm(w_true) alone --
        # when the true flow has zero (or near-zero) sideways/vertical
        # component, dividing by its own near-zero norm makes the
        # relative error meaningless (it can read billions of percent
        # even when the absolute prediction error is tiny). This mirrors
        # a near-zero-denominator metric artifact found earlier in
        # pinn_reconstruction_test.py.
        scale = np.linalg.norm(speed_true) + 1e-8
        rel_err_u = np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-8)
        rel_err_v = np.linalg.norm(v_pred - v_true) / scale
        rel_err_w = np.linalg.norm(w_pred - w_true) / scale
        rms_v = np.sqrt(np.mean(v_pred ** 2))
        rms_w = np.sqrt(np.mean(w_pred ** 2))
        speed_pred = np.sqrt(u_pred ** 2 + v_pred ** 2 + w_pred ** 2)
        rel_err_speed = np.linalg.norm(speed_pred - speed_true) / (np.linalg.norm(speed_true) + 1e-8)
        return rel_err_u, rel_err_v, rel_err_w, rel_err_speed, rms_v, rms_w

    print(f"\n=== RESULT: recovered vs. true velocity, INSIDE THE GRADIENT-RICH BOX ONLY ===")
    print(f"  (NOTE: true v and w are exactly 0 in this test flow, so their % errors are "
          f"normalized against the overall speed scale, not their own near-zero norm --")
    print(f"   RMS(v_pred)/RMS(w_pred) are also shown directly, in the same units as u, so "
          f"you can judge how large the spurious sideways/vertical velocity really is.)")
    for frac in [0.3, 0.5, 0.7, 1.0]:
        t_val = frac * args.T
        ru, rv, rw, rs, rms_v, rms_w = eval_at(t_val)
        print(f"  t={t_val:.2f} (frac={frac}): u err={100*ru:.2f}%  speed err={100*rs:.2f}%  "
              f"| v err (vs speed scale)={100*rv:.2f}%  w err (vs speed scale)={100*rw:.2f}%  "
              f"| RMS(v_pred)={rms_v:.4f}  RMS(w_pred)={rms_w:.4f}  (U0={U0})")

    ru, rv, rw, rs, rms_v, rms_w = eval_at(0.5 * args.T)
    print(f"\n  Primary result (mid-time, t={0.5*args.T:.2f}): speed error {100*rs:.2f}%")
    print(f"  Boundary condition given on: {sides} (default 'left' = window/inlet only)")
    print(f"  NOTE: this run's sensors span multiple heights (z). If you want to test "
          f"the realistic single-height-sensor case, edit find_gradient_rich_box usage "
          f"or filter sensor_ijk to a fixed k before training, and compare.")


if __name__ == "__main__":
    main()
