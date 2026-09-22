"""
HFM feasibility test, take 2: a through-flow "window" room instead of a
closed swirl.

--------------------------------------------------------------------------
WHY THIS FILE EXISTS -- THE STORY SO FAR, IN PLAIN LANGUAGE (for explaining
to your professor)
--------------------------------------------------------------------------
The question behind all of this: can we recover a room's airflow from CO2
sensor readings alone, using the "Hidden Fluid Mechanics" (HFM) method from
a 2018/2020 paper by Raissi, Yazdani and Karniadakis -- so we can extend
your thesis's neural-operator method to natural window ventilation without
running new CFD simulations, which we don't have time or resources for.

We tested this on a made-up 2D room with a single closed recirculating
swirl (hfm_synthetic_test.py, in this same folder), and iterated it through
several rounds of fixes as we found real problems, each verified by
running it and checking the numbers, not just assumed:
  1. First attempt: only gave the network the CO2 transport equation.
     Result: 80-450% velocity error. Root cause: the transport equation
     only constrains velocity ALONG the direction concentration is
     changing, not across it -- a real, provable mathematical
     non-uniqueness (the same "aperture problem" as in optical flow).
  2. Added the full Navier-Stokes momentum equations (as the real HFM
     paper does), hoping the extra physics would remove that freedom.
     Result: WORSE (270-840% error). Investigating why led us to re-read
     the actual paper closely, where we found three things our test was
     missing: (a) the paper only claims velocity recovery where
     concentration has real, meaningful gradient -- we were scoring over
     the whole room including flat, signal-free areas; (b) the paper
     supplies a known velocity at one boundary when its own signal is
     weak there; (c) a helper "d = 1-C" variable the paper uses.
  3. Fixed all three, plus two more we verified independently: our chosen
     viscosity implied an unrealistically turbulent flow regime the paper
     never tested (Reynolds number ~42,000 vs. the paper's 60-185), and
     our made-up swirl needed a "manufactured forcing" correction term to
     stay physically consistent once viscosity mattered. Result with a
     GENEROUS boundary condition (true velocity given all around the
     region's edge): 1.16% velocity error -- a real success.
  4. But that boundary condition was generous compared to the paper's own
     benchmarks, which only give velocity at ONE inlet edge. Tightening
     to just one edge: error jumped back up to ~54%.

Investigating that gap is what led to THIS file. The paper's benchmarks
that succeed with just one known edge (flow past a cylinder, flow through
a channel) all have real THROUGH-FLOW: fluid enters one side and exits
another. Our swirl was a closed recirculating cell with no throughflow at
all -- a structurally different, and probably harder, kind of flow than
anything the paper actually validated its "one boundary is enough" claim
on. This matters for your thesis in a good way: a real window room -- air
comes in one opening, goes out another -- is naturally a through-flow
room, much closer to what the paper tested than our closed swirl was. So
this file replaces the swirl with the simplest possible through-flow
case: classic channel (Poiseuille) flow, one inlet, one outlet, no-slip
walls top and bottom -- exactly the kind of setup in the paper's own
"channel flow over an obstacle" benchmark (their Figure 11-14).

--------------------------------------------------------------------------
WHAT'S DIFFERENT FROM hfm_synthetic_test.py
--------------------------------------------------------------------------
- The "true" airflow is now steady channel (Poiseuille) flow: enters at
  x=0 with a parabolic profile (fast in the middle, zero at the walls,
  like a window's airflow settling into the room), flows straight across,
  exits at x=1. This is a genuine textbook-exact solution of the full
  viscous Navier-Stokes equations for ANY viscosity -- unlike the swirl,
  it does NOT need a manufactured forcing correction; the physics is
  consistent by construction.
- The known-boundary-velocity condition (the paper's "inlet" trick) is
  given ONLY at the left edge (x=0) -- the "window" -- by default, not
  all four sides. Top and bottom (the walls) and the right edge (the
  "outlet", e.g. a door or second window) are left for the network to
  infer purely from concentration and physics, matching the paper's own
  benchmark design.
- Concentration boundary handling changed to match a through-flow room:
  fresh air (C=0) enters at the inlet, CO2 exits freely at the outlet
  (zero-gradient, not clamped to zero), and no CO2 passes through the
  solid walls (zero-gradient there too).
Everything else (domain restriction to where concentration signal
actually exists, the auxiliary d variable, the momentum equations, the
Adam-with-decaying-learning-rate recipe, scoring at mid-time rather than
the edge of the time window) carries over unchanged from the previous
file, all still verified independently before being adopted.

--------------------------------------------------------------------------
WHAT WAS AND WASN'T TESTED BEFORE HANDING THIS TO YOU
--------------------------------------------------------------------------
- The forward solver (pure NumPy) and the channel-flow formula's
  consistency with the viscous momentum equation were checked by hand
  (numerically verified: d^2u/dy^2 matches the analytic value exactly).
- The PyTorch/PINN part could not be run in my environment (no network
  access there to install PyTorch). Please run it and report exactly
  what happens.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python hfm_channel_flow_test.py                     # left edge only (matches the paper)
    python hfm_channel_flow_test.py --bc-sides all       # easier version, for comparison
"""

import argparse
import numpy as np


U0 = 0.6  # made-up peak channel-flow speed (at the center, y=0.5)


def true_velocity(x, y):
    """Classic Poiseuille (channel) flow: parabolic profile in y, uniform
    in x, zero cross-flow. Zero at the walls y=0 and y=1 (no-slip), peak
    at the centerline y=0.5. This is an exact solution of the steady
    viscous Navier-Stokes equations for ANY viscosity nu, given the
    matching pressure field p(x) = -8*nu*U0*x (linear pressure drop along
    the flow direction) -- verified by hand (see docstring)."""
    u = U0 * 4.0 * y * (1.0 - y)
    v = np.zeros_like(u) if hasattr(u, "shape") else 0.0
    return u, v


def solve_forward(nx=64, ny=64, T=1.0, D=0.01, K=0.2,
                   source_xy=(0.2, 0.5), source_sigma=0.05, source_strength=5.0,
                   n_snapshots=21, verbose=True):
    """Forward finite-difference solve of advection-diffusion in a channel:
    fresh air (C=0) enters at the left, CO2 exits freely at the right
    (zero-gradient), no CO2 passes through the top/bottom walls
    (zero-gradient there too) -- physically appropriate for a through-flow
    room, unlike the closed-box zero-everywhere boundary used for the
    swirl case."""
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    U, V = true_velocity(X, Y)

    dt_diff = dx * dx / (4.0 * D)
    dt_adv = dx / (np.max(np.abs(U)) + np.max(np.abs(V)) + 1e-8)
    dt = 0.4 * min(dt_diff, dt_adv)
    n_steps = int(np.ceil(T / dt))
    dt = T / n_steps

    S = source_strength * np.exp(
        -((X - source_xy[0]) ** 2 + (Y - source_xy[1]) ** 2) / (2 * source_sigma ** 2)
    )

    C = np.zeros((nx, ny))
    snap_every = max(1, n_steps // (n_snapshots - 1))
    times, snaps = [0.0], [C.copy()]

    for step in range(1, n_steps + 1):
        dCdx = np.where(U > 0,
                         (C - np.roll(C, 1, axis=0)) / dx,
                         (np.roll(C, -1, axis=0) - C) / dx)
        dCdy = np.where(V > 0,
                         (C - np.roll(C, 1, axis=1)) / dy,
                         (np.roll(C, -1, axis=1) - C) / dy)
        laplacian = (
            (np.roll(C, -1, axis=0) - 2 * C + np.roll(C, 1, axis=0)) / dx ** 2
            + (np.roll(C, -1, axis=1) - 2 * C + np.roll(C, 1, axis=1)) / dy ** 2
        )
        C = C + dt * (-U * dCdx - V * dCdy + D * laplacian + S - K * C)

        # Through-flow boundary conditions (different from the swirl case):
        C[0, :] = 0.0            # inlet: fresh air
        C[-1, :] = C[-2, :]      # outlet: zero-gradient (free exit)
        C[:, 0] = C[:, 1]        # bottom wall: zero-gradient (no flux through wall)
        C[:, -1] = C[:, -2]      # top wall: zero-gradient (no flux through wall)

        if step % snap_every == 0 or step == n_steps:
            times.append(step * dt)
            snaps.append(C.copy())

    snaps = np.stack(snaps, axis=0)
    times = np.array(times)

    if verbose:
        print(f"[forward solve] grid {nx}x{ny}, dt={dt:.5f}, steps={n_steps}, "
              f"snapshots={len(times)}")
        print(f"[forward solve] concentration range: "
              f"[{snaps.min():.4f}, {snaps.max():.4f}] "
              f"(should be finite and bounded, not exploding)")
        if not np.isfinite(snaps).all():
            raise RuntimeError("Forward solve produced NaN/Inf -- solver is unstable, "
                                "reduce dt or check parameters before continuing.")

    return x, y, times, snaps, S


def find_gradient_rich_box(x, y, snaps, threshold_frac=0.05, pad_cells=3):
    """Same idea as before: only claim/test velocity recovery where
    concentration actually varies meaningfully (per the HFM paper's own
    stated requirement)."""
    c_max_over_time = snaps.max(axis=0)
    threshold = threshold_frac * c_max_over_time.max()
    mask = c_max_over_time > threshold
    ii, jj = np.where(mask)
    nx, ny = len(x), len(y)
    i_lo, i_hi = max(0, ii.min() - pad_cells), min(nx - 1, ii.max() + pad_cells)
    j_lo, j_hi = max(0, jj.min() - pad_cells), min(ny - 1, jj.max() + pad_cells)
    return x[i_lo], x[i_hi], y[j_lo], y[j_hi], (i_lo, i_hi, j_lo, j_hi)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--D", type=float, default=0.01, help="diffusion coefficient (placeholder)")
    parser.add_argument("--K", type=float, default=0.2, help="extra removal rate beyond through-flow (placeholder)")
    parser.add_argument("--nu", type=float, default=None,
                         help="viscosity for training's momentum residual. Default: same as --D "
                              "(Re~60, matching the HFM paper's tested range -- see the swirl "
                              "test's docstring for the Reynolds-number reasoning, verified by hand).")
    parser.add_argument("--n-sensors", type=int, default=25)
    parser.add_argument("--n-bc", type=int, default=200)
    parser.add_argument("--bc-sides", type=str, default="left",
                         help="which edges get known true velocity: comma list from "
                              "{left,right,top,bottom} or 'all'. Default 'left' = only the "
                              "window/inlet is known, matching the paper's own benchmark design.")
    parser.add_argument("--n-collocation", type=int, default=20000)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-pde", type=float, default=10.0)
    parser.add_argument("--lambda-aux", type=float, default=10.0)
    parser.add_argument("--lambda-mom", type=float, default=10.0)
    parser.add_argument("--lambda-bc", type=float, default=10.0)
    parser.add_argument("--lbfgs-iters", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="hfm_channel_result.png")
    args = parser.parse_args()

    if args.nu is None:
        args.nu = args.D

    np.random.seed(args.seed)

    x, y, times, snaps, S = solve_forward(nx=args.nx, ny=args.ny, T=args.T, D=args.D, K=args.K)
    nx, ny = len(x), len(y)

    x_lo, x_hi, y_lo, y_hi, (i_lo, i_hi, j_lo, j_hi) = find_gradient_rich_box(x, y, snaps)
    print(f"[domain] gradient-rich box: x in [{x_lo:.3f}, {x_hi:.3f}], "
          f"y in [{y_lo:.3f}, {y_hi:.3f}]")

    interior = [(i, j) for i in range(max(2, i_lo), min(nx - 2, i_hi) + 1)
                for j in range(max(2, j_lo), min(ny - 2, j_hi) + 1)]
    idx = np.random.choice(len(interior), size=min(args.n_sensors, len(interior)), replace=False)
    sensor_ij = [interior[k] for k in idx]
    sensor_xy = np.array([[x[i], y[j]] for i, j in sensor_ij])
    sensor_readings = np.stack([snaps[:, i, j] for i, j in sensor_ij], axis=1)

    print(f"[data] {len(sensor_ij)} fixed sensors x {len(times)} time snapshots = "
          f"{len(sensor_ij) * len(times)} training points for the data loss")

    sides = [s.strip().lower() for s in args.bc_sides.split(",")] if args.bc_sides != "all" \
        else ["left", "right", "top", "bottom"]
    n_side = max(1, args.n_bc // max(1, len(sides)))
    side_x, side_y = [], []
    if "top" in sides:
        side_x.append(np.random.uniform(x_lo, x_hi, n_side)); side_y.append(np.full(n_side, y_hi))
    if "bottom" in sides:
        side_x.append(np.random.uniform(x_lo, x_hi, n_side)); side_y.append(np.full(n_side, y_lo))
    if "left" in sides:
        side_x.append(np.full(n_side, x_lo)); side_y.append(np.random.uniform(y_lo, y_hi, n_side))
    if "right" in sides:
        side_x.append(np.full(n_side, x_hi)); side_y.append(np.random.uniform(y_lo, y_hi, n_side))
    if not side_x:
        raise ValueError(f"--bc-sides '{args.bc_sides}' matched none of left/right/top/bottom")
    bc_x = np.concatenate(side_x)
    bc_y = np.concatenate(side_y)
    bc_t = np.random.uniform(0, args.T, len(bc_x))
    bc_u_true, bc_v_true = true_velocity(bc_x, bc_y)
    print(f"[data] known-velocity sides: {sides}")
    print(f"[data] {len(bc_x)} boundary points with known true velocity given "
          f"(the 'window' inlet, mirroring the paper's own measured-inlet approach)")

    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[train] using device: {device}")

    class HFMNet(nn.Module):
        def __init__(self, hidden, n_layers):
            super().__init__()
            dims = [3] + [hidden] * n_layers + [4]
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.Tanh())
            self.net = nn.Sequential(*layers)

        def forward(self, x, y, t):
            out = self.net(torch.cat([x, y, t], dim=1))
            psi, C, daux, p = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]
            return psi, C, daux, p

    def _d(f, wrt):
        return torch.autograd.grad(f, wrt, grad_outputs=torch.ones_like(f), create_graph=True)[0]

    def derive_fields(model, x, y, t):
        psi, C, daux, p = model(x, y, t)
        psi_x, psi_y = _d(psi, x), _d(psi, y)
        u, v = psi_y, -psi_x

        u_x, u_y, u_t = _d(u, x), _d(u, y), _d(u, t)
        v_x, v_y, v_t = _d(v, x), _d(v, y), _d(v, t)
        u_xx, u_yy = _d(u_x, x), _d(u_y, y)
        v_xx, v_yy = _d(v_x, x), _d(v_y, y)

        p_x, p_y = _d(p, x), _d(p, y)

        C_x, C_y, C_t = _d(C, x), _d(C, y), _d(C, t)
        C_xx, C_yy = _d(C_x, x), _d(C_y, y)

        d_x, d_y, d_t = _d(daux, x), _d(daux, y), _d(daux, t)
        d_xx, d_yy = _d(d_x, x), _d(d_y, y)

        return dict(
            u=u, v=v, C=C, daux=daux, p=p,
            u_x=u_x, u_y=u_y, u_t=u_t, u_xx=u_xx, u_yy=u_yy,
            v_x=v_x, v_y=v_y, v_t=v_t, v_xx=v_xx, v_yy=v_yy,
            p_x=p_x, p_y=p_y,
            C_x=C_x, C_y=C_y, C_t=C_t, C_xx=C_xx, C_yy=C_yy,
            d_x=d_x, d_y=d_y, d_t=d_t, d_xx=d_xx, d_yy=d_yy,
        )

    def source_term(x, y):
        return 5.0 * torch.exp(-((x - 0.2) ** 2 + (y - 0.5) ** 2) / (2 * 0.05 ** 2))

    def physics_losses(model, cx, cy, ct, D, K, nu):
        f = derive_fields(model, cx, cy, ct)
        S_val = source_term(cx, cy)

        transport = (f["C_t"] + f["u"] * f["C_x"] + f["v"] * f["C_y"]
                     - D * (f["C_xx"] + f["C_yy"]) - S_val + K * f["C"])
        aux = (f["d_t"] + f["u"] * f["d_x"] + f["v"] * f["d_y"]
               - D * (f["d_xx"] + f["d_yy"]) + S_val - K + K * f["daux"])
        # NOTE: no manufactured forcing needed here (unlike the swirl test)
        # -- channel/Poiseuille flow is an exact solution of the viscous
        # momentum equations for any nu, by construction (verified by
        # hand). If training pushes far from the true field, that's a
        # genuine result, not an equation mismatch artifact.
        mom_u = (f["u_t"] + f["u"] * f["u_x"] + f["v"] * f["u_y"]
                 + f["p_x"] - nu * (f["u_xx"] + f["u_yy"]))
        mom_v = (f["v_t"] + f["u"] * f["v_x"] + f["v"] * f["v_y"]
                 + f["p_y"] - nu * (f["v_xx"] + f["v_yy"]))

        transport_loss = torch.mean(transport ** 2)
        aux_loss = torch.mean(aux ** 2)
        momentum_loss = torch.mean(mom_u ** 2) + torch.mean(mom_v ** 2)
        return transport_loss, aux_loss, momentum_loss

    model = HFMNet(args.hidden, args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)], gamma=0.1
    )
    mse = nn.MSELoss()

    n_t = len(times)
    n_sensors = len(sensor_ij)
    sx = np.repeat(sensor_xy[:, 0], n_t)
    sy = np.repeat(sensor_xy[:, 1], n_t)
    st = np.tile(times, n_sensors)
    sC = sensor_readings.T.reshape(-1)

    data_x = torch.tensor(sx, dtype=torch.float32, device=device).view(-1, 1)
    data_y = torch.tensor(sy, dtype=torch.float32, device=device).view(-1, 1)
    data_t = torch.tensor(st, dtype=torch.float32, device=device).view(-1, 1)
    data_C = torch.tensor(sC, dtype=torch.float32, device=device).view(-1, 1)
    data_daux = torch.tensor(1.0 - sC, dtype=torch.float32, device=device).view(-1, 1)

    bc_x_t = torch.tensor(bc_x, dtype=torch.float32, device=device).view(-1, 1)
    bc_y_t = torch.tensor(bc_y, dtype=torch.float32, device=device).view(-1, 1)
    bc_t_t = torch.tensor(bc_t, dtype=torch.float32, device=device).view(-1, 1)
    bc_u_t = torch.tensor(bc_u_true, dtype=torch.float32, device=device).view(-1, 1)
    bc_v_t = torch.tensor(bc_v_true, dtype=torch.float32, device=device).view(-1, 1)

    def sample_collocation(n):
        cx = (x_lo + torch.rand(n, 1, device=device) * (x_hi - x_lo)).requires_grad_(True)
        cy = (y_lo + torch.rand(n, 1, device=device) * (y_hi - y_lo)).requires_grad_(True)
        ct = (torch.rand(n, 1, device=device) * args.T).requires_grad_(True)
        return cx, cy, ct

    print(f"[train] training for {args.epochs} epochs, {args.n_collocation} collocation "
          f"points/epoch, lambda_pde={args.lambda_pde}, lambda_aux={args.lambda_aux}, "
          f"lambda_mom={args.lambda_mom}, lambda_bc={args.lambda_bc}")

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        data_fields = derive_fields(
            model,
            data_x.clone().requires_grad_(True),
            data_y.clone().requires_grad_(True),
            data_t.clone().requires_grad_(True),
        )
        data_loss = mse(data_fields["C"], data_C) + mse(data_fields["daux"], data_daux)

        bc_fields = derive_fields(
            model,
            bc_x_t.clone().requires_grad_(True),
            bc_y_t.clone().requires_grad_(True),
            bc_t_t.clone().requires_grad_(True),
        )
        bc_loss = mse(bc_fields["u"], bc_u_t) + mse(bc_fields["v"], bc_v_t)

        cx, cy, ct = sample_collocation(args.n_collocation)
        transport_loss, aux_loss, momentum_loss = physics_losses(model, cx, cy, ct, args.D, args.K, args.nu)

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

    if args.lbfgs_iters > 0:
        print(f"\n[train] stage 2: L-BFGS refinement, up to {args.lbfgs_iters} iterations")
        n_lbfgs_colloc = max(args.n_collocation, 20000)
        lb_cx_base = x_lo + torch.rand(n_lbfgs_colloc, 1, device=device) * (x_hi - x_lo)
        lb_cy_base = y_lo + torch.rand(n_lbfgs_colloc, 1, device=device) * (y_hi - y_lo)
        lb_ct_base = torch.rand(n_lbfgs_colloc, 1, device=device) * args.T

        lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=args.lbfgs_iters,
            history_size=50, tolerance_grad=1e-9, tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )
        call_count = [0]

        def closure():
            lbfgs.zero_grad()
            data_fields = derive_fields(
                model,
                data_x.clone().requires_grad_(True),
                data_y.clone().requires_grad_(True),
                data_t.clone().requires_grad_(True),
            )
            data_loss = mse(data_fields["C"], data_C) + mse(data_fields["daux"], data_daux)

            bc_fields = derive_fields(
                model,
                bc_x_t.clone().requires_grad_(True),
                bc_y_t.clone().requires_grad_(True),
                bc_t_t.clone().requires_grad_(True),
            )
            bc_loss = mse(bc_fields["u"], bc_u_t) + mse(bc_fields["v"], bc_v_t)

            lb_cx = lb_cx_base.clone().requires_grad_(True)
            lb_cy = lb_cy_base.clone().requires_grad_(True)
            lb_ct = lb_ct_base.clone().requires_grad_(True)
            transport_loss, aux_loss, momentum_loss = physics_losses(model, lb_cx, lb_cy, lb_ct, args.D, args.K, args.nu)

            loss = (data_loss + args.lambda_bc * bc_loss
                    + args.lambda_pde * transport_loss + args.lambda_aux * aux_loss
                    + args.lambda_mom * momentum_loss)
            loss.backward()

            call_count[0] += 1
            if call_count[0] % 200 == 0 or call_count[0] == 1:
                print(f"  lbfgs call {call_count[0]:6d}  data={data_loss.item():.6e}  "
                      f"bc={bc_loss.item():.6e}  transport={transport_loss.item():.6e}  "
                      f"aux={aux_loss.item():.6e}  momentum={momentum_loss.item():.6e}  "
                      f"total={loss.item():.6e}")
            return loss

        lbfgs.step(closure)
        print(f"[train] L-BFGS finished after {call_count[0]} closure evaluations")

    model.eval()
    gx_full, gy_full = np.meshgrid(x, y, indexing="ij")
    box_mask = ((gx_full >= x_lo) & (gx_full <= x_hi) & (gy_full >= y_lo) & (gy_full <= y_hi))
    gx, gy = gx_full[box_mask], gy_full[box_mask]
    u_true, v_true = true_velocity(gx, gy)
    speed_true = np.sqrt(u_true ** 2 + v_true ** 2)

    def eval_at(t_val):
        ex = torch.tensor(gx.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
        ey = torch.tensor(gy.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
        et = torch.full_like(ex, t_val, requires_grad=True)
        f = derive_fields(model, ex, ey, et)
        u_pred = f["u"].detach().cpu().numpy().reshape(-1)
        v_pred = f["v"].detach().cpu().numpy().reshape(-1)
        rel_err_u = np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-8)
        rel_err_v = np.linalg.norm(v_pred - v_true) / (np.linalg.norm(v_true) + 1e-8)
        speed_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
        rel_err_speed = np.linalg.norm(speed_pred - speed_true) / (np.linalg.norm(speed_true) + 1e-8)
        return u_pred, v_pred, rel_err_u, rel_err_v, rel_err_speed

    print(f"\n=== RESULT: recovered vs. true velocity, INSIDE THE GRADIENT-RICH BOX ONLY ===")
    results = {}
    for frac in [0.3, 0.5, 0.7, 1.0]:
        t_val = frac * args.T
        u_pred, v_pred, ru, rv, rs = eval_at(t_val)
        results[frac] = (u_pred, v_pred, ru, rv, rs)
        tag = "  (near training-window edge)" if frac == 1.0 else ""
        print(f"  t={t_val:.2f} (frac={frac}): u err={100*ru:.2f}%  v err={100*rv:.2f}%  "
              f"speed err={100*rs:.2f}%{tag}")

    eval_t = 0.5 * args.T
    u_pred, v_pred, rel_err_u, rel_err_v, rel_err_speed = results[0.5]
    print(f"\n  Primary result (mid-time, t={eval_t:.2f}): speed error {100*rel_err_speed:.2f}%")
    print(f"  Boundary condition given on: {sides} (default 'left' = window/inlet only)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    step = 2
    gx2d = gx_full[i_lo:i_hi+1:step, j_lo:j_hi+1:step]
    gy2d = gy_full[i_lo:i_hi+1:step, j_lo:j_hi+1:step]
    u_true2d, v_true2d = true_velocity(gx2d, gy2d)

    ex2 = torch.tensor(gx2d.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
    ey2 = torch.tensor(gy2d.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
    et2 = torch.full_like(ex2, eval_t, requires_grad=True)
    f2 = derive_fields(model, ex2, ey2, et2)
    u_pred2d = f2["u"].detach().cpu().numpy().reshape(gx2d.shape)
    v_pred2d = f2["v"].detach().cpu().numpy().reshape(gx2d.shape)

    axes[0].quiver(gx2d, gy2d, u_true2d, v_true2d)
    axes[0].set_title("True velocity (channel flow, box only)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[1].quiver(gx2d, gy2d, u_pred2d, v_pred2d)
    axes[1].scatter(sensor_xy[:, 0], sensor_xy[:, 1], c="red", s=15, marker="x", label="sensors")
    axes[1].scatter(bc_x, bc_y, c="blue", s=6, marker=".", label="boundary (known velocity)")
    axes[1].set_title(f"Recovered velocity (speed error: {100*rel_err_speed:.1f}%)")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    axes[1].legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved comparison figure to {args.out}")


if __name__ == "__main__":
    main()
