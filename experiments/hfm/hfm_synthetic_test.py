"""
Stage 1 of the Hidden-Fluid-Mechanics (HFM) feasibility test -- v3.

Question this answers: if a network only ever sees CO2 concentration
readings (never velocity), can it recover the underlying airflow velocity
field just from those readings plus the known physics?

Nothing here is real: the "room", the flow, and the CO2 field are all made
up so we know the true answer and can check the network against it. If this
doesn't work on an easy made-up case, it has no chance on real sensor data,
so this is the cheapest possible way to find out whether the whole HFM idea
is worth pursuing further.

--------------------------------------------------------------------------
WHAT WE LEARNED FROM v1 AND v2, AND WHAT CHANGED HERE (v3)
--------------------------------------------------------------------------
v1 (transport equation only) gave ~80-450% velocity error -- a genuine
non-identifiability (the "aperture problem": transport only constrains
velocity ALONG the concentration gradient, not across it).

v2 added the full incompressible Navier-Stokes momentum equations (like the
real HFM paper), hoping the extra physics would remove that freedom.
Result: WORSE (270-840% error), not better.

Before concluding the whole idea is dead, I went back and read the actual
2018 HFM paper in full (Raissi, Yazdani, Karniadakis -- the one this whole
approach is based on) to check what we might have missed. Found three
concrete things v2 was missing that the paper explicitly says matter:

1. THE BIG ONE. The paper states plainly: "there must exist enough
   concentration gradient normal to the boundaries... in order for our
   method to be able to infer a unique solution for the velocity field."
   v1 and v2 both scattered sensors and collocation points UNIFORMLY across
   the whole unit square, including large areas far from the source where
   concentration is near zero and barely changing. The paper never claims
   to recover velocity in a signal-free region -- and neither should we.
   FIX: restrict sensors, collocation points, and the final evaluation to
   a "gradient-rich" box around the source, found automatically from where
   concentration actually rises above a small threshold at some point in
   time. We only claim (and only test) velocity recovery inside that box.

2. Where the paper's own concentration signal is too weak (e.g. upstream of
   their cylinder), they explicitly supply a known velocity boundary
   condition there, rather than expecting the transport equation alone to
   figure it out. FIX: we now feed the network the TRUE velocity (known
   because this is a synthetic test) at points on the edge of the
   restricted box, as an extra data term -- mirroring what the paper does
   with a real, physically-measured inlet velocity.

3. The paper trains an auxiliary "complement" field d = 1 - c alongside c
   itself, with its own transport equation, and reports it "improves the
   accuracy of predictions... at practically no additional cost." FIX:
   added as a second network output with its own data and physics loss.
   (Caveat from a second review: the paper's own stated reason for d is
   helping the network detect an UNKNOWN geometry -- ours is a known
   square, so this one is a lower-confidence fix, included for
   completeness/fidelity to the paper's recipe, not expected to be the
   main lever.)

A second review of this plan (a different AI, asked to check it) confirmed
all three above and found more gaps, independently verified before being
applied here:

4. Our nu (1.5e-5, real air) combined with our made-up velocity/length
   scales gives Reynolds number ~42,000 -- deeply turbulent. The paper
   only tested Re 60-185 and explicitly says it avoids turbulent regimes
   ("the optimizer may fail to converge"). FIX: --nu now defaults to
   equal --D (0.01), giving Re~63 -- verified by hand, inside the paper's
   tested range. This is a choice of which regime to test, not a claim
   about real air.
5. Consequence of fix 4: with nu no longer negligible, our made-up swirl
   (only an exact solution when viscosity ~0) would need to decay over
   time -- but we hold it constant, so training would penalize the TRUE
   answer for not satisfying the equations. FIX: added a manufactured
   forcing term (derived and numerically verified by hand: F =
   2*nu*pi^2*true_velocity) so the true field exactly solves the forced
   equations, same idea as a fan/pressure difference sustaining real
   airflow against friction.
6. The paper reports its worst errors right at the start/end of its
   training time window (least data there). We were only ever scoring at
   t=T, exactly that edge. FIX: now scores at several times (0.3T, 0.5T,
   0.7T, T) and treats 0.5T as the primary number.
7. L-BFGS is OFF by default now (the paper only uses Adam with a decaying
   learning rate: 1e-3 -> 1e-4 -> 1e-5); we added that decay schedule.
   Collocation points per epoch raised from 8000 to 20000 (still far
   below the paper's millions, but a cheap partial step in that direction
   given our compute budget).

Two suggestions from that second review were NOT applied, with reasons:
- Deeper network with sin activation (matching the paper's architecture
  exactly): the paper itself calls this an unconfirmed conjecture ("should
  be interpreted as conjectures rather than firm results"), and plain deep
  sin networks are known in the wider literature to be hard to train
  without a specific weight initialization scheme the 2018 paper doesn't
  describe. Changing this now, untested, on top of everything else risked
  introducing a new failure mode we couldn't tell apart from the others.
  Kept tanh, which we've already seen train stably.
- "No-penetration" walls (stream function = 0) at the room's physical
  edges: correct fact about our made-up swirl, but doesn't apply to the
  gradient-rich BOX from fix 1, since that box's edges are an internal
  cutoff we chose, not the room's real walls. Would only matter if
  evaluation extended out to x,y = 0 or 1, which fix 1 deliberately avoids.

None of this is guaranteed to work -- it's a more faithful reproduction of
the paper's actual recipe, tested honestly, not a promise.

--------------------------------------------------------------------------
WHAT THIS SCRIPT DOES, IN ORDER
--------------------------------------------------------------------------
1. Makes up a known 2D velocity field (a swirl) and a CO2 source, then
   solves the advection-diffusion equation forward to get a ground-truth
   concentration field C(x,y,t). Same as before.
2. Finds the "gradient-rich" region automatically (where C rises above 5%
   of its peak value at some point in time) -- this is the only region
   where the rest of the script operates.
3. Places sensors only inside that region, and also samples a handful of
   points on the region's boundary where we tell the network the TRUE
   velocity (the one piece of "cheating" that mirrors the paper's own use
   of a known inlet condition -- justified because the paper does the same
   when its own signal is insufficient).
4. Trains a network outputting a stream function psi, concentration C, its
   complement d = 1-C, and pressure p. Velocity comes from psi via
   autograd (divergence-free by construction). Trained on: sensor data,
   the boundary velocity data, the transport equation for C, the same
   transport equation for d, and the full incompressible Navier-Stokes
   momentum equations for (u, v, p).
5. Compares the recovered velocity to the true one, but ONLY inside the
   gradient-rich region -- consistent with what the method can actually
   claim.

--------------------------------------------------------------------------
ASSUMPTIONS MADE UP FOR THIS TEST -- NOT ALL ARE REAL PHYSICAL VALUES
--------------------------------------------------------------------------
- D = 0.01 and K = 0.5: still placeholders (see earlier chat discussion,
  still unresolved for the real room).
- nu = 1.5e-5 m^2/s: real air kinematic viscosity, not invented.
- U0 = 0.2, source location/strength: invented to make an easy test case.
- The boundary velocity values fed to the network in step 3 are "cheating"
  in the sense that a real deployment wouldn't know them exactly -- this
  mirrors the paper's own use of a measured inlet condition, and the point
  of this test is to see if the method can work AT ALL under
  favorable-but-honest conditions before worrying about where real
  boundary velocity data would come from.

--------------------------------------------------------------------------
WHAT WAS AND WASN'T TESTED BEFORE HANDING THIS TO YOU
--------------------------------------------------------------------------
- The forward solver (pure NumPy) was run and confirmed stable.
- The PyTorch/PINN part could NOT be run in my environment (no network
  access to install PyTorch there). Derivative chains checked by hand.
  Please run it and report exactly what happens.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python hfm_synthetic_test.py
"""

import argparse
import numpy as np


# ---------------------------------------------------------------------
# The made-up "true" velocity field (single recirculating swirl, zero
# velocity at the walls of the unit square, divergence-free by
# construction because it comes from a stream function).
# ---------------------------------------------------------------------
U0 = 0.2  # made-up velocity scale


def true_velocity(x, y):
    """x, y: numpy arrays, any shape, values in [0, 1]. Returns (u, v)."""
    u = U0 * np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    v = -U0 * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    return u, v


# ---------------------------------------------------------------------
# Forward finite-difference solve of the advection-diffusion equation
# with the known velocity field above, to build the synthetic "ground
# truth" concentration field. Pure NumPy -- this part was tested.
# ---------------------------------------------------------------------
def solve_forward(nx=64, ny=64, T=1.0, D=0.01, K=0.5,
                   source_xy=(0.3, 0.5), source_sigma=0.05, source_strength=5.0,
                   n_snapshots=21, verbose=True):
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
        C[0, :] = C[-1, :] = C[:, 0] = C[:, -1] = 0.0

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
    """FIX 1 from the HFM paper: find the region where concentration
    actually varies (rises above threshold_frac of its peak at some point
    in time), pad it a little, and return that box. Everywhere outside
    this box has too little signal for velocity to be identifiable, per
    the paper's own stated requirement."""
    c_max_over_time = snaps.max(axis=0)  # (nx, ny)
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
    parser.add_argument("--K", type=float, default=0.5, help="removal/decay rate (placeholder)")
    parser.add_argument("--nu", type=float, default=None,
                         help="kinematic viscosity used in training's momentum residual. "
                              "Default: same value as --D. Reasoning (verified by hand, see chat): "
                              "real air viscosity (1.5e-5) combined with our made-up velocity/length "
                              "scales gives Reynolds number ~42,000 -- deeply turbulent, nothing like "
                              "what the HFM paper actually tested (Re 60-185). Using nu=D instead "
                              "gives Re~63, inside the paper's validated range. This is a choice about "
                              "which regime to test in, not a claim about real air.")
    parser.add_argument("--n-sensors", type=int, default=25)
    parser.add_argument("--n-bc", type=int, default=200, help="boundary points (known true velocity given here, per the paper's own approach)")
    parser.add_argument("--bc-sides", type=str, default="left",
                         help="which edges of the box get known true velocity, comma-separated "
                              "from {left,right,top,bottom}, or 'all'. Default 'left' matches the "
                              "paper's own cylinder benchmark, which only supplies velocity at ONE "
                              "inlet edge and infers the rest from concentration + physics -- a "
                              "harder, more meaningful test than giving all four sides.")
    parser.add_argument("--n-collocation", type=int, default=20000, help="raised from 8000: the paper uses far denser collocation (millions of points); this is a cheap partial step toward that")
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3, help="initial LR; decayed 1e-3->1e-4->1e-5 over training, matching the paper's schedule")
    parser.add_argument("--lambda-pde", type=float, default=10.0, help="weight on the C transport-residual loss")
    parser.add_argument("--lambda-aux", type=float, default=10.0, help="weight on the auxiliary d=1-C transport-residual loss")
    parser.add_argument("--lambda-mom", type=float, default=10.0, help="weight on the momentum-residual loss")
    parser.add_argument("--lambda-bc", type=float, default=10.0, help="weight on the boundary-velocity data loss")
    parser.add_argument("--lbfgs-iters", type=int, default=0,
                         help="default OFF now: L-BFGS made results worse in both earlier tests "
                              "(v1: 77%%->450%% error, v2: even higher). The paper itself only uses "
                              "Adam. Set >0 to re-enable if you want to experiment.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="hfm_synthetic_result.png")
    args = parser.parse_args()

    if args.nu is None:
        args.nu = args.D

    np.random.seed(args.seed)

    x, y, times, snaps, S = solve_forward(nx=args.nx, ny=args.ny, T=args.T, D=args.D, K=args.K)
    nx, ny = len(x), len(y)

    # ---- FIX 1: restrict everything to the gradient-rich box ----
    x_lo, x_hi, y_lo, y_hi, (i_lo, i_hi, j_lo, j_hi) = find_gradient_rich_box(x, y, snaps)
    print(f"[domain] gradient-rich box: x in [{x_lo:.3f}, {x_hi:.3f}], "
          f"y in [{y_lo:.3f}, {y_hi:.3f}] -- everything below (sensors, "
          f"collocation points, evaluation) is restricted to this region. "
          f"Outside it, concentration barely changes and velocity isn't "
          f"expected to be recoverable (per the HFM paper).")

    # ---- sensors, restricted to the box ----
    interior = [(i, j) for i in range(max(2, i_lo), min(nx - 2, i_hi) + 1)
                for j in range(max(2, j_lo), min(ny - 2, j_hi) + 1)]
    idx = np.random.choice(len(interior), size=min(args.n_sensors, len(interior)), replace=False)
    sensor_ij = [interior[k] for k in idx]
    sensor_xy = np.array([[x[i], y[j]] for i, j in sensor_ij])
    sensor_readings = np.stack([snaps[:, i, j] for i, j in sensor_ij], axis=1)

    print(f"[data] {len(sensor_ij)} fixed sensors x {len(times)} time snapshots = "
          f"{len(sensor_ij) * len(times)} training points for the data loss")

    # ---- FIX 2: boundary points on the edge of the box, with TRUE velocity given ----
    # (this is the deliberate "cheat" mirroring the paper's known inlet velocity;
    # see the docstring for why this is a fair thing to test first)
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
          f"(mirrors the paper's own use of a measured inlet condition)")

    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[train] using device: {device}")

    class HFMNet(nn.Module):
        """(x, y, t) -> (psi, C, d, p). d = 1-C is the FIX-3 auxiliary
        "complement" variable from the paper -- a separate output, tied to
        C only through matching data targets and its own physics equation,
        not architecturally forced to equal 1-C. Velocity comes from psi
        via autograd (divergence-free by construction)."""

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
        return 5.0 * torch.exp(-((x - 0.3) ** 2 + (y - 0.5) ** 2) / (2 * 0.05 ** 2))

    def true_velocity_torch(x, y):
        """Same formula as the NumPy true_velocity(), for use inside the
        manufactured-forcing term below (needs torch ops to stay in the
        autograd graph's dtype/device, though no gradient is taken through
        it -- it's a known, fixed function evaluated at collocation points)."""
        u = U0 * np.pi * torch.sin(np.pi * x) * torch.cos(np.pi * y)
        v = -U0 * np.pi * torch.cos(np.pi * x) * torch.sin(np.pi * y)
        return u, v

    def manufactured_forcing(x, y, nu):
        """FIX from the second review: our made-up swirl is only an exact,
        unforced solution of the momentum equations when viscosity is
        ~zero (it's a steady solution of the INVISCID Euler equations,
        verified by hand: vorticity = 2*pi^2*psi everywhere). Once nu is
        no longer negligible (as it now isn't, since nu=D by default), the
        true swirl would actually decay over time unless something keeps
        pushing it -- like a fan or pressure difference does in a real
        room. Without accounting for that, the network's training would
        penalize the TRUE answer for not satisfying the equations we hand
        it, biasing training away from the right solution.
        Fix: add a known forcing term so the true field exactly solves the
        forced momentum equations. Derived and numerically verified by
        hand (see chat): for this specific swirl, the needed forcing is
        simply F = 2*nu*pi^2*true_velocity(x,y) -- i.e. exactly enough to
        cancel the viscous dissipation of this one eigenmode."""
        u_true, v_true = true_velocity_torch(x, y)
        Fu = 2.0 * nu * (np.pi ** 2) * u_true
        Fv = 2.0 * nu * (np.pi ** 2) * v_true
        return Fu, Fv

    def physics_losses(model, cx, cy, ct, D, K, nu):
        f = derive_fields(model, cx, cy, ct)
        S_val = source_term(cx, cy)
        Fu, Fv = manufactured_forcing(cx, cy, nu)

        transport = (f["C_t"] + f["u"] * f["C_x"] + f["v"] * f["C_y"]
                     - D * (f["C_xx"] + f["C_yy"]) - S_val + K * f["C"])
        # Auxiliary d = 1-C transport equation, derived by substituting
        # C = 1-d into the transport residual above (see docstring history
        # for the derivation): d_t + u d_x + v d_y - D(d_xx+d_yy) + S - K + K d = 0
        aux = (f["d_t"] + f["u"] * f["d_x"] + f["v"] * f["d_y"]
               - D * (f["d_xx"] + f["d_yy"]) + S_val - K + K * f["daux"])
        mom_u = (f["u_t"] + f["u"] * f["u_x"] + f["v"] * f["u_y"]
                 + f["p_x"] - nu * (f["u_xx"] + f["u_yy"]) - Fu)
        mom_v = (f["v_t"] + f["u"] * f["v_x"] + f["v"] * f["v_y"]
                 + f["p_y"] - nu * (f["v_xx"] + f["v_yy"]) - Fv)

        transport_loss = torch.mean(transport ** 2)
        aux_loss = torch.mean(aux ** 2)
        momentum_loss = torch.mean(mom_u ** 2) + torch.mean(mom_v ** 2)
        return transport_loss, aux_loss, momentum_loss

    model = HFMNet(args.hidden, args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # LR schedule matching the paper's recipe (1e-3 -> 1e-4 -> 1e-5 over
    # training, in thirds) instead of our previous constant LR.
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)], gamma=0.1
    )
    mse = nn.MSELoss()

    # ---- sensor data tensors ----
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

    # ---- boundary-velocity data tensors ----
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

    print(f"[train] training for {args.epochs} epochs, {args.n_collocation} "
          f"collocation points/epoch (sampled only within the gradient-rich box), "
          f"lambda_pde={args.lambda_pde}, lambda_aux={args.lambda_aux}, "
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

            # fresh leaves every call -- see earlier note: reusing a
            # persistent requires_grad tensor across L-BFGS's repeated
            # closure calls caused a "backward through the graph a second
            # time" crash previously.
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

    # ---- evaluate ONLY inside the gradient-rich box ----
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

    # FIX from the second review: the paper itself reports its WORST errors
    # right at the start/end of the training time window (least data
    # there). We were evaluating only at t=T, exactly the edge most prone
    # to that effect. Now checking multiple times, with a mid-time as the
    # primary/plotted number, not the edge.
    print(f"\n=== RESULT: recovered vs. true velocity, INSIDE THE GRADIENT-RICH BOX ONLY ===")
    results = {}
    for frac in [0.3, 0.5, 0.7, 1.0]:
        t_val = frac * args.T
        u_pred, v_pred, ru, rv, rs = eval_at(t_val)
        results[frac] = (u_pred, v_pred, ru, rv, rs)
        tag = "  (near training-window edge, expect this to be the least reliable)" if frac == 1.0 else ""
        print(f"  t={t_val:.2f} (frac={frac}): u err={100*ru:.2f}%  v err={100*rv:.2f}%  "
              f"speed err={100*rs:.2f}%{tag}")

    eval_t = 0.5 * args.T  # mid-time: the primary number and the one plotted
    u_pred, v_pred, rel_err_u, rel_err_v, rel_err_speed = results[0.5]
    print(f"\n  Primary result (mid-time, t={eval_t:.2f}): speed error {100*rel_err_speed:.2f}%")
    print("  This is now a fair test matching what the HFM paper actually claims:")
    print("  velocity recovery only inside the region with real concentration signal,")
    print("  with a known boundary velocity supplied at the edge (like the paper's inlet),")
    print("  scored away from the start/end of the time window.")

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
    axes[0].set_title("True velocity (box only)")
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
