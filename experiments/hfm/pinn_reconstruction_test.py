"""
Small synthetic test of pipeline steps 1-4 from our plan: collect sparse
sensor data, train a physics-informed network to reconstruct the full CO2
field from it, and check the reconstruction against sensors it never saw.

This is a DIFFERENT, much simpler idea than the earlier hfm_synthetic_test.py.
That script tried to recover velocity from concentration (which we found is
badly underdetermined). This script does NOT try to recover velocity at
all -- it only uses the diffusion part of the physics (no advection/velocity
term) to help fill in CO2 values between sparse sensors. The question this
answers: is "diffusion-only" physics a good enough helper for spatial
reconstruction, given that real rooms also have advection (airflow) which
this reconstruction deliberately ignores?

--------------------------------------------------------------------------
TWO BUGS FIXED AFTER THE FIRST RUN (see them in the code as FIX 1 / FIX 2)
--------------------------------------------------------------------------
The first run gave a 154% full-field error, but a lot of that turned out to
be two fixable oversights, not proof the whole idea is broken:
- FIX 1: the network could output negative CO2 (physically impossible) in
  regions with no nearby sensor to constrain it. Fixed by passing the
  output through softplus, so it's architecturally impossible to be
  negative, not just discouraged.
- FIX 2: the network was never told that C=0 everywhere in the room at
  t=0 -- it only saw that at the training sensors' specific spots, and
  didn't generalize it elsewhere (the held-out sensor's reconstruction
  started at -0.12 at t=0, when the true answer is exactly 0 everywhere).
  Fixed by adding an explicit loss term sampling random points across the
  whole room at t=0 and penalizing anything other than 0 there.
This run will show whether fixing these two issues gets the error down to
something reasonable, or whether the remaining error is really about
advection being ignored (the harder, not-yet-fixed limitation).

--------------------------------------------------------------------------
WHAT THIS SCRIPT DOES, IN ORDER (mirrors pipeline steps 1-4)
--------------------------------------------------------------------------
1. "Collect real data": generates one made-up, known CO2 field using the
   SAME advection+diffusion forward solver as before (known swirl velocity
   + a source), then samples it at a handful of fixed sensor locations
   over time. This stands in for real sensor logs -- note the underlying
   truth DOES include advection (airflow), the same way a real room would,
   even though the reconstruction network below is never told about it.
2. Splits those sensors into a training set and a small held-out set. The
   held-out sensors are never used for training -- only for checking the
   result afterward, exactly like you'd do with real data with no CFD to
   compare against.
3. Trains a small network that takes (x, y, t) and outputs CO2 directly
   (no velocity, no stream function -- this is a much simpler network than
   the HFM one). It's trained to (a) match the training sensors' readings,
   and (b) satisfy the plain diffusion equation (no advection term) at
   random points in space and time.
4. Checks the trained network's predictions at the held-out sensors against
   their true readings -- this is the only validation you'd have with real
   data, no CFD. As a bonus (only possible because this is a synthetic
   test where we made up the full answer key), it also compares the
   reconstruction against the TRUE full field everywhere, so we can see
   whether the held-out-sensor check would have caught it if the
   diffusion-only simplification was hiding a bigger, room-wide error.

--------------------------------------------------------------------------
ASSUMPTIONS MADE UP FOR THIS TEST -- NOT REAL PHYSICAL VALUES
--------------------------------------------------------------------------
Same as before: D = 0.01 and K = 0.5 are placeholders (see hfm_synthetic_
test.py for the same disclaimer), and the "true" swirl velocity/source are
invented so we have a known answer to check against. The reconstruction
network is NEVER given D, K, or the velocity as ground truth to imitate --
it only uses the same D, K values in its own physics-residual loss, same
as you'd have to guess/assume with real data.

--------------------------------------------------------------------------
WHAT WAS AND WASN'T TESTED BEFORE HANDING THIS TO YOU
--------------------------------------------------------------------------
- The forward solver (step 1, pure NumPy) reuses the exact function already
  tested for hfm_synthetic_test.py.
- The PyTorch/PINN part (steps 3-4) could NOT be run in my environment --
  still no network access there to install PyTorch. This network is much
  simpler than the HFM one (plain scalar output, only first/second
  derivatives, no stream function, no momentum) so there's less to go
  wrong, but please run it and tell me what happens either way.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python pinn_reconstruction_test.py

Look at two numbers in the output: the held-out sensor error (what you'd
actually get to see with real data) and the full-field error (only visible
here because it's synthetic). If they're both low, diffusion-only
reconstruction works for this case. If the held-out error is low but the
full-field error is high, that's a warning sign: the held-out check alone
would have missed a real problem, and advection matters more than assumed.
"""

import argparse
import numpy as np

# Reuse the exact same, already-tested forward solver and velocity field
# from the HFM script, so the "ground truth" here has the same realistic
# ingredients (a swirl + a point source) -- this truth DOES include
# advection, unlike the reconstruction model below, which is the whole
# point of the comparison.
from hfm_synthetic_test import true_velocity, solve_forward


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--D", type=float, default=0.01, help="diffusion coefficient used in the RECONSTRUCTION's physics loss (placeholder)")
    parser.add_argument("--K", type=float, default=0.5, help="removal/decay rate used in the RECONSTRUCTION's physics loss (placeholder)")
    parser.add_argument("--n-sensors", type=int, default=25, help="total fixed sensors (train + held-out)")
    parser.add_argument("--n-holdout", type=int, default=5, help="how many of those sensors are held out, never trained on")
    parser.add_argument("--n-collocation", type=int, default=8000)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-pde", type=float, default=10.0)
    parser.add_argument("--lambda-ic", type=float, default=10.0,
                         help="weight on the t=0 initial-condition loss (C=0 everywhere at t=0)")
    parser.add_argument("--n-ic", type=int, default=2000,
                         help="random (x,y) points used to enforce the t=0 condition, spread across the whole room, not just sensor locations")
    parser.add_argument("--lbfgs-iters", type=int, default=2000, help="set 0 to skip")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="pinn_reconstruction_result.png")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ---- Step 1: "collect real data" (synthetic here, but plays that role) ----
    x, y, times, snaps, S = solve_forward(nx=args.nx, ny=args.ny, T=args.T, D=args.D, K=args.K)
    nx, ny = len(x), len(y)

    interior = [(i, j) for i in range(2, nx - 2) for j in range(2, ny - 2)]
    idx = np.random.choice(len(interior), size=args.n_sensors, replace=False)
    sensor_ij = [interior[k] for k in idx]
    sensor_xy = np.array([[x[i], y[j]] for i, j in sensor_ij])
    sensor_readings = np.stack([snaps[:, i, j] for i, j in sensor_ij], axis=1)  # (n_t, n_sensors)

    # ---- Step 2: split into training sensors and held-out sensors ----
    perm = np.random.permutation(args.n_sensors)
    holdout_idx = perm[:args.n_holdout]
    train_idx = perm[args.n_holdout:]

    train_xy = sensor_xy[train_idx]
    train_readings = sensor_readings[:, train_idx]
    holdout_xy = sensor_xy[holdout_idx]
    holdout_readings = sensor_readings[:, holdout_idx]

    print(f"[data] {args.n_sensors} total sensors: {len(train_idx)} for training, "
          f"{len(holdout_idx)} held out for checking only")

    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[train] using device: {device}")

    class ReconstructionNet(nn.Module):
        """(x, y, t) -> C. No velocity, no stream function -- this network
        never represents or is told about airflow at all.

        FIX 1: the raw network output is passed through softplus, which is
        always >= 0. This makes it architecturally impossible for the
        network to output negative CO2 -- not just discouraged by a
        penalty, actually impossible -- which is what the negative blob
        in the first result was violating."""

        def __init__(self, hidden, n_layers):
            super().__init__()
            dims = [3] + [hidden] * n_layers + [1]
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.Tanh())
            self.net = nn.Sequential(*layers)
            self.softplus = nn.Softplus()

        def forward(self, x, y, t):
            raw = self.net(torch.cat([x, y, t], dim=1))
            return self.softplus(raw)

    def _d(f, wrt):
        return torch.autograd.grad(f, wrt, grad_outputs=torch.ones_like(f), create_graph=True)[0]

    def source_term(x, y):
        return 5.0 * torch.exp(-((x - 0.3) ** 2 + (y - 0.5) ** 2) / (2 * 0.05 ** 2))

    def diffusion_residual(model, x, y, t, D, K):
        """Plain diffusion equation, no advection term:
        C_t - D*(C_xx + C_yy) - S(x,y) + K*C = 0
        This is deliberately missing the u*C_x + v*C_y advection term --
        that's the simplification being tested."""
        C = model(x, y, t)
        C_x, C_y, C_t = _d(C, x), _d(C, y), _d(C, t)
        C_xx, C_yy = _d(C_x, x), _d(C_y, y)
        residual = C_t - D * (C_xx + C_yy) - source_term(x, y) + K * C
        return torch.mean(residual ** 2)

    model = ReconstructionNet(args.hidden, args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    # ---- training-sensor tensors ----
    n_t = len(times)
    n_train = len(train_idx)
    sx = np.repeat(train_xy[:, 0], n_t)
    sy = np.repeat(train_xy[:, 1], n_t)
    st = np.tile(times, n_train)
    sC = train_readings.T.reshape(-1)

    data_x = torch.tensor(sx, dtype=torch.float32, device=device).view(-1, 1)
    data_y = torch.tensor(sy, dtype=torch.float32, device=device).view(-1, 1)
    data_t = torch.tensor(st, dtype=torch.float32, device=device).view(-1, 1)
    data_C = torch.tensor(sC, dtype=torch.float32, device=device).view(-1, 1)

    print(f"[train] training for {args.epochs} epochs, "
          f"{args.n_collocation} collocation points/epoch, "
          f"lambda_pde={args.lambda_pde}, lambda_ic={args.lambda_ic}")

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        C_pred = model(data_x, data_y, data_t)
        data_loss = mse(C_pred, data_C)

        cx = torch.rand(args.n_collocation, 1, device=device, requires_grad=True)
        cy = torch.rand(args.n_collocation, 1, device=device, requires_grad=True)
        ct = torch.rand(args.n_collocation, 1, device=device, requires_grad=True) * args.T
        pde_loss = diffusion_residual(model, cx, cy, ct, args.D, args.K)

        # FIX 2: explicitly teach the network that C=0 everywhere in the
        # room at t=0, not just at the training sensors' specific spots.
        # This is what the held-out sensor was violating (it predicted
        # -0.12 at t=0 there, despite never having been shown that
        # location's initial value).
        ic_x = torch.rand(args.n_ic, 1, device=device)
        ic_y = torch.rand(args.n_ic, 1, device=device)
        ic_t = torch.zeros(args.n_ic, 1, device=device)
        ic_loss = torch.mean(model(ic_x, ic_y, ic_t) ** 2)

        loss = data_loss + args.lambda_pde * pde_loss + args.lambda_ic * ic_loss
        loss.backward()
        optimizer.step()

        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(f"  epoch {epoch:6d}  data_loss={data_loss.item():.6e}  "
                  f"pde_loss={pde_loss.item():.6e}  ic_loss={ic_loss.item():.6e}  "
                  f"total={loss.item():.6e}")

    if args.lbfgs_iters > 0:
        print(f"\n[train] stage 2: L-BFGS refinement, up to {args.lbfgs_iters} iterations")
        n_lbfgs_colloc = max(args.n_collocation, 20000)
        lb_cx_base = torch.rand(n_lbfgs_colloc, 1, device=device)
        lb_cy_base = torch.rand(n_lbfgs_colloc, 1, device=device)
        lb_ct_base = torch.rand(n_lbfgs_colloc, 1, device=device) * args.T
        # IC points don't need requires_grad (no derivative taken w.r.t.
        # them, just a plain forward pass + MSE against 0), so reusing
        # this fixed batch across closure calls is safe.
        lb_ic_x = torch.rand(args.n_ic, 1, device=device)
        lb_ic_y = torch.rand(args.n_ic, 1, device=device)
        lb_ic_t = torch.zeros(args.n_ic, 1, device=device)

        lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=args.lbfgs_iters,
            history_size=50, tolerance_grad=1e-9, tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )
        call_count = [0]

        def closure():
            lbfgs.zero_grad()
            C_pred = model(data_x, data_y, data_t)
            data_loss = mse(C_pred, data_C)

            # fresh leaves every call -- reusing persistent requires_grad
            # tensors across L-BFGS's repeated closure calls previously
            # caused a "backward through the graph a second time" crash
            # in the HFM script; cloning fresh each call avoids it.
            lb_cx = lb_cx_base.clone().requires_grad_(True)
            lb_cy = lb_cy_base.clone().requires_grad_(True)
            lb_ct = lb_ct_base.clone().requires_grad_(True)
            pde_loss = diffusion_residual(model, lb_cx, lb_cy, lb_ct, args.D, args.K)

            ic_loss = torch.mean(model(lb_ic_x, lb_ic_y, lb_ic_t) ** 2)

            loss = data_loss + args.lambda_pde * pde_loss + args.lambda_ic * ic_loss
            loss.backward()

            call_count[0] += 1
            if call_count[0] % 200 == 0 or call_count[0] == 1:
                print(f"  lbfgs call {call_count[0]:6d}  data_loss={data_loss.item():.6e}  "
                      f"pde_loss={pde_loss.item():.6e}  ic_loss={ic_loss.item():.6e}  "
                      f"total={loss.item():.6e}")
            return loss

        lbfgs.step(closure)
        print(f"[train] L-BFGS finished after {call_count[0]} closure evaluations")

    # ---- Step 4a: check against held-out sensors (the real-world-available check) ----
    model.eval()
    n_holdout = len(holdout_idx)
    hx = np.repeat(holdout_xy[:, 0], n_t)
    hy = np.repeat(holdout_xy[:, 1], n_t)
    ht = np.tile(times, n_holdout)
    h_true = holdout_readings.T.reshape(-1)

    hx_t = torch.tensor(hx, dtype=torch.float32, device=device).view(-1, 1)
    hy_t = torch.tensor(hy, dtype=torch.float32, device=device).view(-1, 1)
    ht_t = torch.tensor(ht, dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        h_pred = model(hx_t, hy_t, ht_t).cpu().numpy().reshape(-1)

    holdout_rel_err = np.linalg.norm(h_pred - h_true) / (np.linalg.norm(h_true) + 1e-8)

    # DIAGNOSTIC: is the held-out error being inflated by early, near-zero
    # timesteps (relative error blows up when the true value is tiny)?
    # Reshape back to (n_holdout, n_t) to check per-timestep, and also
    # compute the SAME metric restricted to the final timestep only, so
    # it's directly comparable to the full-field error below (which is
    # only ever measured at t=T).
    h_pred_2d = h_pred.reshape(n_holdout, n_t)
    h_true_2d = h_true.reshape(n_holdout, n_t)
    per_t_true_norm = np.linalg.norm(h_true_2d, axis=0)
    per_t_err_norm = np.linalg.norm(h_pred_2d - h_true_2d, axis=0)
    per_t_rel_err = per_t_err_norm / (per_t_true_norm + 1e-8)
    holdout_rel_err_finalT = per_t_rel_err[-1]
    print(f"[diagnostic] held-out relative error by timestep (first 5): "
          f"{np.round(100*per_t_rel_err[:5], 1)} %")
    print(f"[diagnostic] held-out relative error by timestep (last 5):  "
          f"{np.round(100*per_t_rel_err[-5:], 1)} %")
    print(f"[diagnostic] held-out error at FINAL timestep only (comparable to full-field metric): "
          f"{100*holdout_rel_err_finalT:.2f}%")

    # ---- Step 4b: check against the FULL true field (only possible because this is synthetic) ----
    gx, gy = np.meshgrid(x, y, indexing="ij")
    eval_t = args.T
    ex = torch.tensor(gx.reshape(-1, 1), dtype=torch.float32, device=device)
    ey = torch.tensor(gy.reshape(-1, 1), dtype=torch.float32, device=device)
    et = torch.full_like(ex, eval_t)
    with torch.no_grad():
        C_field_pred = model(ex, ey, et).cpu().numpy().reshape(nx, ny)
    C_field_true = snaps[-1]  # last saved snapshot, at t = T

    full_field_rel_err = np.linalg.norm(C_field_pred - C_field_true) / (np.linalg.norm(C_field_true) + 1e-8)

    print("\n=== RESULT ===")
    print(f"  held-out sensor error (the check you'd actually have with real data): "
          f"{100*holdout_rel_err:.2f}%")
    print(f"  full-field error vs. true CO2 field (only visible in this synthetic test): "
          f"{100*full_field_rel_err:.2f}%")
    print("  If both are low: diffusion-only reconstruction is a reasonable stand-in for")
    print("  this room, advection isn't hurting much. If held-out error is low but")
    print("  full-field error is high: the held-out check alone would have missed a real")
    print("  problem, and advection matters more than this simplification assumes.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    im0 = axes[0].imshow(C_field_true.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
    axes[0].set_title("True CO2 field (never fully seen in reality)")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(C_field_pred.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
    axes[1].scatter(train_xy[:, 0], train_xy[:, 1], c="white", s=15, marker="o", label="train sensors")
    axes[1].scatter(holdout_xy[:, 0], holdout_xy[:, 1], c="red", s=25, marker="x", label="held-out sensors")
    axes[1].set_title(f"Diffusion-only reconstruction (full-field err: {100*full_field_rel_err:.1f}%)")
    axes[1].legend(loc="upper right", fontsize=7)
    plt.colorbar(im1, ax=axes[1])

    # time series at one held-out sensor, predicted vs. true
    axes[2].plot(times, holdout_readings[:, 0], "o-", label="true (held-out sensor)")
    hx0 = np.full(n_t, holdout_xy[0, 0])
    hy0 = np.full(n_t, holdout_xy[0, 1])
    hx0_t = torch.tensor(hx0, dtype=torch.float32, device=device).view(-1, 1)
    hy0_t = torch.tensor(hy0, dtype=torch.float32, device=device).view(-1, 1)
    ht0_t = torch.tensor(times, dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        h0_pred = model(hx0_t, hy0_t, ht0_t).cpu().numpy().reshape(-1)
    axes[2].plot(times, h0_pred, "s--", label="reconstructed")
    axes[2].set_title("Held-out sensor: true vs. reconstructed")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("CO2")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved figure to {args.out}")


if __name__ == "__main__":
    main()
