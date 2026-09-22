"""
3D version of pino_parametric_test.py, plus presentation-ready plots.

--------------------------------------------------------------------------
WHAT'S NEW COMPARED TO pino_parametric_test.py (the 2D version)
--------------------------------------------------------------------------
1. Same idea (ONE network trained across a RANGE of window velocities V,
   using only physics + boundary conditions, no real CO2 data), but now
   in 3D, using the vector-potential/curl trick already validated in
   hfm_channel_flow_3d_test.py to guarantee divergence-free velocity.
2. The trained model is now saved to disk (pino_parametric_3d_model.pth),
   so it doesn't need to be retrained just to make more plots later.
3. Presentation plots, meant to be shown as-is to your professor:
   - Predicted vs. TRUE flow, side by side, at three V values: the low
     edge of the trained range, the middle, and a value ABOVE the trained
     range (extrapolation) -- makes the generalization claim visual, not
     just a table of numbers.
   - A separate error-vs-V line chart, with the trained range shaded, so
     it's immediately visible where the model is reliable vs. guessing.

--------------------------------------------------------------------------
THE TEST SCENARIO (same as the 2D version, extended to 3D)
--------------------------------------------------------------------------
    u(x, y, z; V) = V * 4*z*(1-z),   v = 0,   w = 0
Exact solution of the full viscous Navier-Stokes equations for ANY V (same
derivation already verified in hfm_channel_flow_3d_test.py and the 2D
parametric test -- linear in V, so exact for any V, not just the ones
tested).

--------------------------------------------------------------------------
WHAT WAS AND WASN'T TESTED BY ME BEFORE HANDING THIS OVER
--------------------------------------------------------------------------
Same situation as every other script here: the pure-NumPy pieces (true
velocity, exact-solution check) can be and were verified in my sandbox.
The PyTorch training loop could not be run in my environment. Please run
it and report exactly what happens.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python pino_parametric_3d_test.py
    python pino_parametric_3d_test.py --v-min 0.2 --v-max 1.2
"""

import argparse
import numpy as np


def true_velocity(x, y, z, V):
    u = V * 4.0 * z * (1.0 - z)
    zero = np.zeros_like(u) if hasattr(u, "shape") else 0.0
    return u, zero, zero


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v-min", type=float, default=0.2)
    parser.add_argument("--v-max", type=float, default=1.2)
    parser.add_argument("--D", type=float, default=0.01)
    parser.add_argument("--K", type=float, default=0.2)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--n-collocation", type=int, default=20000)
    parser.add_argument("--n-bc", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-mom", type=float, default=10.0)
    parser.add_argument("--lambda-transport", type=float, default=10.0)
    parser.add_argument("--lambda-bc", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-out", type=str, default="pino_parametric_3d_model.pth")
    parser.add_argument("--fig-out", type=str, default="pino_parametric_3d_result.png")
    parser.add_argument("--error-fig-out", type=str, default="pino_parametric_3d_error_vs_V.png")
    args = parser.parse_args()

    np.random.seed(args.seed)
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = 0.0, 1.0, 0.0, 1.0, 0.0, 1.0

    print(f"[setup] training a single 3D operator over V in [{args.v_min}, {args.v_max}]")

    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[train] using device: {device}")

    class ParametricOperator3D(nn.Module):
        """(x, y, z, V) -> (A1, A2, A3, C, p). Steady-state parametric
        operator, vector-potential trick for divergence-free velocity."""
        def __init__(self, hidden, n_layers):
            super().__init__()
            dims = [4] + [hidden] * n_layers + [5]
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.Tanh())
            self.net = nn.Sequential(*layers)

        def forward(self, x, y, z, V):
            out = self.net(torch.cat([x, y, z, V], dim=1))
            A1, A2, A3, C, p = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4], out[:, 4:5]
            return A1, A2, A3, C, p

    def _d(f, wrt):
        return torch.autograd.grad(f, wrt, grad_outputs=torch.ones_like(f), create_graph=True)[0]

    def derive_fields(model, x, y, z, V):
        A1, A2, A3, C, p = model(x, y, z, V)

        A1_y, A1_z = _d(A1, y), _d(A1, z)
        A2_x, A2_z = _d(A2, x), _d(A2, z)
        A3_x, A3_y = _d(A3, x), _d(A3, y)

        u = A3_y - A2_z
        v = A1_z - A3_x
        w = A2_x - A1_y

        u_x, u_y, u_z = _d(u, x), _d(u, y), _d(u, z)
        v_x, v_y, v_z = _d(v, x), _d(v, y), _d(v, z)
        w_x, w_y, w_z = _d(w, x), _d(w, y), _d(w, z)
        u_xx, u_yy, u_zz = _d(u_x, x), _d(u_y, y), _d(u_z, z)
        v_xx, v_yy, v_zz = _d(v_x, x), _d(v_y, y), _d(v_z, z)
        w_xx, w_yy, w_zz = _d(w_x, x), _d(w_y, y), _d(w_z, z)

        p_x, p_y, p_z = _d(p, x), _d(p, y), _d(p, z)
        C_x, C_y, C_z = _d(C, x), _d(C, y), _d(C, z)
        C_xx, C_yy, C_zz = _d(C_x, x), _d(C_y, y), _d(C_z, z)

        return dict(u=u, v=v, w=w, C=C, p=p,
                    u_x=u_x, u_y=u_y, u_z=u_z, u_xx=u_xx, u_yy=u_yy, u_zz=u_zz,
                    v_x=v_x, v_y=v_y, v_z=v_z, v_xx=v_xx, v_yy=v_yy, v_zz=v_zz,
                    w_x=w_x, w_y=w_y, w_z=w_z, w_xx=w_xx, w_yy=w_yy, w_zz=w_zz,
                    p_x=p_x, p_y=p_y, p_z=p_z,
                    C_x=C_x, C_y=C_y, C_z=C_z, C_xx=C_xx, C_yy=C_yy, C_zz=C_zz)

    def source_term(x, y, z):
        return 5.0 * torch.exp(-((x - 0.2) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) / (2 * 0.06 ** 2))

    def physics_losses(model, cx, cy, cz, cV, D, K, nu):
        f = derive_fields(model, cx, cy, cz, cV)
        S_val = source_term(cx, cy, cz)

        transport = (f["u"] * f["C_x"] + f["v"] * f["C_y"] + f["w"] * f["C_z"]
                     - D * (f["C_xx"] + f["C_yy"] + f["C_zz"]) - S_val + K * f["C"])
        mom_u = (f["u"] * f["u_x"] + f["v"] * f["u_y"] + f["w"] * f["u_z"]
                 + f["p_x"] - nu * (f["u_xx"] + f["u_yy"] + f["u_zz"]))
        mom_v = (f["u"] * f["v_x"] + f["v"] * f["v_y"] + f["w"] * f["v_z"]
                 + f["p_y"] - nu * (f["v_xx"] + f["v_yy"] + f["v_zz"]))
        mom_w = (f["u"] * f["w_x"] + f["v"] * f["w_y"] + f["w"] * f["w_z"]
                 + f["p_z"] - nu * (f["w_xx"] + f["w_yy"] + f["w_zz"]))

        transport_loss = torch.mean(transport ** 2)
        momentum_loss = torch.mean(mom_u ** 2) + torch.mean(mom_v ** 2) + torch.mean(mom_w ** 2)
        return transport_loss, momentum_loss

    model = ParametricOperator3D(args.hidden, args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)], gamma=0.1
    )
    mse = nn.MSELoss()

    def sample_collocation(n):
        cx = (x_lo + torch.rand(n, 1, device=device) * (x_hi - x_lo)).requires_grad_(True)
        cy = (y_lo + torch.rand(n, 1, device=device) * (y_hi - y_lo)).requires_grad_(True)
        cz = (z_lo + torch.rand(n, 1, device=device) * (z_hi - z_lo)).requires_grad_(True)
        cV = (args.v_min + torch.rand(n, 1, device=device) * (args.v_max - args.v_min)).requires_grad_(True)
        return cx, cy, cz, cV

    def sample_bc(n):
        # Left face (x=0, the "window"): known velocity = V (real sensor).
        by = torch.rand(n, 1, device=device) * (y_hi - y_lo) + y_lo
        bz = torch.rand(n, 1, device=device) * (z_hi - z_lo) + z_lo
        bV = args.v_min + torch.rand(n, 1, device=device) * (args.v_max - args.v_min)
        u_target = bV * 4.0 * bz * (1.0 - bz)
        v_target = torch.zeros_like(u_target)
        w_target = torch.zeros_like(u_target)
        bx = torch.full_like(by, x_lo)
        return bx, by, bz, bV, u_target, v_target, w_target

    def sample_walls(n):
        # No-slip on z=0, z=1 ONLY (top/bottom plates). y is NOT a wall in
        # this test flow -- true_velocity(x,y,z,V) = V*4*z*(1-z) does not
        # depend on y at all, so it is nonzero at y=0/y=1 (verified: e.g.
        # V=0.7, z=0.5 gives u=0.7 regardless of y). Imposing no-slip at
        # y=0/y=1 (an earlier bug in this script) directly contradicted the
        # true solution and was fighting the physics loss throughout
        # training -- that is why the loss plateaued and errors were high.
        n_each = n // 2
        pts = []
        for fixed_val in [z_lo, z_hi]:
            wx = torch.rand(n_each, 1, device=device) * (x_hi - x_lo) + x_lo
            wy = torch.rand(n_each, 1, device=device) * (y_hi - y_lo) + y_lo
            wz = torch.full_like(wx, fixed_val)
            pts.append((wx, wy, wz))
        wx = torch.cat([p[0] for p in pts], dim=0)
        wy = torch.cat([p[1] for p in pts], dim=0)
        wz = torch.cat([p[2] for p in pts], dim=0)
        wV = args.v_min + torch.rand(wx.shape[0], 1, device=device) * (args.v_max - args.v_min)
        return wx, wy, wz, wV

    print(f"[train] training for {args.epochs} epochs, {args.n_collocation} collocation points/epoch, "
          f"sampling V ~ U({args.v_min}, {args.v_max}) at every step")

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        bx, by, bz, bV, u_t, v_t, w_t = sample_bc(args.n_bc)
        bc_fields = derive_fields(
            model, bx.clone().requires_grad_(True), by.clone().requires_grad_(True),
            bz.clone().requires_grad_(True), bV.clone().requires_grad_(True)
        )
        bc_loss = mse(bc_fields["u"], u_t) + mse(bc_fields["v"], v_t) + mse(bc_fields["w"], w_t) + mse(bc_fields["C"], torch.zeros_like(u_t))

        wx, wy, wz, wV = sample_walls(args.n_bc)
        wall_fields = derive_fields(
            model, wx.clone().requires_grad_(True), wy.clone().requires_grad_(True),
            wz.clone().requires_grad_(True), wV.clone().requires_grad_(True)
        )
        wall_loss = (mse(wall_fields["u"], torch.zeros_like(wall_fields["u"]))
                     + mse(wall_fields["v"], torch.zeros_like(wall_fields["v"]))
                     + mse(wall_fields["w"], torch.zeros_like(wall_fields["w"])))

        cx, cy, cz, cV = sample_collocation(args.n_collocation)
        transport_loss, momentum_loss = physics_losses(model, cx, cy, cz, cV, args.D, args.K, args.nu)

        loss = args.lambda_bc * (bc_loss + wall_loss) + args.lambda_transport * transport_loss + args.lambda_mom * momentum_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(f"  epoch {epoch:6d}  lr={scheduler.get_last_lr()[0]:.1e}  "
                  f"bc={bc_loss.item():.6e}  wall={wall_loss.item():.6e}  "
                  f"transport={transport_loss.item():.6e}  momentum={momentum_loss.item():.6e}  "
                  f"total={loss.item():.6e}")

    torch.save({"model_state_dict": model.state_dict(),
                "hidden": args.hidden, "layers": args.layers,
                "v_min": args.v_min, "v_max": args.v_max}, args.model_out)
    print(f"\n[save] model checkpoint saved to {args.model_out}")

    model.eval()
    gy, gz = np.meshgrid(np.linspace(0.05, 0.95, 25), np.linspace(0.05, 0.95, 25), indexing="ij")
    gx = np.full_like(gy, 0.5)  # mid-plane slice at x=0.5 for visualization
    gy_flat, gz_flat, gx_flat = gy.reshape(-1), gz.reshape(-1), gx.reshape(-1)

    def eval_at_V(V_val):
        ex = torch.tensor(gx_flat.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
        ey = torch.tensor(gy_flat.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
        ez = torch.tensor(gz_flat.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
        eV = torch.full_like(ex, V_val, requires_grad=True)
        f = derive_fields(model, ex, ey, ez, eV)
        u_pred = f["u"].detach().cpu().numpy().reshape(gy.shape)
        v_pred = f["v"].detach().cpu().numpy().reshape(gy.shape)
        w_pred = f["w"].detach().cpu().numpy().reshape(gy.shape)
        u_true, v_true, w_true = true_velocity(gx, gy, gz, V_val)
        speed_true = np.sqrt(u_true ** 2 + v_true ** 2 + w_true ** 2)
        speed_pred = np.sqrt(u_pred ** 2 + v_pred ** 2 + w_pred ** 2)
        rel_err = np.linalg.norm(speed_pred - speed_true) / (np.linalg.norm(speed_true) + 1e-8)
        return u_pred, v_pred, w_pred, u_true, v_true, w_true, rel_err

    print("\n=== RESULT: 3D operator generalization test ===")
    test_Vs = [args.v_min, 0.5 * (args.v_min + args.v_max), args.v_max,
               args.v_max + 0.3 * (args.v_max - args.v_min)]
    labels = ["V_min (trained edge)", "V_mid (trained)", "V_max (trained edge)", "V_max+30% (extrapolation)"]
    results = {}
    for V_val, label in zip(test_Vs, labels):
        u_p, v_p, w_p, u_t2, v_t2, w_t2, err = eval_at_V(V_val)
        results[V_val] = (u_p, v_p, w_p, u_t2, v_t2, w_t2, err)
        print(f"  {label}: V={V_val:.3f}, speed error {100*err:.2f}%")

    # Fine sweep for the error-vs-V chart
    sweep_lo = args.v_min - 0.3 * (args.v_max - args.v_min)
    sweep_hi = args.v_max + 0.3 * (args.v_max - args.v_min)
    sweep_Vs = np.linspace(sweep_lo, sweep_hi, 25)
    sweep_errs = []
    for V_val in sweep_Vs:
        *_, err = eval_at_V(float(V_val))
        sweep_errs.append(100 * err)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: predicted vs true, at 3 representative V values (skip extrapolation for space)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for col, (V_val, label) in enumerate(zip(test_Vs[:3], labels[:3])):
        u_p, v_p, w_p, u_t2, v_t2, w_t2, err = results[V_val]
        axes[0, col].quiver(gy, gz, u_t2, w_t2)
        axes[0, col].set_title(f"TRUE flow, {label}\nV={V_val:.2f}")
        axes[0, col].set_xlabel("y"); axes[0, col].set_ylabel("z")
        axes[1, col].quiver(gy, gz, u_p, w_p)
        axes[1, col].set_title(f"PREDICTED flow, error={100*err:.1f}%")
        axes[1, col].set_xlabel("y"); axes[1, col].set_ylabel("z")
    plt.tight_layout()
    plt.savefig(args.fig_out, dpi=150)
    print(f"[save] predicted-vs-true comparison figure saved to {args.fig_out}")

    # Figure 2: error vs V, with trained range shaded
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.axvspan(args.v_min, args.v_max, color="lightgreen", alpha=0.3, label="Trained range")
    ax2.plot(sweep_Vs, sweep_errs, marker="o", color="steelblue")
    ax2.set_xlabel("Window velocity V")
    ax2.set_ylabel("Speed error (%)")
    ax2.set_title("Generalization error vs. window velocity\n(one model, never retrained per V)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.error_fig_out, dpi=150)
    print(f"[save] error-vs-V chart saved to {args.error_fig_out}")


if __name__ == "__main__":
    main()
