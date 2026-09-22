"""
A PINTO-style test on OUR OWN channel-flow scenario, meant to be a fair,
apples-to-apples comparison against pino_parametric_test.py.

--------------------------------------------------------------------------
WHY THIS SCRIPT EXISTS / WHAT MAKES IT DIFFERENT FROM pino_parametric_test.py
--------------------------------------------------------------------------
pino_parametric_test.py feeds the network a single number (V, the window
velocity) and assumes the WHOLE boundary is known continuously. That is not
your real situation -- you will have 6-7 discrete sensors, not a
continuously-known boundary.

This script tests the actual PINTO idea (Boya & Subramani, arXiv:2412.09009)
on that more realistic setup: the network is given ONLY a handful of fixed
sensor points along the window (position + reading at each), and has to
recover the correct flow everywhere else in the room using cross-attention
over those sparse points, combined with the physics equations. No single
"V" number is ever given to the network directly -- it only sees sensor
readings, exactly like the real deployment.

Architecture (a simplified, from-scratch PyTorch reproduction of PINTO's
cross-attention mechanism, not a copy of their TensorFlow code):
    - Query Point Encoding (QPE): encodes the (x, y) point being asked about
    - Boundary Position Encoding (BPE): encodes each sensor's location
    - Boundary Value Encoding (BVE): encodes each sensor's reading
    - Two cross-attention blocks (torch.nn.MultiheadAttention): the query's
      representation attends over the sensor positions/readings, pulling in
      whichever sensors are most relevant -- this is the actual mechanism
      from the paper (their Eq. 6-8 is scaled dot-product attention, which
      is exactly what nn.MultiheadAttention implements).
    - Output projection -> (psi, C, p), same streamfunction trick as
      pino_parametric_test.py to guarantee divergence-free velocity.

--------------------------------------------------------------------------
THE TEST SCENARIO
--------------------------------------------------------------------------
Same exact-solution family as before: u(x,y;V) = V*4*y*(1-y), v=0. But now
the network only ever sees 6 fixed sensor points along the left edge (at
y = 0.1, 0.25, 0.4, 0.6, 0.75, 0.9 -- denser near mid-height, matching
where velocity varies fastest) -- never the formula, never a raw V value.

For a fresh V sampled each training step, the sensor readings are computed
from the true profile at those 6 fixed y-positions and fed to the network
as its only knowledge of the boundary condition.

--------------------------------------------------------------------------
WHAT'S VERIFIED VS. NOT
--------------------------------------------------------------------------
Same as every script in this project: the true-solution math (exact NS
solution, linear in V) was already verified numerically in earlier scripts
and is unchanged here. The training loop itself has NOT been run by me --
please run it and paste the output.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python pinto_channel_flow_test.py
    python pinto_channel_flow_test.py --n-sensors 4
"""

import argparse
import numpy as np


def true_velocity(x, y, V):
    u = V * 4.0 * y * (1.0 - y)
    zero = np.zeros_like(u) if hasattr(u, "shape") else 0.0
    return u, zero


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v-min", type=float, default=0.2)
    parser.add_argument("--v-max", type=float, default=1.2)
    parser.add_argument("--D", type=float, default=0.01)
    parser.add_argument("--K", type=float, default=0.2)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--n-sensors", type=int, default=6,
                         help="Number of fixed sensor points along the window (left edge)")
    parser.add_argument("--n-collocation", type=int, default=20000)
    parser.add_argument("--n-wall", type=int, default=300)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-mom", type=float, default=10.0)
    parser.add_argument("--lambda-transport", type=float, default=10.0)
    parser.add_argument("--lambda-bc", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-out", type=str, default="pinto_channel_flow_model.pth")
    parser.add_argument("--fig-out", type=str, default="pinto_channel_flow_result.png")
    parser.add_argument("--error-fig-out", type=str, default="pinto_channel_flow_error_vs_V.png")
    args = parser.parse_args()

    np.random.seed(args.seed)
    x_lo, x_hi, y_lo, y_hi = 0.0, 1.0, 0.0, 1.0

    # Fixed sensor y-positions along the window (left edge). Denser near
    # mid-height, where the parabolic profile changes fastest -- a
    # reasonable placement choice, not a measured real one (per the
    # professor's "assume for now" guidance).
    sensor_y = np.linspace(0.1, 0.9, args.n_sensors)
    print(f"[setup] {args.n_sensors} fixed sensors at y = {np.round(sensor_y, 3).tolist()}")
    print(f"[setup] training a single cross-attention operator over V in [{args.v_min}, {args.v_max}]")
    print("[setup] the network NEVER sees V directly -- only these sensor readings")

    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[train] using device: {device}")

    sensor_y_t = torch.tensor(sensor_y, dtype=torch.float32, device=device)  # (L,)
    L = args.n_sensors
    m = args.embed_dim

    class CrossAttnOperator(nn.Module):
        """PINTO-style operator: query point attends over sensor
        (position, value) pairs via cross-attention, no raw parameter
        (like V) ever given directly."""
        def __init__(self, m, n_heads):
            super().__init__()
            self.qpe = nn.Sequential(nn.Linear(2, m), nn.Tanh())
            self.bpe = nn.Sequential(nn.Linear(1, m), nn.Tanh())
            self.bve = nn.Sequential(nn.Linear(2, m), nn.Tanh())
            self.mha1 = nn.MultiheadAttention(embed_dim=m, num_heads=n_heads, batch_first=True)
            self.dense1 = nn.Sequential(nn.Linear(m, m), nn.Tanh(), nn.Linear(m, m))
            self.mha2 = nn.MultiheadAttention(embed_dim=m, num_heads=n_heads, batch_first=True)
            self.dense2 = nn.Sequential(nn.Linear(m, m), nn.Tanh(), nn.Linear(m, m))
            self.out = nn.Linear(m, 3)  # (psi, C, p)

        def forward(self, x, y, sensor_pos, sensor_val):
            # x, y: (N, 1). sensor_pos: (N, L, 1). sensor_val: (N, L, 2).
            q = self.qpe(torch.cat([x, y], dim=1)).unsqueeze(1)          # (N, 1, m)
            k = self.bpe(sensor_pos)                                     # (N, L, m)
            v = self.bve(sensor_val)                                     # (N, L, m)

            attn1, _ = self.mha1(q, k, v)
            h1 = torch.tanh(q + attn1)
            h1 = h1 + self.dense1(h1)

            attn2, _ = self.mha2(h1, k, v)
            h2 = torch.tanh(h1 + attn2)
            h2 = h2 + self.dense2(h2)

            out = self.out(h2.squeeze(1))                                # (N, 3)
            psi, C, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
            return psi, C, p

    def _d(f, wrt):
        return torch.autograd.grad(f, wrt, grad_outputs=torch.ones_like(f), create_graph=True)[0]

    def make_sensor_context(V_batch):
        """V_batch: (N, 1) tensor of sampled V's -> sensor_pos (N, L, 1), sensor_val (N, L, 2)."""
        N = V_batch.shape[0]
        pos = sensor_y_t.view(1, L, 1).expand(N, L, 1)
        u_val = V_batch * 4.0 * sensor_y_t.view(1, L) * (1.0 - sensor_y_t.view(1, L))  # (N, L)
        v_val = torch.zeros_like(u_val)
        val = torch.stack([u_val, v_val], dim=-1)  # (N, L, 2)
        return pos, val

    def derive_fields(model, x, y, sensor_pos, sensor_val):
        psi, C, p = model(x, y, sensor_pos, sensor_val)
        psi_x, psi_y = _d(psi, x), _d(psi, y)
        u, v = psi_y, -psi_x

        u_x, u_y = _d(u, x), _d(u, y)
        v_x, v_y = _d(v, x), _d(v, y)
        u_xx, u_yy = _d(u_x, x), _d(u_y, y)
        v_xx, v_yy = _d(v_x, x), _d(v_y, y)

        p_x, p_y = _d(p, x), _d(p, y)
        C_x, C_y = _d(C, x), _d(C, y)
        C_xx, C_yy = _d(C_x, x), _d(C_y, y)

        return dict(u=u, v=v, C=C, p=p,
                    u_x=u_x, u_y=u_y, u_xx=u_xx, u_yy=u_yy,
                    v_x=v_x, v_y=v_y, v_xx=v_xx, v_yy=v_yy,
                    p_x=p_x, p_y=p_y, C_x=C_x, C_y=C_y, C_xx=C_xx, C_yy=C_yy)

    def source_term(x, y):
        return 5.0 * torch.exp(-((x - 0.2) ** 2 + (y - 0.5) ** 2) / (2 * 0.06 ** 2))

    def physics_losses(model, cx, cy, sensor_pos, sensor_val, D, K, nu):
        f = derive_fields(model, cx, cy, sensor_pos, sensor_val)
        S_val = source_term(cx, cy)
        transport = (f["u"] * f["C_x"] + f["v"] * f["C_y"]
                     - D * (f["C_xx"] + f["C_yy"]) - S_val + K * f["C"])
        mom_u = f["u"] * f["u_x"] + f["v"] * f["u_y"] + f["p_x"] - nu * (f["u_xx"] + f["u_yy"])
        mom_v = f["u"] * f["v_x"] + f["v"] * f["v_y"] + f["p_y"] - nu * (f["v_xx"] + f["v_yy"])
        return torch.mean(transport ** 2), torch.mean(mom_u ** 2) + torch.mean(mom_v ** 2)

    model = CrossAttnOperator(m, args.n_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)], gamma=0.1
    )
    mse = nn.MSELoss()

    def sample_V(n):
        return (args.v_min + torch.rand(n, 1, device=device) * (args.v_max - args.v_min))

    def sample_collocation(n):
        cx = (x_lo + torch.rand(n, 1, device=device) * (x_hi - x_lo)).requires_grad_(True)
        cy = (y_lo + torch.rand(n, 1, device=device) * (y_hi - y_lo)).requires_grad_(True)
        return cx, cy

    def sample_walls(n):
        n_each = n // 2
        wx1 = torch.rand(n_each, 1, device=device) * (x_hi - x_lo) + x_lo
        wy1 = torch.full_like(wx1, y_lo)
        wx2 = torch.rand(n_each, 1, device=device) * (x_hi - x_lo) + x_lo
        wy2 = torch.full_like(wx2, y_hi)
        wx = torch.cat([wx1, wx2], dim=0)
        wy = torch.cat([wy1, wy2], dim=0)
        return wx, wy

    def sample_bc_check(n):
        # Query directly at the sensor points, to reinforce self-consistency.
        idx = torch.randint(0, L, (n, 1), device=device)
        by = sensor_y_t[idx.squeeze(1)].unsqueeze(1)
        bx = torch.full_like(by, x_lo)
        return bx, by

    print(f"[train] training for {args.epochs} epochs, {args.n_collocation} collocation points/epoch")

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        # --- sensor consistency loss (predict correctly AT the sensors) ---
        bx, by = sample_bc_check(args.n_wall)
        bV = sample_V(args.n_wall)
        b_pos, b_val = make_sensor_context(bV)
        bc_fields = derive_fields(model, bx.clone().requires_grad_(True), by.clone().requires_grad_(True), b_pos, b_val)
        u_target = bV.squeeze(1).unsqueeze(1) * 4.0 * by * (1.0 - by)
        bc_loss = mse(bc_fields["u"], u_target) + mse(bc_fields["v"], torch.zeros_like(u_target)) + mse(bc_fields["C"], torch.zeros_like(u_target))

        # --- wall no-slip loss ---
        wx, wy = sample_walls(args.n_wall)
        wV = sample_V(args.n_wall)
        w_pos, w_val = make_sensor_context(wV)
        wall_fields = derive_fields(model, wx.clone().requires_grad_(True), wy.clone().requires_grad_(True), w_pos, w_val)
        wall_loss = mse(wall_fields["u"], torch.zeros_like(wall_fields["u"])) + mse(wall_fields["v"], torch.zeros_like(wall_fields["v"]))

        # --- physics loss over the interior ---
        cx, cy = sample_collocation(args.n_collocation)
        cV = sample_V(args.n_collocation)
        c_pos, c_val = make_sensor_context(cV)
        transport_loss, momentum_loss = physics_losses(model, cx, cy, c_pos, c_val, args.D, args.K, args.nu)

        loss = args.lambda_bc * (bc_loss + wall_loss) + args.lambda_transport * transport_loss + args.lambda_mom * momentum_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(f"  epoch {epoch:6d}  lr={scheduler.get_last_lr()[0]:.1e}  "
                  f"bc={bc_loss.item():.6e}  wall={wall_loss.item():.6e}  "
                  f"transport={transport_loss.item():.6e}  momentum={momentum_loss.item():.6e}  "
                  f"total={loss.item():.6e}")

    torch.save({"model_state_dict": model.state_dict(), "embed_dim": m, "n_heads": args.n_heads,
                "sensor_y": sensor_y.tolist(), "v_min": args.v_min, "v_max": args.v_max}, args.model_out)
    print(f"\n[save] model checkpoint saved to {args.model_out}")

    model.eval()
    gx, gy = np.meshgrid(np.linspace(0.02, 0.98, 40), np.linspace(0.02, 0.98, 40), indexing="ij")
    gx_flat, gy_flat = gx.reshape(-1), gy.reshape(-1)

    def eval_at_V(V_val):
        ex = torch.tensor(gx_flat.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
        ey = torch.tensor(gy_flat.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
        eV = torch.full((ex.shape[0], 1), V_val, device=device)
        e_pos, e_val = make_sensor_context(eV)
        f = derive_fields(model, ex, ey, e_pos, e_val)
        u_pred = f["u"].detach().cpu().numpy().reshape(gx.shape)
        v_pred = f["v"].detach().cpu().numpy().reshape(gx.shape)
        u_true, v_true = true_velocity(gx, gy, V_val)
        speed_true = np.sqrt(u_true ** 2 + v_true ** 2)
        speed_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
        rel_err = np.linalg.norm(speed_pred - speed_true) / (np.linalg.norm(speed_true) + 1e-8)
        return u_pred, v_pred, u_true, v_true, rel_err

    print("\n=== RESULT: cross-attention operator, fed ONLY sparse sensor readings ===")
    test_Vs = [args.v_min, 0.5 * (args.v_min + args.v_max), args.v_max,
               args.v_max + 0.3 * (args.v_max - args.v_min)]
    labels = ["V_min (trained edge)", "V_mid (trained)", "V_max (trained edge)", "V_max+30% (extrapolation)"]
    results = {}
    for V_val, label in zip(test_Vs, labels):
        u_p, v_p, u_t, v_t, err = eval_at_V(V_val)
        results[V_val] = (u_p, v_p, u_t, v_t, err)
        print(f"  {label}: V={V_val:.3f}, speed error {100*err:.2f}%")

    sweep_lo = args.v_min - 0.3 * (args.v_max - args.v_min)
    sweep_hi = args.v_max + 0.3 * (args.v_max - args.v_min)
    sweep_Vs = np.linspace(sweep_lo, sweep_hi, 25)
    sweep_errs = [100 * eval_at_V(float(V))[-1] for V in sweep_Vs]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for col, (V_val, label) in enumerate(zip(test_Vs[:3], labels[:3])):
        u_p, v_p, u_t, v_t, err = results[V_val]
        axes[0, col].quiver(gx[::3, ::3], gy[::3, ::3], u_t[::3, ::3], v_t[::3, ::3])
        axes[0, col].scatter(np.zeros_like(sensor_y), sensor_y, c="red", marker="^", zorder=5, label="sensors")
        axes[0, col].set_title(f"TRUE flow, {label}\nV={V_val:.2f}")
        axes[0, col].legend(fontsize=7)
        axes[1, col].quiver(gx[::3, ::3], gy[::3, ::3], u_p[::3, ::3], v_p[::3, ::3])
        axes[1, col].set_title(f"PREDICTED (from {L} sensors only), error={100*err:.1f}%")
    plt.tight_layout()
    plt.savefig(args.fig_out, dpi=150)
    print(f"[save] predicted-vs-true comparison figure saved to {args.fig_out}")

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.axvspan(args.v_min, args.v_max, color="lightgreen", alpha=0.3, label="Trained range")
    ax2.plot(sweep_Vs, sweep_errs, marker="o", color="darkorange")
    ax2.set_xlabel("Window velocity V (never given directly -- inferred from sensors)")
    ax2.set_ylabel("Speed error (%)")
    ax2.set_title(f"Cross-attention operator: error vs. V, using only {L} fixed sensors")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.error_fig_out, dpi=150)
    print(f"[save] error-vs-V chart saved to {args.error_fig_out}")


if __name__ == "__main__":
    main()
