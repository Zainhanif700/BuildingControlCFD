"""
Visualize a trained GNOT checkpoint's predictions in the real room.

Picks a scenario (which windows are open, how fast, how many people, what
time), queries the model on a grid of points throughout the real room, and
renders:
  - a 3D quiver plot of predicted airflow (arrows sized/colored by speed)
  - CO2 concentration as a colored scatter cloud
  - the real room geometry (walls outline, window/door locations, columns)
    drawn alongside so the prediction is visually grounded in the actual
    room, not a generic empty box.

This is meant to be a clear, presentable "here's what the model predicts"
figure -- e.g. for showing Prof. Wagner -- not a rigorous accuracy report
(see the (planned) GNOT vs PINTO comparison for that).

Usage: edit SCENARIO below, then run on the server (needs torch + the
trained checkpoint):
    python3 visualize_gnot.py
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from gnot_model import GNOTOperator, check_checkpoint_compat
from point_sampler import ROOM_X, ROOM_Y, ROOM_Z, WINDOWS, DOORS, COLUMNS, NUM_WINDOWS

# EDIT THIS to switch which trained version you're visualizing.
#   "v1_smooth_co2" -- original run, KNOWN BAD CO2 (room-wide smooth gradient,
#                      confirmed physically impossible by the closed-window test)
#   "v2_co2_fix"    -- multi-octave Fourier features + adaptive CO2 loss weight
VERSION = "v8_nondim"  # pre-v8 checkpoints are refused by check_checkpoint_compat

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(HERE, "checkpoints", VERSION, f"gnot_{VERSION}_final.pth")
FIG_DIR = os.path.join(HERE, "figures", VERSION)
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# EDIT THIS: pick the scenario you want to visualize.
# ---------------------------------------------------------------------------
SCENARIO = {
    "V": [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0],  # m/s per window (0 = closed)
    "N_people": 20.0,
    "t": 60.0,  # seconds into the 0-120s ramp window
}

GRID_NX, GRID_NY, GRID_NZ = 14, 9, 6  # query grid resolution


def in_any_column(x, y):
    inside = np.zeros_like(x, dtype=bool)
    for cx, cy, r, _, _ in COLUMNS:
        inside |= (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    return inside


def build_query_grid():
    xs = np.linspace(ROOM_X[0] + 0.3, ROOM_X[1] - 0.3, GRID_NX)
    ys = np.linspace(ROOM_Y[0] + 0.3, ROOM_Y[1] - 0.3, GRID_NY)
    zs = np.linspace(ROOM_Z[0] + 0.2, ROOM_Z[1] - 0.2, GRID_NZ)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()
    keep = ~in_any_column(X, Y)  # don't query inside solid columns
    return X[keep], Y[keep], Z[keep]


def draw_room_geometry(ax):
    """Wireframe room outline + window (blue) / door (red) / column (gray) markers."""
    xlo, xhi = ROOM_X; ylo, yhi = ROOM_Y; zlo, zhi = ROOM_Z
    corners = np.array([[xlo, ylo, zlo], [xhi, ylo, zlo], [xhi, yhi, zlo], [xlo, yhi, zlo],
                        [xlo, ylo, zhi], [xhi, ylo, zhi], [xhi, yhi, zhi], [xlo, yhi, zhi]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edges:
        ax.plot(*zip(corners[i], corners[j]), color="black", linewidth=0.8, alpha=0.5)

    for k, (xlo_w, xhi_w, zlo_w, zhi_w) in enumerate(WINDOWS):
        v = SCENARIO["V"][k]
        color = "deepskyblue" if v > 0.01 else "lightgray"
        verts = [[(xlo_w, yhi, zlo_w), (xhi_w, yhi, zlo_w), (xhi_w, yhi, zhi_w), (xlo_w, yhi, zhi_w)]]
        ax.add_collection3d(Poly3DCollection(verts, facecolor=color, alpha=0.6, edgecolor="k"))

    for (xlo_d, xhi_d, zlo_d, zhi_d) in DOORS:
        verts = [[(xlo_d, ylo, zlo_d), (xhi_d, ylo, zlo_d), (xhi_d, ylo, zhi_d), (xlo_d, ylo, zhi_d)]]
        ax.add_collection3d(Poly3DCollection(verts, facecolor="orangered", alpha=0.6, edgecolor="k"))

    theta = np.linspace(0, 2 * np.pi, 16)
    for cx, cy, r, czlo, czhi in COLUMNS:
        cx_pts = cx + r * np.cos(theta)
        cy_pts = cy + r * np.sin(theta)
        for zz in [czlo, czhi]:
            ax.plot(cx_pts, cy_pts, zz, color="saddlebrown", linewidth=1.0, alpha=0.7)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GNOTOperator().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    check_checkpoint_compat(ckpt, CKPT_PATH)  # v8_nondim: refuse pre-v8 checkpoints
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from iter {ckpt.get('iter', '?')}: {CKPT_PATH}")

    xg, yg, zg = build_query_grid()
    n = xg.shape[0]
    print(f"Querying {n} points in the real room...")

    x = torch.tensor(xg, dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
    y = torch.tensor(yg, dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
    z = torch.tensor(zg, dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
    t = torch.full((n, 1), SCENARIO["t"], device=device)
    V = torch.tensor(SCENARIO["V"], dtype=torch.float32, device=device).view(1, NUM_WINDOWS).expand(n, -1)
    N_people = torch.full((n, 1), SCENARIO["N_people"], device=device)

    A1, A2, A3, C, p = model(x, y, z, t, V, N_people)
    u, v, w = model.velocity_from_potential(A1, A2, A3, x, y, z)

    u = u.detach().cpu().numpy().ravel()
    v = v.detach().cpu().numpy().ravel()
    w = w.detach().cpu().numpy().ravel()
    c = C.detach().cpu().numpy().ravel()
    speed = np.sqrt(u ** 2 + v ** 2 + w ** 2)

    print(f"Velocity magnitude: min={speed.min():.4f}, max={speed.max():.4f}, mean={speed.mean():.4f} m/s")
    print(f"CO2 (c): min={c.min():.4f}, max={c.max():.4f}, mean={c.mean():.4f}")

    # --- figure 1: airflow quiver ---
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    draw_room_geometry(ax)
    q = ax.quiver(xg, yg, zg, u, v, w, length=0.6, normalize=True, color=plt.cm.viridis(speed / (speed.max() + 1e-9)))
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    win_str = ", ".join(f"W{i+1}={vv:.1f}" for i, vv in enumerate(SCENARIO["V"]))
    ax.set_title(f"GNOT [{VERSION}] predicted airflow at t={SCENARIO['t']:.0f}s, N_people={SCENARIO['N_people']:.0f}\n{win_str}")
    out1 = os.path.join(FIG_DIR, f"gnot_{VERSION}_airflow.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved: {out1}")

    # --- figure 2: CO2 concentration cloud ---
    fig2 = plt.figure(figsize=(13, 10))
    ax2 = fig2.add_subplot(111, projection="3d")
    draw_room_geometry(ax2)
    sc = ax2.scatter(xg, yg, zg, c=c, cmap="YlOrRd", s=25, alpha=0.7)
    fig2.colorbar(sc, ax=ax2, shrink=0.6, label="predicted CO2 (c)")
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)"); ax2.set_zlabel("Z (m)")
    ax2.set_title(f"GNOT [{VERSION}] predicted CO2 at t={SCENARIO['t']:.0f}s, N_people={SCENARIO['N_people']:.0f}\n{win_str}")
    out2 = os.path.join(FIG_DIR, f"gnot_{VERSION}_co2.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
