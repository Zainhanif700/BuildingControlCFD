"""
Clear top-down (2D) slice visualization of a trained GNOT checkpoint.

The 3D quiver/scatter in visualize_gnot.py is hard to read precisely
(perspective, occlusion). This instead takes a horizontal slice at a fixed
height and plots a clean top-down view -- same style as geometry/floor_plan.py
-- so it's unambiguous whether the model learned sensible physics:
  - velocity arrows should point INTO the room at open windows, be near-zero
    at closed windows/walls, and curve around the columns (never through them)
  - CO2 should be highest near the room center at breathing height (where the
    occupancy source is) and lowest near the open windows (clean air in)

Usage: python3 visualize_gnot_slice.py
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from gnot_model import GNOTOperator, check_checkpoint_compat
from point_sampler import ROOM_X, ROOM_Y, ROOM_Z, WINDOWS, DOORS, COLUMNS, NUM_WINDOWS, BREATHING_HEIGHT

# EDIT THIS to switch which trained version you're visualizing.
#   "v1_smooth_co2" -- original run, KNOWN BAD CO2 (room-wide smooth gradient,
#                      confirmed physically impossible by the closed-window test)
#   "v2_co2_fix"    -- multi-octave Fourier features + adaptive CO2 loss weight
VERSION = "v8_nondim"  # pre-v8 checkpoints are refused by check_checkpoint_compat

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(HERE, "checkpoints", VERSION, f"gnot_{VERSION}_final.pth")
FIG_DIR = os.path.join(HERE, "figures", VERSION)
os.makedirs(FIG_DIR, exist_ok=True)

SCENARIO = {
    "V": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # all windows closed -- isolates the CO2 source
    "N_people": 20.0,
    "t": 60.0,
}
SLICE_Z = BREATHING_HEIGHT  # single source of truth lives in point_sampler.py now
GRID_N = 40      # resolution of the 2D slice (GRID_N x GRID_N)


def in_any_column(x, y):
    inside = np.zeros_like(x, dtype=bool)
    for cx, cy, r, _, _ in COLUMNS:
        inside |= (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    return inside


def draw_floor_plan(ax):
    xlo, xhi = ROOM_X; ylo, yhi = ROOM_Y
    ax.add_patch(patches.Rectangle((xlo, ylo), xhi - xlo, yhi - ylo, fill=False, edgecolor="black", linewidth=2))
    for i, (wxlo, wxhi, _, _) in enumerate(WINDOWS):
        v = SCENARIO["V"][i]
        color = "deepskyblue" if v > 0.01 else "lightgray"
        ax.add_patch(patches.Rectangle((wxlo, yhi - 0.1), wxhi - wxlo, 0.2, facecolor=color, edgecolor="k", zorder=5))
    for (dxlo, dxhi, _, _) in DOORS:
        ax.add_patch(patches.Rectangle((dxlo, ylo - 0.1), dxhi - dxlo, 0.2, facecolor="orangered", edgecolor="k", zorder=5))
    for cx, cy, r, _, _ in COLUMNS:
        ax.add_patch(patches.Circle((cx, cy), r, facecolor="saddlebrown", edgecolor="k", alpha=0.8, zorder=5))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GNOTOperator().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    check_checkpoint_compat(ckpt, CKPT_PATH)  # v8_nondim: refuse pre-v8 checkpoints
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from iter {ckpt.get('iter', '?')}")

    xs = np.linspace(ROOM_X[0] + 0.1, ROOM_X[1] - 0.1, GRID_N)
    ys = np.linspace(ROOM_Y[0] + 0.1, ROOM_Y[1] - 0.1, GRID_N)
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    xg, yg = Xg.ravel(), Yg.ravel()
    inside_column = in_any_column(xg, yg)  # mask, don't drop (keep grid regular for imshow)

    n = xg.shape[0]
    x = torch.tensor(xg, dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
    y = torch.tensor(yg, dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
    z = torch.full((n, 1), SLICE_Z, device=device, requires_grad=True)
    t = torch.full((n, 1), SCENARIO["t"], device=device)
    V = torch.tensor(SCENARIO["V"], dtype=torch.float32, device=device).view(1, NUM_WINDOWS).expand(n, -1)
    N_people = torch.full((n, 1), SCENARIO["N_people"], device=device)

    A1, A2, A3, C, p = model(x, y, z, t, V, N_people)
    u, v, w = model.velocity_from_potential(A1, A2, A3, x, y, z)

    u = u.detach().cpu().numpy().ravel(); u[inside_column] = np.nan
    v = v.detach().cpu().numpy().ravel(); v[inside_column] = np.nan
    c = C.detach().cpu().numpy().ravel(); c[inside_column] = np.nan
    speed = np.sqrt(u ** 2 + v ** 2)

    print(f"[z={SLICE_Z}m slice] speed: min={np.nanmin(speed):.4f} max={np.nanmax(speed):.4f} mean={np.nanmean(speed):.4f}")
    print(f"[z={SLICE_Z}m slice] CO2:   min={np.nanmin(c):.4f} max={np.nanmax(c):.4f} mean={np.nanmean(c):.4f}")

    win_str = ", ".join(f"W{i+1}={vv:.1f}" for i, vv in enumerate(SCENARIO["V"]))

    # --- velocity quiver (arrow length = ACTUAL speed this time, not normalized) ---
    fig, ax = plt.subplots(figsize=(12, 7))
    draw_floor_plan(ax)
    ax.quiver(xg, yg, u, v, speed, cmap="viridis", scale=8, width=0.003)
    ax.set_xlim(ROOM_X[0] - 1, ROOM_X[1] + 1); ax.set_ylim(ROOM_Y[0] - 1, ROOM_Y[1] + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(f"GNOT [{VERSION}] airflow, top-down slice at z={SLICE_Z}m, t={SCENARIO['t']:.0f}s, "
                 f"N_people={SCENARIO['N_people']:.0f}\n{win_str}")
    out1 = os.path.join(FIG_DIR, f"gnot_{VERSION}_slice_velocity_closed.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved: {out1}")

    # --- CO2 heatmap ---
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    Cgrid = c.reshape(GRID_N, GRID_N)
    im = ax2.pcolormesh(Xg, Yg, Cgrid, shading="auto", cmap="YlOrRd")
    draw_floor_plan(ax2)
    fig2.colorbar(im, ax=ax2, label="predicted CO2 (c)")
    ax2.set_xlim(ROOM_X[0] - 1, ROOM_X[1] + 1); ax2.set_ylim(ROOM_Y[0] - 1, ROOM_Y[1] + 1)
    ax2.set_aspect("equal")
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
    ax2.set_title(f"GNOT [{VERSION}] CO2, top-down slice at z={SLICE_Z}m, t={SCENARIO['t']:.0f}s, "
                  f"N_people={SCENARIO['N_people']:.0f}\n{win_str}")
    out2 = os.path.join(FIG_DIR, f"gnot_{VERSION}_slice_co2_closed.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
