"""
Closed-window CO2/velocity diagnostic -- takes ANY checkpoint path as an
argument, so it's reusable across versions (v3_isotropic_ff, v4_source_sampling,
future ones) without editing the script each time.

Scenario: all 8 windows closed (V=0 everywhere), 20 people, t=60s. With no
inflow forcing at all, velocity should be near-zero everywhere (any large
speed here is the still-unfixed "spurious closed-window velocity" artifact,
fix #3, not yet attempted). CO2 should show a compact, localized bump near
the room center at breathing height -- NOT a room-wide band (the original
bug) and not a flat near-zero field (the "undertrained" symptom seen on the
fix-#1-only checkpoint).

This is fast: one forward pass over a 40x40 grid (1600 points), no training,
seconds on GPU even without one.

Usage:
    python3 closed_window_diagnostic.py checkpoints/v4_source_sampling/gnot_v4_source_sampling_partial10000.pth
"""
import sys
import torch
import numpy as np

from gnot_model import GNOTOperator
from point_sampler import ROOM_X, ROOM_Y, ROOM_Z, COLUMNS, NUM_WINDOWS, BREATHING_HEIGHT


def in_any_column(x, y):
    inside = np.zeros_like(x, dtype=bool)
    for cx, cy, r, _, _ in COLUMNS:
        inside |= (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    return inside


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 closed_window_diagnostic.py <checkpoint_path>")
        sys.exit(1)
    ckpt_path = sys.argv[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GNOTOperator().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path} (iter={ckpt.get('iter', '?')}, "
          f"version={ckpt.get('version', '?')}, co2_weight={ckpt.get('co2_weight', '?')})")

    xs = np.linspace(ROOM_X[0] + 0.1, ROOM_X[1] - 0.1, 40)
    ys = np.linspace(ROOM_Y[0] + 0.1, ROOM_Y[1] - 0.1, 40)
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    xg, yg = Xg.ravel(), Yg.ravel()
    mask = in_any_column(xg, yg)
    n = xg.shape[0]

    x = torch.tensor(xg, dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
    y = torch.tensor(yg, dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
    z = torch.full((n, 1), BREATHING_HEIGHT, device=device).requires_grad_(True)
    t = torch.full((n, 1), 60.0, device=device)
    V = torch.zeros(n, NUM_WINDOWS, device=device)  # all windows CLOSED
    N_people = torch.full((n, 1), 20.0, device=device)

    A1, A2, A3, C, p = model(x, y, z, t, V, N_people)
    u, v, w = model.velocity_from_potential(A1, A2, A3, x, y, z)

    c = C.detach().cpu().numpy().ravel()
    c[mask] = np.nan
    speed = torch.sqrt(u ** 2 + v ** 2 + w ** 2).detach().cpu().numpy().ravel()
    speed[mask] = np.nan

    Cgrid = c.reshape(40, 40)
    peak_idx = np.nanargmax(Cgrid)
    pi, pj = np.unravel_index(peak_idx, Cgrid.shape)
    row = Cgrid[pi, :]  # fixed x, varying y
    col = Cgrid[:, pj]  # fixed y, varying x
    cmax, cmin = np.nanmax(Cgrid), np.nanmin(Cgrid)
    half_max = cmin + (cmax - cmin) / 2  # relative to the field's own range, not assuming a 0 baseline

    print(f"\nCO2 min/max/mean: {np.nanmin(c):.6f} / {np.nanmax(c):.6f} / {np.nanmean(c):.6f}")
    print(f"speed min/max/mean (all windows closed -- should be near 0): "
          f"{np.nanmin(speed):.6f} / {np.nanmax(speed):.6f} / {np.nanmean(speed):.6f}")
    print(f"CO2 peak at grid cell {(pi, pj)} -> (x,y) = ({xs[pi]:.2f}, {ys[pj]:.2f}) "
          f"(true source is at room center: ({(ROOM_X[0]+ROOM_X[1])/2:.2f}, {(ROOM_Y[0]+ROOM_Y[1])/2:.2f}))")
    print(f"row (fixed x={xs[pi]:.2f}, varying y) above half-max count: {np.sum(row > half_max)} / {len(row)}")
    print(f"col (fixed y={ys[pj]:.2f}, varying x) above half-max count: {np.sum(col > half_max)} / {len(col)}")
    print("\nInterpretation: if CO2 max/min are both tiny and close together (e.g. both "
          "~1e-3 in magnitude), the field is still too undertrained to show real structure. "
          "If the row/col counts are both similarly small (e.g. both under ~10/40), that's "
          "good localization. If ONE of them is much larger than the other (e.g. one near "
          "40/40), that's the room-wide 'band' artifact repeating.")


if __name__ == "__main__":
    main()
