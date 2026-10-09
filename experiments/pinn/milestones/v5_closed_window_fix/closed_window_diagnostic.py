"""
=== MILESTONE SNAPSHOT: v5_closed_window_fix (2026-09-25) ===
STATUS: fix #3 (spurious closed-window velocity artifact) is CONFIRMED FIXED
and validated across checkpoints iter10000 through iter20000 (closed-window
speed dropped from ~0.3-0.6 m/s to ~0.02-0.10 m/s; open-window airflow still
behaves correctly, no regression).

CO2 source-localization is NOT solved as of this snapshot -- the CO2 field
still does not localize around the true source (it looks like a near-uniform
haze, and its overall magnitude oscillates between checkpoints rather than
settling, most likely due to the constant learning rate with no decay
schedule). See README.md in this folder for the full diagnostic trail,
checkpoint-by-checkpoint evidence, and recommended next steps (adding an LR
decay schedule).

This file is a COPY for reference/reproducibility of this specific milestone.
Ongoing CO2 work continues in the live experiments/gnot/ files, not here --
do not edit this copy.
=== END MILESTONE HEADER ===
"""

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

Two scenarios (pass as a second CLI argument, defaults to "closed"):
  closed -- all 8 windows at V=0. True physical solution for velocity is
            exactly u=v=w=0 everywhere (zero forcing, zero IC, no body-force
            term) -- any large speed here is the "spurious closed-window
            velocity" bug (fix #3). CO2 should show a compact, localized
            bump near the room center at breathing height.
  open   -- all 8 windows at V=V_MAX (5 m/s), the "normal airflow" regime.
            Added as a companion check after fix #3 (correlated closed/
            partial-closed scenario oversampling in point_sampler.py) --
            since that fix reduces the fraction of purely-uniform-sampled
            training points from 100% to 50%, this checks the open-window
            regime hasn't regressed as a side effect. Expect clearly
            non-zero, inflow-directed velocity here (unlike the closed case).

Usage:
    python3 closed_window_diagnostic.py <checkpoint_path> [closed|open]
"""
import sys
import torch
import numpy as np

from gnot_model import GNOTOperator
from point_sampler import ROOM_X, ROOM_Y, ROOM_Z, COLUMNS, NUM_WINDOWS, BREATHING_HEIGHT, V_MAX


def in_any_column(x, y):
    inside = np.zeros_like(x, dtype=bool)
    for cx, cy, r, _, _ in COLUMNS:
        inside |= (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    return inside


def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: python3 closed_window_diagnostic.py <checkpoint_path> [closed|open]")
        sys.exit(1)
    ckpt_path = sys.argv[1]
    scenario = sys.argv[2] if len(sys.argv) == 3 else "closed"
    if scenario not in ("closed", "open"):
        print(f"Unknown scenario '{scenario}', expected 'closed' or 'open'")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GNOTOperator().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path} (iter={ckpt.get('iter', '?')}, "
          f"version={ckpt.get('version', '?')}, co2_weight={ckpt.get('co2_weight', '?')})")
    print(f"Scenario: {scenario}")

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
    if scenario == "closed":
        V = torch.zeros(n, NUM_WINDOWS, device=device)
    else:
        V = torch.full((n, NUM_WINDOWS), V_MAX, device=device)
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
    if scenario == "closed":
        print(f"speed min/max/mean (all windows closed -- should be near 0): "
              f"{np.nanmin(speed):.6f} / {np.nanmax(speed):.6f} / {np.nanmean(speed):.6f}")
    else:
        print(f"speed min/max/mean (all windows at V_MAX={V_MAX} -- should be clearly non-zero, "
              f"inflow-directed): {np.nanmin(speed):.6f} / {np.nanmax(speed):.6f} / {np.nanmean(speed):.6f}")
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
