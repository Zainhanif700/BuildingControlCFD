"""
Export a trained GNOT checkpoint's predictions (over a 3D grid, for a chosen
scenario) plus the real room geometry to a single JSON file, for the
interactive 3D viewer (viewer.html).

This is deliberately checkpoint-agnostic: point it at ANY saved checkpoint
(an intermediate one like gnot_v2_co2_fix_iter4000.pth while training is
still running, or the final one once it's done) and it produces the same
JSON shape, so the viewer never needs to change -- just re-run this and drop
in the new file.

Usage:
    python3 export_grid_for_viewer.py --checkpoint checkpoints/v2_co2_fix/gnot_v2_co2_fix_iter4000.pth --out viewer_data.json
    python3 export_grid_for_viewer.py --checkpoint checkpoints/v2_co2_fix/gnot_v2_co2_fix_final.pth --out viewer_data.json --windows 5.0,0,0,0,0,0,0,5.0 --n-people 20 --t 60
"""
import argparse
import json
import os

import numpy as np
import torch

from gnot_model import GNOTOperator, check_checkpoint_compat
from point_sampler import ROOM_X, ROOM_Y, ROOM_Z, WINDOWS, DOORS, COLUMNS, NUM_WINDOWS

HERE = os.path.dirname(os.path.abspath(__file__))


def in_any_column(x, y):
    inside = np.zeros_like(x, dtype=bool)
    for cx, cy, r, _, _ in COLUMNS:
        inside |= (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    return inside


def build_grid(nx, ny, nz):
    xs = np.linspace(ROOM_X[0] + 0.3, ROOM_X[1] - 0.3, nx)
    ys = np.linspace(ROOM_Y[0] + 0.3, ROOM_Y[1] - 0.3, ny)
    zs = np.linspace(ROOM_Z[0] + 0.2, ROOM_Z[1] - 0.2, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()
    keep = ~in_any_column(X, Y)
    return X[keep], Y[keep], Z[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to a .pth checkpoint")
    ap.add_argument("--out", default="viewer_data.json")
    ap.add_argument("--windows", default="5.0,0,0,0,0,0,0,5.0",
                     help="comma-separated V1..V8 in m/s (0 = closed)")
    ap.add_argument("--n-people", type=float, default=20.0)
    ap.add_argument("--t", type=float, default=60.0, help="seconds into the 0-120s ramp")
    ap.add_argument("--nx", type=int, default=16)
    ap.add_argument("--ny", type=int, default=10)
    ap.add_argument("--nz", type=int, default=7)
    args = ap.parse_args()

    V_list = [float(v) for v in args.windows.split(",")]
    assert len(V_list) == NUM_WINDOWS, f"expected {NUM_WINDOWS} window values, got {len(V_list)}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GNOTOperator().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    check_checkpoint_compat(ckpt, args.checkpoint)  # v8_nondim: refuse pre-v8 checkpoints
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    ckpt_iter = ckpt.get("iter", "?")
    ckpt_version = ckpt.get("version", "?")
    print(f"Loaded checkpoint: {args.checkpoint} (version={ckpt_version}, iter={ckpt_iter})")

    xg, yg, zg = build_grid(args.nx, args.ny, args.nz)
    n = xg.shape[0]
    print(f"Querying {n} grid points...")

    x = torch.tensor(xg, dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
    y = torch.tensor(yg, dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
    z = torch.tensor(zg, dtype=torch.float32, device=device).view(-1, 1).requires_grad_(True)
    t = torch.full((n, 1), args.t, device=device)
    V = torch.tensor(V_list, dtype=torch.float32, device=device).view(1, NUM_WINDOWS).expand(n, -1)
    N_people = torch.full((n, 1), args.n_people, device=device)

    A1, A2, A3, C, p = model(x, y, z, t, V, N_people)
    u, v, w = model.velocity_from_potential(A1, A2, A3, x, y, z)

    u = u.detach().cpu().numpy().ravel()
    v = v.detach().cpu().numpy().ravel()
    w = w.detach().cpu().numpy().ravel()
    c = C.detach().cpu().numpy().ravel()
    speed = np.sqrt(u ** 2 + v ** 2 + w ** 2)

    print(f"Speed: min={speed.min():.4f} max={speed.max():.4f} mean={speed.mean():.4f} m/s")
    print(f"CO2:   min={c.min():.4f} max={c.max():.4f} mean={c.mean():.4f}")

    data = {
        "meta": {
            "checkpoint": os.path.basename(args.checkpoint),
            "checkpoint_version": ckpt_version,
            "checkpoint_iter": ckpt_iter,
            "scenario": {"V": V_list, "n_people": args.n_people, "t": args.t},
        },
        "room": {"x": list(ROOM_X), "y": list(ROOM_Y), "z": list(ROOM_Z)},
        "windows": [{"x_lo": w0, "x_hi": w1, "z_lo": w2, "z_hi": w3, "v": V_list[i]}
                    for i, (w0, w1, w2, w3) in enumerate(WINDOWS)],
        "doors": [{"x_lo": d0, "x_hi": d1, "z_lo": d2, "z_hi": d3} for (d0, d1, d2, d3) in DOORS],
        "columns": [{"cx": cx, "cy": cy, "r": r, "z_lo": zlo, "z_hi": zhi}
                    for (cx, cy, r, zlo, zhi) in COLUMNS],
        "grid": {
            "x": xg.round(3).tolist(), "y": yg.round(3).tolist(), "z": zg.round(3).tolist(),
            "u": u.round(4).tolist(), "v": v.round(4).tolist(), "w": w.round(4).tolist(),
            "speed": speed.round(4).tolist(), "co2": c.round(5).tolist(),
        },
    }

    out_path = os.path.join(HERE, args.out)
    with open(out_path, "w") as f:
        json.dump(data, f)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Saved: {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
