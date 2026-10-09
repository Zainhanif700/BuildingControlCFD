"""
Presentation figures of the trained physics-only model for several window settings.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from gnot_model import GNOTOperator, check_checkpoint_compat
from point_sampler import ROOM_X, ROOM_Y, ROOM_Z, WINDOWS, COLUMNS, NUM_WINDOWS, BREATHING_HEIGHT, T_MAX
from validate_closed_room import room_outline
from fd_reference_closed_room import SX, SY

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "milestones", "v13_fullocc", "gnot_v13_fullocc_final.pth")

N_PEOPLE = 20.0
T_FLOW = 60.0
T_CO2 = T_MAX
SCENARIOS = [
    ("all windows closed",           [0, 0, 0, 0, 0, 0, 0, 0]),
    ("W1 + W8 open, 2 m/s",          [2, 0, 0, 0, 0, 0, 0, 2]),
    ("middle window W4 open, 3 m/s", [0, 0, 0, 3, 0, 0, 0, 0]),
    ("all 8 windows open, 1 m/s",    [1, 1, 1, 1, 1, 1, 1, 1]),
]


def in_any_column(x, y):
    inside = np.zeros(np.shape(x), dtype=bool)
    for cx, cy, r, _, _ in COLUMNS:
        inside |= (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    return inside


def predict(model, device, xg, yg, zg, t, V, n_people, velocity=True, batch=1024):
    """Model outputs at points (flat numpy arrays)."""
    xg, yg, zg = (np.broadcast_to(np.asarray(a, dtype=np.float32), np.shape(xg)).ravel() for a in (xg, yg, zg))
    out = {k: [] for k in ("u", "v", "w", "c")}
    for i in range(0, len(xg), batch):
        n = len(xg[i:i + batch])
        mk = lambda a: torch.tensor(a[i:i + batch], device=device).view(-1, 1).requires_grad_(velocity)
        x, y, z = mk(xg), mk(yg), mk(zg)
        tt = torch.full((n, 1), float(t), device=device)
        VV = torch.tensor(V, dtype=torch.float32, device=device).view(1, NUM_WINDOWS).expand(n, -1)
        NN = torch.full((n, 1), float(n_people), device=device)
        with torch.set_grad_enabled(velocity):
            A1, A2, A3, C, _ = model(x, y, z, tt, VV, NN)
            if velocity:
                u, v, w = model.velocity_from_potential(A1, A2, A3, x, y, z)
                for k, a in (("u", u), ("v", v), ("w", w)):
                    out[k].append(a.detach().cpu().numpy().ravel())
        out["c"].append(C.detach().cpu().numpy().ravel())
    return {k: np.concatenate(a) if a else None for k, a in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cpu", help="cpu (default, leaves the GPU to training) or cuda")
    args = ap.parse_args()
    device = args.device
    ckpt = torch.load(args.checkpoint, map_location=device)
    check_checkpoint_compat(ckpt, args.checkpoint)
    model = GNOTOperator().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    version = ckpt.get("version", "unknown")
    out_dir = os.path.join(HERE, "figures", version, "presentation")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Checkpoint {args.checkpoint} (version={version}, iter={ckpt.get('iter', '?')}), N={N_PEOPLE:.0f}")

    xs = np.linspace(ROOM_X[0] + 0.05, ROOM_X[1] - 0.05, 78)
    ys = np.linspace(ROOM_Y[0] + 0.05, ROOM_Y[1] - 0.05, 46)
    X, Y = np.meshgrid(xs, ys)
    col = in_any_column(X, Y)
    ys_v = np.linspace(ROOM_Y[0] + 0.05, ROOM_Y[1] - 0.05, 46)
    zs_v = np.linspace(ROOM_Z[0] + 0.05, ROOM_Z[1] - 0.05, 16)
    YV, ZV = np.meshgrid(ys_v, zs_v)

    res = []
    for name, V in SCENARIOS:
        f = predict(model, device, X, Y, BREATHING_HEIGHT, T_FLOW, V, N_PEOPLE)
        c = predict(model, device, X, Y, BREATHING_HEIGHT, T_CO2, V, N_PEOPLE, velocity=False)["c"]
        open_idx = [i for i, v in enumerate(V) if v > 0]
        x_cut = 0.5 * (WINDOWS[open_idx[0]][0] + WINDOWS[open_idx[0]][1]) if open_idx else SX
        g = predict(model, device, np.full_like(YV, x_cut), YV, ZV, T_FLOW, V, N_PEOPLE)
        sh, shv = X.shape, YV.shape
        r = {"name": name, "V": V, "x_cut": x_cut,
             "u": f["u"].reshape(sh), "v": f["v"].reshape(sh),
             "speed": np.sqrt(f["u"] ** 2 + f["v"] ** 2 + f["w"] ** 2).reshape(sh),
             "c": c.reshape(sh),
             "cut_v": g["v"].reshape(shv), "cut_w": g["w"].reshape(shv),
             "cut_speed": np.sqrt(g["u"] ** 2 + g["v"] ** 2 + g["w"] ** 2).reshape(shv)}
        for k in ("speed", "c"):
            r[k][col] = np.nan
        r["u"][col] = 0.0
        r["v"][col] = 0.0
        colv = in_any_column(np.full_like(YV, x_cut), YV)
        r["cut_speed"][colv] = np.nan
        res.append(r)
        print(f"  {name:32s}: max speed {np.nanmax(r['speed']):.2f} m/s, "
              f"mean CO2 at t={T_CO2:.0f}s {np.nanmean(r['c']):.4f}")

    s_max = max(np.nanmax(r["speed"]) for r in res) or 1.0
    s_max = max(s_max, max(np.nanmax(r["cut_speed"]) for r in res))
    c_max = max(np.nanmax(r["c"]) for r in res)
    fig, axes = plt.subplots(len(res), 3, figsize=(20, 4.2 * len(res)),
                             gridspec_kw={"width_ratios": [1.6, 1.6, 1.0]})
    for row, r in zip(axes, res):
        ax = row[0]
        im0 = ax.pcolormesh(X, Y, r["speed"], cmap="plasma", vmin=0, vmax=s_max, shading="auto")
        if np.nanmax(r["speed"]) > 1e-6:
            ax.streamplot(xs, ys, r["u"], r["v"], color="w", density=1.4, linewidth=0.7, arrowsize=0.8)
        room_outline(ax)
        ax.set_title(f"{r['name']}: airflow at z={BREATHING_HEIGHT:.2f} m, t={T_FLOW:.0f} s")
        ax = row[1]
        im1 = ax.pcolormesh(X, Y, r["c"], cmap="viridis", vmin=0, vmax=c_max, shading="auto")
        room_outline(ax)
        ax.set_title(f"CO2 at z={BREATHING_HEIGHT:.2f} m, t={T_CO2:.0f} s, {N_PEOPLE:.0f} people")
        ax = row[2]
        im2 = ax.pcolormesh(YV, ZV, r["cut_speed"], cmap="plasma", vmin=0, vmax=s_max, shading="auto")
        ax.quiver(YV[::2, ::3], ZV[::2, ::3], r["cut_v"][::2, ::3], r["cut_w"][::2, ::3], color="w",
                  scale=s_max * 12, width=0.004)
        ax.set_xlim(ROOM_Y[0], ROOM_Y[1])
        ax.set_ylim(ROOM_Z[0], ROOM_Z[1])
        ax.set_aspect("equal")
        ax.set_xlabel("y [m]  (windows at right, doors at left)")
        ax.set_ylabel("z [m]")
        ax.set_title(f"vertical cut at x={r['x_cut']:.2f} m")
    fig.colorbar(im0, ax=axes[:, 0], fraction=0.02, pad=0.01, label="speed [m/s]")
    fig.colorbar(im1, ax=axes[:, 1], fraction=0.02, pad=0.01, label="excess CO2 [model units]")
    fig.colorbar(im2, ax=axes[:, 2], fraction=0.04, pad=0.02, label="speed [m/s]")
    fig.suptitle(f"One trained physics-informed GNOT ({version}), four window settings -- no retraining. "
                 f"Only the closed case is validated against a reference so far.", fontsize=13)
    f1 = os.path.join(out_dir, "scenario_overview.png")
    fig.savefig(f1, dpi=130, bbox_inches="tight")
    plt.close(fig)

    times = np.linspace(0.0, T_MAX, 13)
    xs_c, ys_c = np.meshgrid(np.linspace(ROOM_X[0] + 0.1, ROOM_X[1] - 0.1, 40),
                             np.linspace(ROOM_Y[0] + 0.1, ROOM_Y[1] - 0.1, 24))
    fluid = ~in_any_column(xs_c, ys_c)
    xc, yc = xs_c[fluid], ys_c[fluid]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    print("\n  CO2 at t=120 s relative to closed windows (source / room average):")
    base = None
    for (name, V), color in zip(SCENARIOS, ["k", "tab:blue", "tab:orange", "tab:green"]):
        src = [predict(model, device, [SX], [SY], BREATHING_HEIGHT, t, V, N_PEOPLE, velocity=False)["c"][0]
               for t in times]
        avg = [predict(model, device, xc, yc, BREATHING_HEIGHT, t, V, N_PEOPLE, velocity=False)["c"].mean()
               for t in times]
        a1.plot(times, src, "-o", ms=3, color=color, label=name)
        a2.plot(times, avg, "-o", ms=3, color=color, label=name)
        if base is None:
            base = (src[-1], avg[-1])
        print(f"    {name:32s}: {src[-1] / base[0] * 100:5.1f}% / {avg[-1] / base[1] * 100:5.1f}%")
    for a, title in ((a1, "CO2 at the source (room centre)"),
                     (a2, "room-average CO2")):
        a.set_title(f"{title}, z={BREATHING_HEIGHT:.2f} m, {N_PEOPLE:.0f} people")
        a.set_xlabel("time [s]")
        a.set_ylabel("excess CO2 [model units]")
        a.grid(alpha=0.3)
        a.legend(fontsize=9)
    fig.tight_layout()
    f2 = os.path.join(out_dir, "scenario_co2_timeseries.png")
    fig.savefig(f2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigures written to\n  {f1}\n  {f2}")


if __name__ == "__main__":
    main()
