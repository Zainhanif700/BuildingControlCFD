"""
Figures for the test cases: CO2 maps (CFD, prediction, difference) and CO2 over time at 4 points.
Usage: python3 plot_forecast.py --tag rans_tr --data-dir data_transitions --transitions --z 1.6
"""
import argparse
import glob
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
from forecast import ForecastGNOT, PlaneData, predict_full, l2_rel, H_IN, T_OUT, DT
from train_forecast import split, SCEN
from point_sampler import ROOM_X, ROOM_Y, WINDOWS, DOORS, COLUMNS

HERE = os.path.dirname(os.path.abspath(__file__))
OFF = 400.0
POINTS = [(3.0, 2.5), (7.8, 4.6), (12.0, 6.5), (14.0, 2.0)]


def load_models(tag, dev):
    paths = sorted(glob.glob(os.path.join(HERE, "checkpoints", tag, "member*.pth")))
    if not paths:
        raise SystemExit(f"no checkpoints in checkpoints/{tag}")
    models, hidx = [], None
    for p in paths:
        ck = torch.load(p, map_location=dev)
        m = ForecastGNOT(d=ck["d"], layers=ck["layers"], c_scale=ck["c_scale"], d_scale=ck["d_scale"]).to(dev)
        m.load_state_dict(ck["state"])
        m.eval()
        models.append(m)
        hidx = ck["hist_idx"].to(dev)
    return models, hidx


def to_image(xy, val):
    xs, ys = np.unique(np.round(xy[:, 0], 4)), np.unique(np.round(xy[:, 1], 4))
    img = np.full((len(ys), len(xs)), np.nan)
    ix = np.searchsorted(xs, np.round(xy[:, 0], 4))
    iy = np.searchsorted(ys, np.round(xy[:, 1], 4))
    img[iy, ix] = val
    dx = xs[1] - xs[0]
    return img, (xs[0] - dx / 2, xs[-1] + dx / 2, ys[0] - dx / 2, ys[-1] + dx / 2)


def ens(models, data, k, t0, N, NA):
    preds = [predict_full(m, data, k, t0, N, NA) for m in models]
    return torch.stack([p[0] for p in preds]).mean(0).cpu().numpy(), preds[0][1].cpu().numpy(), preds[0][2].cpu().numpy()


def draw_room(ax, V):
    """Room outline with open/closed windows (top wall), doors (bottom wall) and columns."""
    ax.plot([ROOM_X[0], ROOM_X[1], ROOM_X[1], ROOM_X[0], ROOM_X[0]], [ROOM_Y[0], ROOM_Y[0], ROOM_Y[1], ROOM_Y[1], ROOM_Y[0]],
            color="black", lw=1)
    for k, (a, b, _, _) in enumerate(WINDOWS):
        ax.plot([a, b], [ROOM_Y[1]] * 2, color="deepskyblue" if V[k] > 0 else "0.55", lw=5, solid_capstyle="butt")
    for a, b, _, _ in DOORS:
        ax.plot([a, b], [ROOM_Y[0]] * 2, color="darkorange", lw=5, solid_capstyle="butt")
    for cx, cy, r, _, _ in COLUMNS:
        ax.add_patch(plt.Circle((cx, cy), r, facecolor="white", edgecolor="black", lw=0.8))
    ax.set_xlim(ROOM_X[0] - 0.3, ROOM_X[1] + 0.3); ax.set_ylim(ROOM_Y[0] - 0.4, ROOM_Y[1] + 0.4)
    ax.set_aspect("equal")


def room_legend():
    from matplotlib.lines import Line2D
    return [Line2D([], [], color="deepskyblue", lw=5, label="open window"),
            Line2D([], [], color="0.55", lw=5, label="closed window"),
            Line2D([], [], color="darkorange", lw=5, label="door"),
            Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="black", ms=8, label="column")]


def plot_maps(rows, xy, name, z, N, NA, V, out):
    """Top view of the room: simulation, forecast and difference, one row per time."""
    fig, ax = plt.subplots(len(rows), 3, figsize=(16, 4.3 * len(rows)), squeeze=False)
    for r, (tmin, tr, pr, e) in enumerate(rows):
        vmin, vmax = np.nanmin(tr), np.nanmax(tr)
        dm = max(float(np.abs(pr - tr).max()), 1e-6)
        panels = [(tr, f"Simulation (CFD), t = {tmin:g} min", "viridis", (vmin, vmax), "CO$_2$ [ppm]"),
                  (pr, f"Forecast (made 3 min earlier), t = {tmin:g} min\nerror {100 * e:.2f} %", "viridis", (vmin, vmax),
                   "CO$_2$ [ppm]"),
                  (pr - tr, f"Forecast $-$ simulation, t = {tmin:g} min", "RdBu_r", (-dm, dm), "difference [ppm]")]
        for j, (val, ttl, cm, lim, cl) in enumerate(panels):
            img, ext = to_image(xy, val)
            im = ax[r, j].imshow(img, origin="lower", extent=ext, cmap=cm, vmin=lim[0], vmax=lim[1])
            plt.colorbar(im, ax=ax[r, j], label=cl, fraction=0.035, pad=0.03)
            draw_room(ax[r, j], V)
            ax[r, j].set_title(ttl, fontsize=11)
            ax[r, j].set_xlabel("x: room length [m]"); ax[r, j].set_ylabel("y: room width [m]")
    people = f"{N:g} people" + (f" (before the window change: {NA:g})" if NA is not None else "")
    fig.suptitle(f"Test setting {name}: CO$_2$ at 1.6 m height (grid layer at {z:.2f} m), top view of the room, {people}\n"
                 "windows on the top wall, doors on the bottom wall; red = forecast too high, blue = too low",
                 fontsize=12)
    fig.legend(handles=room_legend(), loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_series(tfull, cfd, tt, pp, err, base, name, N, V, out):
    """CO2 over time at the points of POINTS: simulation (lines) and the forecast made 3 min earlier (dots)."""
    fig = plt.figure(figsize=(13, 5.2))
    ax = fig.add_axes((0.07, 0.12, 0.63, 0.76))
    pl = fig.add_axes((0.73, 0.18, 0.26, 0.64))
    ax.axvspan(0, H_IN * DT / 60, color="0.9", label="first 6 min: history only, no forecast yet")
    for j, (px, py) in enumerate(POINTS):
        l, = ax.plot(tfull, cfd[:, j], "-", lw=2, label=f"point {j + 1} (x = {px:g} m, y = {py:g} m): simulation")
        ax.plot(tt, pp[:, j], "o", ms=4, color=l.get_color())
        pl.plot(px, py, "o", ms=9, color=l.get_color(), markeredgecolor="black")
        pl.annotate(str(j + 1), (px, py), xytext=(6, 4), textcoords="offset points", fontsize=10, weight="bold")
    ax.plot([], [], "o", color="0.3", ms=4, label="dots: forecast made 3 min earlier")
    ax.set_xlabel("time after the window change [min]")
    ax.set_ylabel("CO$_2$ [ppm] (outdoor air: 400 ppm)")
    ax.set_title(f"Test setting {name}, {N:g} people: simulation (lines) vs. 3-minute forecast (dots)\n"
                 f"forecast error {100 * err:.2f} %, 'no change' guess {100 * base:.2f} %", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower right"); ax.grid(alpha=0.3)
    draw_room(pl, V)
    pl.set_title("positions in the room (top view, 1.6 m)", fontsize=9)
    pl.set_xlabel("x [m]", fontsize=8); pl.set_ylabel("y [m]", fontsize=8); pl.tick_params(labelsize=7)
    pl.legend(handles=room_legend(), fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--scenarios", default=SCEN)
    ap.add_argument("--transitions", action="store_true")
    ap.add_argument("--z", type=float, default=None)
    ap.add_argument("--case", default=None, help="test case name; default all test cases")
    ap.add_argument("--t0-min", default="6,15", help="start times of the forecast [min] for the maps")
    ap.add_argument("--N", type=float, default=45.0, help="people now")
    ap.add_argument("--NA", type=float, default=45.0, help="people before (transition data)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    _, tep = split(args.data_dir, args.scenarios, False)
    if args.case:
        tep = [p for p in tep if os.path.basename(p)[:-4] == args.case]
        if not tep:
            raise SystemExit(f"{args.case} is not a test case")
    models, hidx = load_models(args.tag, dev)
    data = PlaneData(tep, dev, seed=0, z=args.z, transitions=args.transitions)
    data.hist_idx = hidx
    NA = args.NA if args.transitions else 0.0
    xy = data.xyz[:, :2].cpu().numpy()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    t0s_map = [int(round(float(m) * 60 / DT)) for m in args.t0_min.split(",")]

    for k, c in enumerate(data.cases):
        name = c["name"][:-4]
        lo, hi = data.t0s[k][0], data.t0s[k][-1]
        t0s = [min(max(t, lo), hi) for t in t0s_map]

        rows = []
        for t0 in t0s:
            p, f, _ = ens(models, data, k, t0, args.N, NA)
            e = l2_rel(torch.tensor(p + OFF), torch.tensor(f + OFF))
            rows.append(((t0 + T_OUT) * DT / 60, f[:, -1] + OFF, p[:, -1] + OFF, e))
        V = [float(v) for v in c["V"]]
        out1 = os.path.join(HERE, "figures", f"{args.tag}_{name}_maps.png")
        plot_maps(rows, xy, name, c["z"], args.N, NA if args.transitions else None, V, out1)

        ip = [int(np.argmin((xy[:, 0] - px) ** 2 + (xy[:, 1] - py) ** 2)) for px, py in POINTS]
        full = (data.frames[k] * (args.N / data.N_ref[k])
                + (data.H[k] * (NA / data.N_ref[k]) if data.H is not None else 0)).cpu().numpy() + OFF
        tt, pp, ee, bb = [], [], [], []
        for t0 in data.t0s[k]:
            p, f, h = ens(models, data, k, t0, args.N, NA)
            tt.append((t0 + T_OUT) * DT / 60); pp.append(p[ip, -1] + OFF)
            ee.append(l2_rel(torch.tensor(p + OFF), torch.tensor(f + OFF)))
            bb.append(l2_rel(torch.tensor(np.repeat(h[:, -1:], T_OUT, 1) + OFF), torch.tensor(f + OFF)))
        out2 = os.path.join(HERE, "figures", f"{args.tag}_{name}_series.png")
        plot_series(np.arange(full.shape[0]) * DT / 60, full[:, ip], tt, np.array(pp), np.mean(ee), np.mean(bb),
                    name, args.N, V, out2)
        print(f"{name}: l2 (paper metric) {100 * np.mean(ee):.3f} %, persistence {100 * np.mean(bb):.3f} %  -> {out1}, {out2}",
              flush=True)


if __name__ == "__main__":
    main()
