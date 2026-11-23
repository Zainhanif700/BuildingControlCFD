"""
Overview figure, version 2026-11-23: the whole forecast of the final model for ONE test scenario in ONE image.
Top: top view of the room at 1.6 m height (windows, doors, columns drawn): simulation, forecast, difference.
Bottom: the average CO2 in the seating area over the whole run, the CO2 at one point (default: the middle of the
seating area; line: simulation, dots: forecast made 3 minutes earlier) and its position in the room. Nothing is trained here.
Usage: python3 demo_overview.py --ckpt demo/final_model --file demo/N14.npz --out demo_overview_N14.png
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common  # noqa: F401  (adds experiments/pinn to the path)
from forecast import PlaneData, predict_full, l2_rel, T_OUT, DT, H_IN
from eval_history_change import load_models
from plot_forecast import draw_room, room_legend, to_image, POINTS, OFF


def ens(models, data, t0, n_now, n_before):
    preds = [predict_full(m, data, 0, t0, n_now, n_before) for m in models]
    return (torch.stack([p[0] for p in preds]).mean(0).cpu().numpy(), preds[0][1].cpu().numpy(),
            preds[0][2].cpu().numpy())


def make_figure(xy, V, name, n_now, n_before, t_map, truth_map, fc_map, tfull, full_pts, tt, pp, err, out,
                title_prefix="Final model", marks=None, seat=None, show_points=None):
    """xy: plane points, V: 8 window speeds + 2 doors, t_map [min], truth_map / fc_map: CO2 at t_map (ppm, absolute),
    tfull / full_pts: simulated CO2 at the 4 points over time, tt / pp: forecasts made 3 min earlier,
    err: dict with the mean errors (paper / strict, model / persistence).
    marks: optional (n, 2) positions that are drawn as yellow stars (the sensors of the sensor model).
    seat: optional dict with the average CO2 in the seating area: full (simulation, every 30 s), fc (forecasts made
    3 min earlier, at tt), pers (the 'stays as it is now' forecast at tt), box ((x0, x1), (y0, y1)), ppm_model, ppm_pers."""
    fig = plt.figure(figsize=(17, 12.5 if seat is not None else 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.45 if seat is not None else 1])
    vmin, vmax = float(np.nanmin(truth_map)), float(np.nanmax(truth_map))
    dm = max(float(np.abs(fc_map - truth_map).max()), 1e-6)
    panels = [(truth_map, f"Simulation (OpenFOAM flow + CO$_2$ solver), t = {t_map:g} min", "viridis", (vmin, vmax), "CO$_2$ [ppm]"),
              (fc_map, f"Forecast of the model (made 3 min earlier), t = {t_map:g} min", "viridis", (vmin, vmax), "CO$_2$ [ppm]"),
              (fc_map - truth_map, f"Forecast $-$ simulation, t = {t_map:g} min", "RdBu_r", (-dm, dm),
               "difference [ppm] (red: too high, blue: too low)")]
    for j, (val, ttl, cm, lim, cl) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        img, ext = to_image(xy, val)
        im = ax.imshow(img, origin="lower", extent=ext, cmap=cm, vmin=lim[0], vmax=lim[1])
        fig.colorbar(im, ax=ax, label=cl, fraction=0.04, pad=0.02)
        draw_room(ax, V)
        if marks is not None:
            ax.plot(marks[:, 0], marks[:, 1], "k*", ms=9, mfc="yellow", mew=0.8)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("x: room length [m]")
        ax.set_ylabel("y: room width [m]")

    if seat is not None:
        sub = gs[1, 0:4].subgridspec(2, 1, height_ratios=[1, 1.3])
        axs = fig.add_subplot(sub[0])
        ax = fig.add_subplot(sub[1], sharex=axs)
        axs.axvspan(0, H_IN * DT / 60, color="0.9")
        axs.plot(tfull, seat["full"], "-", color="black", lw=2.5, label="simulation: average in the seating area")
        axs.plot(tt, seat["pers"], "--", color="0.5", lw=1.5, label="simple forecast: stays as it is now")
        axs.plot(tt, seat["fc"], "o", color="#d62728", ms=4, label="model: forecast made 3 min earlier")
        top = max(float(np.max(seat["full"])), float(np.max(seat["fc"])))
        for lim, col in ((700, "#e6a817"), (800, "#d62728")):
            if top > 0.85 * lim:
                axs.axhline(lim, color=col, ls=":", lw=1)
                axs.annotate(f"{lim} ppm", xy=(1.0, lim), xycoords=("axes fraction", "data"), xytext=(4, 0),
                             textcoords="offset points", va="center", ha="left", color=col, fontsize=9, annotation_clip=False)
        axs.set_ylabel("CO$_2$ [ppm]")
        axs.set_title("Average CO$_2$ in the seating area (dashed box in the position panel): "
                      f"mean error of the model {seat['ppm_model']:.1f} ppm, of the simple forecast {seat['ppm_pers']:.1f} ppm",
                      fontsize=10)
        axs.legend(fontsize=8.5, loc="best")
        axs.grid(alpha=0.3)
        axs.tick_params(labelbottom=False)
    else:
        ax = fig.add_subplot(gs[1, 0:4])
    pl = fig.add_subplot(gs[1, 4:6])
    ax.axvspan(0, H_IN * DT / 60, color="0.9", label="first 6 min: history only, no forecast yet")
    for j, (px, py) in enumerate(POINTS):
        if show_points is not None and j not in show_points:
            continue
        line, = ax.plot(tfull, full_pts[:, j], "-", lw=2, label=f"point {j + 1} (x = {px:g} m, y = {py:g} m): simulation")
        ax.plot(tt, pp[:, j], "o", ms=4, color=line.get_color())
        pl.plot(px, py, "o", ms=9, color=line.get_color(), markeredgecolor="black")
        pl.annotate(str(j + 1), (px, py), xytext=(6, 4), textcoords="offset points", fontsize=10, weight="bold")
    ax.plot([], [], "o", color="0.3", ms=4, label="dots: forecast made 3 min earlier")
    ax.set_xlabel("time after the window change [min]")
    ax.set_ylabel("CO$_2$ [ppm] (outdoor air: 400 ppm)")
    many = show_points is None or len(show_points) > 1
    ax.set_title(("CO$_2$ at four points" if many else "CO$_2$ at one point") +
                 " over the whole run: simulation (line) and 3-minute forecast (dots)", fontsize=10)
    ax.legend(fontsize=8.5, loc="best")
    ax.grid(alpha=0.3)
    draw_room(pl, V)
    if marks is not None:
        pl.plot(marks[:, 0], marks[:, 1], "k*", ms=9, mfc="yellow", mew=0.8)
    handles = room_legend()
    if seat is not None:
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        (x0, x1), (y0, y1) = seat["box"]
        pl.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", ec="black", lw=1.2))
        handles.append(Line2D([], [], color="black", ls="--", lw=1.2, label="seating area"))
    if marks is not None:
        from matplotlib.lines import Line2D
        handles.append(Line2D([], [], marker="*", ls="", markerfacecolor="yellow", markeredgecolor="black", ms=10,
                              label="sensor"))
    pl.set_title(("positions of the four points" if many else "position of the point") + " (top view, 1.6 m)", fontsize=10)
    pl.set_xlabel("x [m]")
    pl.set_ylabel("y [m]")
    pl.legend(handles=handles, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)

    wins = [round(float(v), 2) for v in V[:8]]
    fig.suptitle(
        f"{title_prefix}, test scenario {name}: forecast of the CO$_2$ in the whole room, {n_now:g} people (before the window change: "
        f"{n_before:g})\nwindow speeds {wins} m/s, doors {[int(v) for v in V[8:]]} (1 = open). Mean error over all samples "
        f"(paper definition / strict): model {100 * err['paper_model']:.2f} % / {100 * err['strict_model']:.2f} %, "
        f"persistence {100 * err['paper_pers']:.2f} % / {100 * err['strict_pers']:.2f} %", fontsize=11)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def collect(models, data, n_now, n_before, map_t0_min=6.0):
    """Forecasts of all samples whose history is complete (t0 of data.t0s) for scenario 0 of data."""
    xy = data.xyz[:, :2].cpu().numpy()
    V = [float(v) for v in data.V[0].cpu().numpy()]
    ip = [int(np.argmin((xy[:, 0] - px) ** 2 + (xy[:, 1] - py) ** 2)) for px, py in POINTS]
    (x0, x1), (y0, y1), _ = common.SEAT_BOX
    sm = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    full = (data.frames[0] * (n_now / data.N_ref[0]) + data.H[0] * (n_before / data.N_ref[0])).cpu().numpy() + OFF
    t0s = data.t0s[0]
    t_map_idx = min(max(int(round(map_t0_min * 60 / DT)), t0s[0]), t0s[-1])
    tt, pp, em, bm, es, bs = [], [], [], [], [], []
    s_fc, s_pers, s_err_m, s_err_p = [], [], [], []
    truth_map = fc_map = None
    for t0 in t0s:
        p, f, h = ens(models, data, t0, n_now, n_before)
        tt.append((t0 + T_OUT) * DT / 60)
        pp.append(p[ip, -1] + OFF)
        base = np.repeat(h[:, -1:], T_OUT, 1)
        em.append(float(l2_rel(torch.tensor(p + OFF), torch.tensor(f + OFF))))
        bm.append(float(l2_rel(torch.tensor(base + OFF), torch.tensor(f + OFF))))
        es.append(float(l2_rel(torch.tensor(p), torch.tensor(f))))
        bs.append(float(l2_rel(torch.tensor(base), torch.tensor(f))))
        s_fc.append(float(p[sm, -1].mean()) + OFF)
        s_pers.append(float(base[sm, -1].mean()) + OFF)
        s_err_m.append(float(np.abs(p[sm].mean(0) - f[sm].mean(0)).mean()))      # mean over the 6 forecast steps [ppm]
        s_err_p.append(float(np.abs(base[sm].mean(0) - f[sm].mean(0)).mean()))
        if t0 == t_map_idx:
            truth_map, fc_map = f[:, -1] + OFF, p[:, -1] + OFF
    err = {"paper_model": np.mean(em), "paper_pers": np.mean(bm), "strict_model": np.mean(es), "strict_pers": np.mean(bs)}
    seat = {"full": full[:, sm].mean(1), "fc": np.array(s_fc), "pers": np.array(s_pers), "box": ((x0, x1), (y0, y1)),
            "ppm_model": float(np.mean(s_err_m)), "ppm_pers": float(np.mean(s_err_p))}
    return {"xy": xy, "V": V, "t_map": (t_map_idx + T_OUT) * DT / 60, "truth_map": truth_map, "fc_map": fc_map,
            "tfull": np.arange(full.shape[0]) * DT / 60, "full_pts": full[:, ip], "tt": np.array(tt), "pp": np.array(pp),
            "err": err, "n_samples": len(t0s), "seat": seat}


def print_errors(name, r):
    e, s = r["err"], r["seat"]
    print(f"{name}: {r['n_samples']} samples; paper definition: model {100 * e['paper_model']:.3f} %, persistence "
          f"{100 * e['paper_pers']:.3f} %; strict: model {100 * e['strict_model']:.2f} %, persistence "
          f"{100 * e['strict_pers']:.2f} %")
    print(f"{name}: average CO2 in the seating area, mean error: model {s['ppm_model']:.1f} ppm, "
          f"simple forecast (stays as it is now, true map) {s['ppm_pers']:.1f} ppm; largest value of the simulation "
          f"{np.max(s['full']):.0f} ppm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="folder with member0.pth ... member4.pth of the final model")
    ap.add_argument("--file", required=True, help="transition file of one scenario, e.g. demo/N14.npz")
    ap.add_argument("--people", type=float, nargs=2, default=(40.0, 40.0), metavar=("NOW", "BEFORE"))
    ap.add_argument("--map-t0-min", type=float, default=6.0, help="the maps show the forecast for t0 + 3 min (default t0 = 6 min)")
    ap.add_argument("--point", type=int, default=2, choices=(0, 1, 2, 3, 4),
                    help="which of the four points to show in the lower panel (default 2, the middle of the seating area); 0 = all four")
    ap.add_argument("--out", default="demo_overview.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    data = PlaneData([args.file], dev, seed=0, z=1.6, transitions=True)
    models, hist_idx = load_models(args.ckpt, dev)
    assert torch.equal(hist_idx.cpu(), data.hist_idx.cpu()), "history points differ from the training"
    n_now, n_before = args.people
    name = os.path.splitext(os.path.basename(args.file))[0]
    r = collect(models, data, n_now, n_before, args.map_t0_min)
    make_figure(r["xy"], r["V"], name, n_now, n_before, r["t_map"], r["truth_map"], r["fc_map"], r["tfull"],
                r["full_pts"], r["tt"], r["pp"], r["err"], args.out, seat=r["seat"],
                show_points=None if args.point == 0 else (args.point - 1,))
    print_errors(name, r)
    print(f"figure saved: {args.out}")


if __name__ == "__main__":
    main()
