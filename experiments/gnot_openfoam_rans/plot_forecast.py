"""
Figures for a trained forecast ensemble (train_forecast.py) on one TEST case:
  1. maps on the plane: CFD | prediction | difference, 3 min ahead, at chosen start times
  2. CO2 over time at 4 points: CFD vs the 3-min-ahead prediction (rolling forecast over 30 min)
CO2 shown as absolute ppm (400 + excess). Transition data: N_A people before, N people now.
Usage (training env, from experiments/gnot_openfoam_rans):
  python3 plot_forecast.py --tag rans_tr --data-dir data_transitions --transitions --z 1.6
  (--case S22 --t0-min 6,15 --N 45 --NA 45; default case = every test case)
Output: figures/<tag>_<case>_maps.png and _series.png, errors printed.
"""
import argparse
import glob
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common  # noqa: F401
from forecast import ForecastGNOT, PlaneData, predict_full, l2_rel, H_IN, T_OUT, DT
from train_forecast import split, SCEN

HERE = os.path.dirname(os.path.abspath(__file__))
OFF = 400.0
POINTS = [(3.0, 2.5), (7.8, 4.6), (12.0, 6.5), (14.0, 2.0)]     # [m], for the time series


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

        # 1. maps, 3 min ahead
        fig, ax = plt.subplots(len(t0s), 3, figsize=(15, 3.6 * len(t0s)), squeeze=False)
        for r, t0 in enumerate(t0s):
            p, f, _ = ens(models, data, k, t0, args.N, NA)
            tr, pr = f[:, -1] + OFF, p[:, -1] + OFF
            vmin, vmax = np.nanmin(tr), np.nanmax(tr)
            e = l2_rel(torch.tensor(p + OFF), torch.tensor(f + OFF))
            dm = max(float(np.abs(pr - tr).max()), 1e-6)
            for j, (val, ttl, cm, lim) in enumerate([
                    (tr, "CFD", "viridis", (vmin, vmax)), (pr, "prediction", "viridis", (vmin, vmax)),
                    (pr - tr, "prediction - CFD", "RdBu_r", (-dm, dm))]):
                img, ext = to_image(xy, val)
                im = ax[r, j].imshow(img, origin="lower", extent=ext, cmap=cm, vmin=lim[0], vmax=lim[1])
                plt.colorbar(im, ax=ax[r, j], label="ppm")
                tt = (t0 + T_OUT) * DT / 60
                ax[r, j].set_title(f"{ttl}, t = {tt:g} min" + (f"  (l2 {100 * e:.2f} %)" if j == 1 else ""))
                ax[r, j].set_xlabel("x [m]"); ax[r, j].set_ylabel("y [m]")
        fig.suptitle(f"{name}: CO2 at z = {c['z']:.2f} m, 3 min ahead (from the last 6 min), N = {args.N:g}"
                     + (f", previous state N = {NA:g}" if args.transitions else ""))
        fig.tight_layout()
        out1 = os.path.join(HERE, "figures", f"{args.tag}_{name}_maps.png")
        fig.savefig(out1, dpi=130); plt.close(fig)

        # 2. rolling 3-min-ahead forecast at 4 points
        ip = [int(np.argmin((xy[:, 0] - px) ** 2 + (xy[:, 1] - py) ** 2)) for px, py in POINTS]
        full = (data.frames[k] * (args.N / data.N_ref[k])
                + (data.H[k] * (NA / data.N_ref[k]) if data.H is not None else 0)).cpu().numpy() + OFF
        tt, pp, ee, bb = [], [], [], []
        for t0 in data.t0s[k]:
            p, f, h = ens(models, data, k, t0, args.N, NA)
            tt.append((t0 + T_OUT) * DT / 60); pp.append(p[ip, -1] + OFF)
            ee.append(l2_rel(torch.tensor(p + OFF), torch.tensor(f + OFF)))
            bb.append(l2_rel(torch.tensor(np.repeat(h[:, -1:], T_OUT, 1) + OFF), torch.tensor(f + OFF)))
        pp = np.array(pp)
        tfull = np.arange(full.shape[0]) * DT / 60
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for j, (px, py) in enumerate(POINTS):
            l, = ax.plot(tfull, full[:, ip[j]], "-", lw=2, label=f"CFD ({px:g}, {py:g}) m")
            ax.plot(tt, pp[:, j], "o", ms=4, color=l.get_color(), label="prediction 3 min ahead (dots)" if j == 0 else None)
        ax.axvspan(0, H_IN * DT / 60, color="0.9", label="first 6 min (history only)")
        ax.set_xlabel("time [min]"); ax.set_ylabel("CO2 [ppm]")
        ax.set_title(f"{name}: CFD vs 3-min-ahead forecast (mean l2 {100 * np.mean(ee):.2f} %, "
                     f"'stays the same' {100 * np.mean(bb):.2f} %)")
        ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
        fig.tight_layout()
        out2 = os.path.join(HERE, "figures", f"{args.tag}_{name}_series.png")
        fig.savefig(out2, dpi=130); plt.close(fig)
        print(f"{name}: l2 (paper metric) {100 * np.mean(ee):.3f} %, persistence {100 * np.mean(bb):.3f} %  -> {out1}, {out2}",
              flush=True)


if __name__ == "__main__":
    main()
