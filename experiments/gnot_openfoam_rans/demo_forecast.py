"""
Demo, version 2026-11-23: the final model of the thesis (ensemble of 5 networks, a folder with member*.pth) forecasts
the CO2 of ONE test scenario. Nothing is trained here.
For every t0 (the last map of the 6-minute history, counted from the window change) the model gets the history and
forecasts the next 3 minutes. The error is compared with "the CO2 stays as it is now" (persistence), in both
definitions: strict (CO2 above the outdoor level) and the paper's definition (+400 ppm outdoor level).
t0 = 0..10: the window change is INSIDE the history (the CO2 before the change is assumed constant);
t0 = 11..16: normal samples, the history is completely after the change.
Usage: python3 demo_forecast.py --ckpt demo/final_model --file demo/N14.npz [--plot demo_forecast.png]
"""
import argparse

import numpy as np
import torch

import common  # noqa: F401  (adds experiments/pinn to the path)
import forecast as F
from forecast import PlaneData, l2_rel
from eval_history_change import load_models, sample_padded, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="folder with member0.pth ... member4.pth of the final model")
    ap.add_argument("--file", required=True, help="transition file of one scenario, e.g. demo/N14.npz")
    ap.add_argument("--offset", type=float, default=400.0, help="outdoor level added for the paper's definition [ppm]")
    ap.add_argument("--people", type=float, nargs=2, default=(40.0, 40.0), metavar=("NOW", "BEFORE"),
                    help="number of people now and before the window change (default 40 and 40; the model was "
                         "trained with 10 to 80)")
    ap.add_argument("--t0-max", type=int, default=16)
    ap.add_argument("--plot", default=None, help="optional: save a figure (CO2 map after 3 min: simulation, forecast)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    data = PlaneData([args.file], dev, seed=0, z=1.6, transitions=True)
    models, hist_idx = load_models(args.ckpt, dev)
    assert torch.equal(hist_idx.cpu(), data.hist_idx.cpu()), "history points differ from the training"
    n_now, n_before = args.people
    print(f"{len(models)} networks, file {args.file}, {data.xyz.shape[0]} plane points, device {dev}, "
          f"people now {n_now:g}, before {n_before:g}", flush=True)

    rows = {}
    keep = {}
    for t0 in range(0, args.t0_max + 1):
        hist, fut = sample_padded(data, 0, t0, n_now, n_before)
        if torch.linalg.norm(fut) == 0:
            continue
        pred = predict(models, data, 0, hist, n_now, t0)
        base = hist[:, -1:].expand_as(fut)
        o = args.offset
        rows[t0] = (float(l2_rel(pred, fut)), float(l2_rel(base, fut)),
                    float(l2_rel(pred + o, fut + o)), float(l2_rel(base + o, fut + o)))
        keep[t0] = (pred, fut)

    print("\nerror of the 3-minute forecast; model | persistence (the CO2 stays as it is now)")
    print(f"{'t0':>3} {'min since change':>17} {'in history':>11} {'strict model':>13} {'strict pers.':>13} "
          f"{'paper model':>12} {'paper pers.':>12}")
    for t0, r in rows.items():
        inside = "change in" if t0 < F.H_IN - 1 else "normal"
        print(f"{t0:3d} {t0 * F.DT / 60:17.1f} {inside:>11} {100 * r[0]:12.2f}% {100 * r[1]:12.2f}% "
              f"{100 * r[2]:11.3f}% {100 * r[3]:11.3f}%")
    for name, sel in (("change inside the history (t0 0..10)", [t for t in rows if t < F.H_IN - 1]),
                      ("normal samples (t0 11..%d)" % args.t0_max, [t for t in rows if t >= F.H_IN - 1])):
        if sel:
            m = lambda i: 100 * np.mean([rows[t][i] for t in sel])
            print(f"mean, {name}: strict model {m(0):.2f}% persistence {m(1):.2f}%  |  "
                  f"paper model {m(2):.3f}% persistence {m(3):.3f}%")
    print("Only one scenario: this is a demo, not the test result of the thesis (16 test scenarios).")

    if args.plot and keep:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            t0 = max(t for t in keep if t >= F.H_IN - 1) if any(t >= F.H_IN - 1 for t in keep) else max(keep)
            pred, fut = keep[t0]
            xy = data.xyz[:, :2].cpu().numpy()
            sim, fc = fut[:, -1].cpu().numpy(), pred[:, -1].cpu().numpy()
            import os
            vmax = float(max(sim.max(), fc.max()))
            dmax = max(0.2 * vmax, 1e-6)
            Vv = data.V[0].cpu().numpy()
            name = os.path.splitext(os.path.basename(args.file))[0]
            fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
            for a, v, title in ((ax[0], sim, "Simulation (reference)"), (ax[1], fc, "Forecast of the model")):
                s = a.scatter(xy[:, 0], xy[:, 1], c=v, s=3, vmin=0, vmax=vmax, cmap="viridis")
                a.set_title(title)
            cb = fig.colorbar(s, ax=ax[:2], shrink=0.85, pad=0.02)
            cb.set_label("CO2 above the outdoor level [ppm]")
            d = ax[2].scatter(xy[:, 0], xy[:, 1], c=fc - sim, s=3, cmap="coolwarm", vmin=-dmax, vmax=dmax)
            ax[2].set_title(f"Forecast minus simulation (largest difference {float(np.abs(fc - sim).max()):.2f} ppm)")
            cb2 = fig.colorbar(d, ax=ax[2], shrink=0.85, pad=0.02)
            cb2.set_label("difference [ppm] (red: forecast too high, blue: too low)")
            for a in ax:
                a.set_aspect("equal")
                a.set_xlabel("x [m]")
                a.set_ylabel("y [m]")
            fig.suptitle(
                f"Scenario {name}: CO2 on the plane at 1.6 m height, 3 minutes ahead (history ends {t0 * F.DT / 60:.1f} min "
                f"after the window change)\nwindow speeds {[round(float(v), 2) for v in Vv[:8]]} m/s, doors {[int(v) for v in Vv[8:]]}"
                f" (1 = open); error of this forecast {100 * rows[t0][0]:.2f} % (persistence {100 * rows[t0][1]:.2f} %); "
                f"white spots = columns", fontsize=11)
            fig.savefig(args.plot, dpi=130)
            print(f"figure saved: {args.plot}")
        except Exception as e:  # the figure is optional
            print(f"figure skipped: {e}")


if __name__ == "__main__":
    main()
