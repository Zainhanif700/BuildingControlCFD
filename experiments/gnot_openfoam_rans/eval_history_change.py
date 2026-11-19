"""
Test of the forecast model when the window setting changes INSIDE the 6-minute history.

In training every history window lies completely after the change of the window setting (a transition run
starts at the change, and the first sample uses maps 0..11). In the room a window can be opened or closed at
any time, so the history can contain the old state and the new state. Here I build such samples by hand
from the transition files of the test scenarios:
  - maps before the change (index < 0) = the old state, which is the map at the change (index 0) repeated
    (the old CO2 of the previous setting is assumed constant before the change, an approximation);
  - maps from the change on = the transition run of the new setting;
  - the model gets the new window setting.
t0 = 0..10 are the samples with the change inside the history (t0 = 0: the change is at the last history map).
t0 = 11..16 are normal samples (the history is completely after the change) and serve as the reference.

Usage (final model of the thesis, on the lab PC):
  python3 eval_history_change.py --ckpt checkpoints/v4_tr --data-dir data_transitions_v4 --scenarios scenarios_v4.txt
"""
import argparse
import os

import numpy as np
import torch

import common
import forecast as F
from forecast import ForecastGNOT, PlaneData, l2_rel
from train_forecast import split

HERE = os.path.dirname(os.path.abspath(__file__))


def load_models(ckpt_dir, dev):
    models, hist_idx = [], None
    for name in sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pth")):
        c = torch.load(os.path.join(ckpt_dir, name), map_location=dev)
        m = ForecastGNOT(d=c["d"], layers=c["layers"], c_scale=c["c_scale"], d_scale=c["d_scale"],
                         open_flags=c.get("open_flags", False)).to(dev)
        m.load_state_dict(c["state"])
        m.eval()
        models.append(m)
        hist_idx = c["hist_idx"]
    assert models, f"no .pth files in {ckpt_dir}"
    return models, hist_idx


def sample_padded(data, k, t0, N, NA):
    """History of 12 maps ending at map t0 (maps before the change are the map at the change) and the next 6 maps."""
    f = data.frames[k] * (N / data.N_ref[k]) + data.H[k] * (NA / data.N_ref[k])
    hist = torch.stack([f[max(i, 0)] for i in range(t0 - F.H_IN + 1, t0 + 1)], 0).T
    fut = f[t0 + 1:t0 + 1 + F.T_OUT].T
    return hist, fut


@torch.no_grad()
def predict(models, data, k, hist, N, chunk=4096):
    hh, hx = hist[data.hist_idx].unsqueeze(0), data.xyz[data.hist_idx].unsqueeze(0)
    V = data.V[k].unsqueeze(0)
    Nt = torch.tensor([[N]], device=data.dev)
    outs = []
    for m in models:
        out = []
        for i in range(0, hist.shape[0], chunk):
            mean, _ = m(data.xyz[i:i + chunk].unsqueeze(0), hist[i:i + chunk].unsqueeze(0), hx, hh, V, Nt)
            out.append(mean[0])
        outs.append(torch.cat(out))
    return torch.stack(outs).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="folder with member*.pth of the final transition model")
    ap.add_argument("--data-dir", required=True, help="transition files, e.g. data_transitions_v4")
    ap.add_argument("--scenarios", default=os.path.join(HERE, "scenarios_v4.txt"))
    ap.add_argument("--draws", type=int, default=3, help="random people numbers (before, now) per case and t0")
    ap.add_argument("--t0-max", type=int, default=16)
    ap.add_argument("--offset", type=float, default=400.0, help="outdoor level added for the paper's metric")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    _, tep = split(args.data_dir, args.scenarios, False)
    te = PlaneData(tep, dev, seed=0, z=1.6, transitions=True)
    models, hist_idx = load_models(args.ckpt, dev)
    assert torch.equal(hist_idx.cpu(), te.hist_idx.cpu()), "history points differ from the training"
    print(f"{len(models)} networks, {len(te.cases)} test scenarios, {te.xyz.shape[0]} plane points, device {dev}", flush=True)

    rng = np.random.default_rng(1234)
    rows = {}
    for k in range(len(te.cases)):
        for t0 in range(0, args.t0_max + 1):
            for _ in range(args.draws):
                n_now, n_before = (float(v) for v in rng.uniform(10.0, 80.0, 2))
                hist, fut = sample_padded(te, k, t0, n_now, n_before)
                if torch.linalg.norm(fut) == 0:
                    continue
                pred = predict(models, te, k, hist, n_now)
                base = hist[:, -1:].expand_as(fut)
                r = rows.setdefault(t0, {"m": [], "p": [], "mo": [], "po": []})
                r["m"].append(l2_rel(pred, fut))
                r["p"].append(l2_rel(base, fut))
                r["mo"].append(l2_rel(pred + args.offset, fut + args.offset))
                r["po"].append(l2_rel(base + args.offset, fut + args.offset))

    print("\nt0 = last history map after the change; the change is inside the history for t0 = 0..10")
    print("error above the outdoor level (strict) / paper metric (+400 ppm); model | persistence")
    print(f"{'t0':>3} {'min since change':>17} {'in history':>11} {'strict model':>13} {'strict pers.':>13} "
          f"{'paper model':>12} {'paper pers.':>12} {'n':>5}")
    for t0 in sorted(rows):
        r = rows[t0]
        inside = "change in" if t0 < F.H_IN - 1 else "normal"
        print(f"{t0:3d} {t0 * F.DT / 60:17.1f} {inside:>11} {100 * np.mean(r['m']):12.2f}% {100 * np.mean(r['p']):12.2f}% "
              f"{100 * np.mean(r['mo']):11.3f}% {100 * np.mean(r['po']):11.3f}% {len(r['m']):5d}")

    def avg(keys, key):
        return 100 * np.mean([v for t0 in keys for v in rows[t0][key]])

    chg = [t for t in rows if t < F.H_IN - 1]
    nor = [t for t in rows if t >= F.H_IN - 1]
    print("\nmean over the samples with the change inside the history (t0 0..10):")
    print(f"  strict: model {avg(chg, 'm'):.2f}%  persistence {avg(chg, 'p'):.2f}%   |  paper: model {avg(chg, 'mo'):.3f}%  persistence {avg(chg, 'po'):.3f}%")
    print("mean over the normal samples of the same runs (t0 11..%d):" % args.t0_max)
    print(f"  strict: model {avg(nor, 'm'):.2f}%  persistence {avg(nor, 'p'):.2f}%   |  paper: model {avg(nor, 'mo'):.3f}%  persistence {avg(nor, 'po'):.3f}%")
    print("Assumption: the CO2 before the change is the map at the change, repeated (no real simulation of the old state).")


if __name__ == "__main__":
    main()
