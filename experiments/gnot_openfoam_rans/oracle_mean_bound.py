"""
Upper bound for a "mixing-box" fix of the first minutes after a window change (no model, CPU, about 1 minute).

Samples as in eval_history_change.py (window change inside the history, t0 = 0..10). For each sample three forecasts:
  persistence       : the last history map stays as it is
  perfect box model : persistence + the TRUE average change of the whole plane at each future time (a uniform shift)
  perfect scaling   : the last history map times the best single factor per future time (a perfect decay of the old pattern)
A real mixing-box formula cannot be better than the perfect box model, because it knows less. If even the perfect
versions stay far from 0, a box formula cannot fix the first minutes, because the error is in the spatial pattern.

Usage: python3 oracle_mean_bound.py --data-dir data_transitions_v4 --scenarios scenarios_v4.txt [--which test]
"""
import argparse
import os

import numpy as np
import torch

import common
import forecast as F
from forecast import PlaneData, l2_rel
from train_forecast import split
from eval_history_change import sample_padded

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--scenarios", default=os.path.join(HERE, "scenarios_v4.txt"))
    ap.add_argument("--which", choices=["test", "train"], default="test")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--offset", type=float, default=400.0)
    args = ap.parse_args()
    trp, tep = split(args.data_dir, args.scenarios, False)
    paths = tep if args.which == "test" else trp
    if args.max_cases:
        paths = paths[:args.max_cases]
    data = PlaneData(paths, torch.device("cpu"), seed=0, z=1.6, transitions=True)
    rng = np.random.default_rng(1234)    # same draws as eval_history_change.py
    rows = {}
    for k in range(len(data.cases)):
        for t0 in range(0, 11):
            for _ in range(args.draws):
                n_now, n_before = (float(v) for v in rng.uniform(10.0, 80.0, 2))
                hist, fut = sample_padded(data, k, t0, n_now, n_before)
                if torch.linalg.norm(fut) == 0:
                    continue
                base = hist[:, -1:].expand_as(fut)
                box = base + (fut - base).mean(0, keepdim=True)
                a = (fut * base).sum(0, keepdim=True) / (base * base).sum(0, keepdim=True).clamp_min(1e-30)
                scale = base * a
                r = rows.setdefault(t0, {k_: [] for k_ in ("p", "b", "s", "po", "bo", "so")})
                o = args.offset
                for key, pred in (("p", base), ("b", box), ("s", scale)):
                    r[key].append(l2_rel(pred, fut))
                    r[key + "o"].append(l2_rel(pred + o, fut + o))
    print(f"scenarios: {args.which} ({len(data.cases)}), change inside the history")
    print("strict error / paper metric: persistence | perfect box (uniform shift) | perfect scaling of the old map")
    print(f"{'t0':>3} {'persist':>9} {'box':>9} {'scale':>9}   {'paper pers':>10} {'paper box':>10} {'paper scale':>11} {'n':>4}")
    for t0 in sorted(rows):
        r = rows[t0]
        m = lambda key: 100 * np.mean(r[key])
        print(f"{t0:3d} {m('p'):8.1f}% {m('b'):8.1f}% {m('s'):8.1f}%   {m('po'):9.2f}% {m('bo'):9.2f}% {m('so'):10.2f}% {len(r['p']):4d}")
    allm = lambda key: 100 * np.mean([v for t0 in rows for v in rows[t0][key]])
    print(f"mean t0 0..10: persistence {allm('p'):.1f}%  perfect box {allm('b'):.1f}%  perfect scaling {allm('s'):.1f}%"
          f"   | paper: {allm('po'):.2f}% {allm('bo'):.2f}% {allm('so'):.2f}%")
    print("For comparison (test, 60 % tau model): model 28.96 %, persistence 40.11 % in the same samples.")


if __name__ == "__main__":
    main()
