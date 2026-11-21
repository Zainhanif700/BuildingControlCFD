"""
Can the warning find more real cases without more false alarms? (no training, uses trained sensor models)

Real case: the average CO2 in the seating area really passes the limit within the next 3 minutes.
For every test sample (every 30 s of every test scenario) the ensemble of sensor models gives a forecast of the seating-area
average. Rules that decide "warn or not" are compared:
  margin     : warn if forecast >= limit - m                              (the rule of the thesis, m = 0, 10, ..., 100 ppm)
  spread     : warn if forecast + k * s >= limit                          (s = spread of the 5 networks, k = 0, 0.5, ..., 4)
  near       : warn if forecast >= limit - m AND the forecast for the next 30 s is >= limit - L   (suppresses warnings far below the limit)
  twice      : warn if the margin rule fires at two consecutive 30-s steps of the same scenario
Reported: found (share of real cases that were found) and false alarm rate (share of the other samples with a warning).
For each rule the setting with the fewest false alarms that still finds at least 85 % (and 90 %) of the real cases.
The settings are chosen on the same test samples that are counted, so the numbers are optimistic, as in the thesis margin table.

Usage (like diagnose_sensor.py, one run per random split, results summed):
  python3 warning_rules.py --data-dir <data_transitions_v4> --scenarios <scenarios_v4.txt> --tag st_chk_d1 --split-seed 1 --transitions
  (repeat with st_chk_d2 / 2 and st_chk_d3 / 3, or give several as --runs st_chk_d1:1,st_chk_d2:2,st_chk_d3:3)
"""
import argparse
import os

import numpy as np
import torch

import stations as st
from sensor_model import SensorPlaneData      # also puts the forecast code on the import path
import common
from forecast import predict_full
from train_forecast import split
from diagnose_sensor import load_members

HERE = os.path.dirname(os.path.abspath(__file__))
LIMITS = (700.0, 800.0)


def collect(tag, split_seed, args, dev):
    """Per test sample: scenario id, step, true seating max, forecast of the seating mean for 6 steps, spread."""
    models, ck = load_members(tag, dev)
    used = st.by_name(ck["stations"])
    train_files, test_files = split(args.data_dir, args.scenarios, with_s00=False)
    if split_seed:
        from train_sensor import resplit
        train_files, test_files = resplit(train_files, test_files, split_seed)
    data = SensorPlaneData(test_files, dev, z=args.z, transitions=args.transitions, stations=used,
                           hist_len=ck.get("hist", 12), t0_min=ck.get("t0_min", 0))
    (x0, x1), (y0, y1), _ = common.SEAT_BOX
    xy = data.xyz[:, :2]
    seat = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    out = st.CO2_OUTDOOR
    rng = np.random.default_rng(1234)
    rows = []
    for k in range(len(data.cases)):
        for t0 in data.t0s[k][::args.every]:
            if data.H is not None:
                nb, na = rng.uniform(10.0, 80.0, 2)
                n, na = float(nb), float(na)
            else:
                n, na = data.N_ref[k], 0.0
            preds = [predict_full(m, data, k, t0, n, na) for m in models]
            fut = preds[0][1]
            if torch.linalg.norm(fut) == 0:
                continue
            per_member = torch.stack([p[0][seat].mean(0) + out for p in preds])          # (members, 6)
            mean = per_member.mean(0)
            spread = per_member.std(0).max()
            rows.append((f"{tag}:{k}", t0, float((fut[seat].mean(0) + out).max()), mean.cpu().numpy(),
                         float(spread)))
    return rows


def rule_flags(rows, limit, kind, p, q=None):
    """Boolean array: does the rule warn for each sample (rows are in time order within a scenario)?"""
    F = np.array([r[3].max() for r in rows])
    f1 = np.array([r[3][0] for r in rows])
    S = np.array([r[4] for r in rows])
    if kind == "margin":
        w = F >= limit - p
    elif kind == "spread":
        w = F + p * S >= limit
    elif kind == "near":
        w = (F >= limit - p) & (f1 >= limit - q)
    elif kind == "twice":
        base = F >= limit - p
        w = base.copy()
        for i in range(len(rows)):
            w[i] = base[i] and i > 0 and rows[i - 1][0] == rows[i][0] and rows[i - 1][1] == rows[i][1] - 1 and base[i - 1]
    else:
        raise ValueError(kind)
    return w


def score(rows, limit, w):
    real = np.array([r[2] >= limit for r in rows])
    tp, fn = int((w & real).sum()), int((~w & real).sum())
    fp, tn = int((w & ~real).sum()), int((~w & ~real).sum())
    return tp, fn, fp, tn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--runs", default=None, help="tag:split-seed list, e.g. st_chk_d1:1,st_chk_d2:2,st_chk_d3:3")
    ap.add_argument("--tag")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--transitions", action="store_true")
    ap.add_argument("--z", type=float, default=1.6)
    ap.add_argument("--every", type=int, default=1, help="use every n-th 30-s step (1 needed for the 'twice' rule)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    runs = [tuple(r.split(":")) for r in args.runs.split(",")] if args.runs else [(args.tag, str(args.split_seed))]
    rows = []
    for tag, seed in runs:
        r = collect(tag, int(seed), args, dev)
        print(f"{tag} (split seed {seed}): {len(r)} test samples", flush=True)
        rows += r

    margins = list(range(0, 101, 10))
    ks = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    Ls = [50, 100, 150, 200]
    for limit in LIMITS:
        real = sum(1 for r in rows if r[2] >= limit)
        neg = len(rows) - real
        print(f"\n=== limit {limit:.0f} ppm: {real} real-case samples, {neg} other samples (steps of 30 s)")
        cands = []
        for m in margins:
            cands.append(("margin", f"m = {m}", rule_flags(rows, limit, "margin", m)))
            cands.append(("twice", f"m = {m}", rule_flags(rows, limit, "twice", m)))
        for k in ks:
            cands.append(("spread", f"k = {k}", rule_flags(rows, limit, "spread", k)))
        for m in margins:
            for L in Ls:
                cands.append(("near", f"m = {m}, L = {L}", rule_flags(rows, limit, "near", m, L)))
        print(f"{'rule':<8} {'setting':<18} {'found':>8} {'false alarm rate':>17}")
        for want in (0.85, 0.90):
            print(f"-- best setting of each rule with found >= {100 * want:.0f} %")
            for kind in ("margin", "spread", "near", "twice"):
                best = None
                for kd, label, w in cands:
                    if kd != kind:
                        continue
                    tp, fn, fp, tn = score(rows, limit, w)
                    if real and tp / real >= want and (best is None or fp < best[2][2]):
                        best = (label, w, (tp, fn, fp, tn))
                if best is None:
                    print(f"{kind:<8} {'(not reached)':<18}")
                else:
                    tp, fn, fp, tn = best[2]
                    print(f"{kind:<8} {best[0]:<18} {100 * tp / real:7.1f}% {100 * fp / max(neg, 1):16.1f}%   "
                          f"(found {tp}, missed {fn}, false alarms {fp})")
        tp, fn, fp, tn = score(rows, limit, rule_flags(rows, limit, "margin", 0))
        print(f"reference: margin 0:  found {100 * tp / max(real, 1):.1f}%, false alarm rate {100 * fp / max(neg, 1):.1f}%")
    print("\nSamples of one scenario are not independent (neighbouring 30-s steps), so the counts overstate how many separate "
          "events there are. The settings are chosen on the counted samples.")


if __name__ == "__main__":
    main()
