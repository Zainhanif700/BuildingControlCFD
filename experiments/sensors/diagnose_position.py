"""
Why does a virtual sensor on the blackboard side ("front", x = 4.0 m) give no gain, while the other places do?
No training, uses only the CO2 maps of the 68 scenarios (small transition files, the part "new people, fresh start"
= the direct-run situation) on the plane at 1.6 m, and takes a few minutes at most.

For every virtual place it measures how well the CO2 at that place tells the average CO2 of the seating area:
  r^2   : squared correlation of (CO2 at the place) and (seating-area average), over all scenarios and times 2..30 min
  ratio : seating average / CO2 at the place at 15 min, median over the scenarios and the spread (10th..90th percentile)
          -> a good place has a ratio that is similar in every scenario
and the same r^2 for two groups of scenarios: with an air path from window W1 or W2 straight to door 1 ("jet path": W1 or W2
open as inlet and door 1 open) and all others.

Usage: python3 diagnose_position.py --data-dir ../gnot_openfoam_rans/data_transitions_v4 --scenarios ../gnot_openfoam_rans/scenarios_v4.txt
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gnot_openfoam_rans"))
import common  # noqa: E402
import forecast as F  # noqa: E402
from train_forecast import split  # noqa: E402

import stations as st  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--sets", default="1,front,back,windowside,doorside,2,4,16")
    args = ap.parse_args()
    trp, tep = split(args.data_dir, args.scenarios, False)
    cases = [F.load_transition(p, 1.6) for p in trp + tep]
    xy = cases[0]["xy"]
    (x0, x1), (y0, y1), _ = common.SEAT_BOX
    seat = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    t_lo, t_mid, t_hi = 4, 30, 60            # 2 min, 15 min, 30 min (maps every 30 s)
    jet = np.array([(c["V"][0] > 0 or c["V"][1] > 0) and c["doors"][0] == 1 for c in cases])
    print(f"{len(cases)} scenarios, {int(seat.sum())} plane points in the seating area, "
          f"{int(jet.sum())} with a jet path W1/W2 -> door 1, {int((~jet).sum())} without")
    y_all = np.stack([c["frames"][:, seat].mean(1) for c in cases])         # (scenario, time)

    def idx_of(x, y):
        return int(np.argmin((xy[:, 0] - x) ** 2 + (xy[:, 1] - y) ** 2))

    print(f"\n{'place':<11} {'(x, y) [m]':<22} {'r^2 all':>8} {'r^2 jet path':>13} {'r^2 no jet':>11} "
          f"{'ratio median':>13} {'10th..90th':>14}")
    for name in args.sets.split(","):
        pts = st.VIRTUAL_SETS[name]
        # for several sensors: the mean of the sensors (a simple stand-in)
        s_all = np.stack([np.mean([c["frames"][:, idx_of(px, py)] for px, py in pts], 0) for c in cases])
        sl = slice(t_lo, t_hi + 1)

        def r2(mask):
            a, b = s_all[mask][:, sl].ravel(), y_all[mask][:, sl].ravel()
            return float(np.corrcoef(a, b)[0, 1] ** 2) if mask.sum() > 2 else float("nan")

        ratio = y_all[:, t_mid] / np.maximum(s_all[:, t_mid], 1e-9)
        label = "; ".join(f"({px:.1f}, {py:.1f})" for px, py in pts[:2]) + (" ..." if len(pts) > 2 else "")
        print(f"{name:<11} {label:<22} {r2(np.ones(len(cases), bool)):8.3f} {r2(jet):13.3f} {r2(~jet):11.3f} "
              f"{np.median(ratio):13.2f} {np.percentile(ratio, 10):6.2f}..{np.percentile(ratio, 90):.2f}")
    print("\nr^2 close to 1 and a ratio that is similar in all scenarios (narrow 10th..90th range) mean that the place tells "
          "the seating average well.")


if __name__ == "__main__":
    main()
