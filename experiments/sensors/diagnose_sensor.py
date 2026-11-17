"""
Shows where the error of the trained sensor model comes from. Only reads the trained members, nothing is trained.
Usage: python diagnose_sensor.py --data-dir ../gnot_openfoam_rans/data_transitions_v4 --transitions \
           --scenarios ../gnot_openfoam_rans/scenarios_v4.txt --tag sensors_v4
For the training and for the test scenarios it prints the relative l2 error (same measure as in the training log) of
  model        the ensemble of the trained members on the whole plane
  persistence  "the CO2 map stays as it is now" (uses the true map, which the sensors do not have)
  uniform      the whole room at the mean of the newest station values (a guess that needs no model)
  zero         "no CO2 above outside air" (always 100 %)
  near / far   the model error at plane points closer than --near metres to a station, and in the rest of the room
"""
import argparse
import glob
import os

import numpy as np
import torch

import stations as st
from sensor_model import SensorForecastGNOT, SensorPlaneData
import common
from forecast import predict_full
from train_forecast import split

HERE = os.path.dirname(os.path.abspath(__file__))


def l2(pred, target):
    return float(torch.linalg.norm(pred - target) / torch.linalg.norm(target).clamp_min(1e-30))


def diagnose(models, data, every=4, near=2.0, seed=1234):
    """One row of errors per scenario (mean over the samples of that scenario)."""
    xy = data.xyz[:, :2]
    near_mask = torch.cdist(xy, xy[data.hist_idx]).min(1).values < near
    far_mask = ~near_mask
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(len(data.cases)):
        acc = {key: [] for key in ("model", "persistence", "uniform", "zero", "near", "far", "paper", "pers_paper",
                                   "rmse_ppm", "pers_rmse_ppm", "excess_ppm")}
        for t0 in data.t0s[k][::every]:
            if data.H is not None:                       # transition data: people now and before
                nb, na = rng.uniform(10.0, 80.0, 2)
                n, na = float(nb), float(na)
            else:
                n, na = data.N_ref[k], 0.0
            preds = [predict_full(m, data, k, t0, n, na) for m in models]
            mean = torch.stack([p[0] for p in preds]).mean(0)
            fut, hist = preds[0][1], preds[0][2]
            if torch.linalg.norm(fut) == 0:
                continue
            acc["model"].append(l2(mean, fut))
            acc["persistence"].append(l2(hist[:, -1:].expand_as(fut), fut))
            acc["uniform"].append(l2(hist[data.hist_idx, -1].mean().expand_as(fut), fut))
            acc["zero"].append(1.0)
            # the paper's measure: the outside air (400 ppm) is added to both fields
            acc["paper"].append(l2(mean + st.CO2_OUTDOOR, fut + st.CO2_OUTDOOR))
            acc["pers_paper"].append(l2(hist[:, -1:].expand_as(fut) + st.CO2_OUTDOOR, fut + st.CO2_OUTDOOR))
            # the same in ppm (the data are CO2 above outside air in ppm): root mean square error and size of the signal
            acc["rmse_ppm"].append(float(torch.sqrt(((mean - fut) ** 2).mean())))
            acc["pers_rmse_ppm"].append(float(torch.sqrt(((hist[:, -1:].expand_as(fut) - fut) ** 2).mean())))
            acc["excess_ppm"].append(float(fut.mean()))
            if near_mask.any():
                acc["near"].append(l2(mean[near_mask], fut[near_mask]))
            if far_mask.any():
                acc["far"].append(l2(mean[far_mask], fut[far_mask]))
        rows.append({key: (float(np.mean(v)) if v else float("nan")) for key, v in acc.items()})
    return rows


def decisions(models, data, every=4, thresholds=(700.0, 800.0, 1000.0), seed=1234):
    """The question behind the forecast: will the CO2 in the seating area pass a limit within the next 3 minutes?
    Also the error of the room average. All CO2 values here are absolute (outside air 400 ppm added)."""
    (x0, x1), (y0, y1), _ = common.SEAT_BOX
    xy = data.xyz[:, :2]
    seat = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    out = st.CO2_OUTDOOR
    rng = np.random.default_rng(seed)
    err = {"seat_model": [], "seat_pers": [], "plane_model": [], "plane_pers": [], "seat_true": []}
    counts = {t: {"model": np.zeros(4, int), "persistence": np.zeros(4, int)} for t in thresholds}   # tp fp fn tn
    for k in range(len(data.cases)):
        for t0 in data.t0s[k][::every]:
            if data.H is not None:
                nb, na = rng.uniform(10.0, 80.0, 2)
                n, na = float(nb), float(na)
            else:
                n, na = data.N_ref[k], 0.0
            preds = [predict_full(m, data, k, t0, n, na) for m in models]
            mean = torch.stack([p[0] for p in preds]).mean(0)
            fut, hist = preds[0][1], preds[0][2]
            if torch.linalg.norm(fut) == 0:
                continue
            true_seat = fut[seat].mean(0) + out                    # seating-area mean for each of the next 6 frames
            model_seat = mean[seat].mean(0) + out
            pers_seat = hist[seat, -1].mean() + out                # persistence: the mean stays as it is now
            err["seat_model"].append(float((model_seat - true_seat).abs().mean()))
            err["seat_pers"].append(float((pers_seat - true_seat).abs().mean()))
            err["plane_model"].append(float((mean.mean(0) - fut.mean(0)).abs().mean()))
            err["plane_pers"].append(float((hist[:, -1].mean() - fut.mean(0)).abs().mean()))
            err["seat_true"].append(float(true_seat.mean()))
            for t in thresholds:
                real = bool(true_seat.max() >= t)
                for name, pred in (("model", bool(model_seat.max() >= t)), ("persistence", bool(pers_seat >= t))):
                    counts[t][name][(0 if real else 3) if pred == real else (1 if pred else 2)] += 1
    return err, counts


def print_decisions(err, counts):
    print("\nTEST scenarios, the room average (mean absolute error in ppm, mean over all samples)")
    print(f"  seating-area mean:  model {np.mean(err['seat_model']):6.1f}   persistence {np.mean(err['seat_pers']):6.1f}"
          f"   (mean CO2 there: {np.mean(err['seat_true']):.0f} ppm)")
    print(f"  whole-plane mean:   model {np.mean(err['plane_model']):6.1f}   persistence {np.mean(err['plane_pers']):6.1f}")
    print("\nTEST scenarios, warning: the seating-area mean passes the limit within 3 min "
          f"({len(err['seat_true'])} samples)")
    print("  limit   real     model: found missed false alarm      persistence: found missed false alarm")
    for t, c in counts.items():
        m, p = c["model"], c["persistence"]
        print(f"  {t:5.0f}  {m[0] + m[2]:5d}            {m[0]:5d} {m[2]:6d} {m[1]:11d}                    "
              f"{p[0]:5d} {p[2]:6d} {p[1]:11d}")
    print("  (the number of real events depends on the people numbers drawn for the samples, 10 to 80)")


def print_table(title, names, rows):
    keys = ("model", "persistence", "uniform", "zero", "near", "far", "paper", "pers_paper")
    print(f"\n{title}  (relative l2 error in %, mean over the samples of each scenario; paper = with 400 ppm added)")
    print("  scenario   " + "".join(f"{k:>13s}" for k in keys))
    for name, r in zip(names, rows):
        print(f"  {name:10s} " + "".join(f"{100 * r[k]:13.1f}" for k in keys))
    print(f"  {'mean':10s} " + "".join(f"{100 * np.nanmean([r[k] for r in rows]):13.1f}" for k in keys))
    ppm = ("rmse_ppm", "pers_rmse_ppm", "excess_ppm")
    print(f"\n{title} in ppm: error of the model, error of persistence, mean CO2 above outside air in the forecast")
    print("  scenario   " + "".join(f"{k:>14s}" for k in ppm))
    for name, r in zip(names, rows):
        print(f"  {name:10s} " + "".join(f"{r[k]:14.1f}" for k in ppm))
    print(f"  {'mean':10s} " + "".join(f"{np.nanmean([r[k] for r in rows]):14.1f}" for k in ppm))


def load_members(tag, dev):
    paths = sorted(glob.glob(os.path.join(HERE, "checkpoints", tag, "member*.pth")))
    assert paths, f"no members found in checkpoints/{tag}"
    models, ck = [], None
    for p in paths:
        ck = torch.load(p, map_location=dev)
        m = SensorForecastGNOT(d=ck["d"], layers=ck["layers"], c_scale=ck["c_scale"],
                               anchor=ck.get("anchor", False), blind=ck.get("blind", False)).to(dev)
        m.load_state_dict(ck["state"])
        models.append(m.eval())
    return models, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--transitions", action="store_true")
    ap.add_argument("--z", type=float, default=1.6)
    ap.add_argument("--every", type=int, default=4, help="use every n-th time of each scenario")
    ap.add_argument("--near", type=float, default=2.0, help="distance to a station [m] for the near / far split")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)

    models, ck = load_members(a.tag, dev)
    used = st.by_name(ck["stations"])                                  # the stations the members were trained with
    print(f"{len(models)} members, stations of the training: {ck['stations']}")

    train_files, test_files = split(a.data_dir, a.scenarios, with_s00=False)
    for title, files in (("TRAIN scenarios", train_files), ("TEST scenarios", test_files)):
        data = SensorPlaneData(files, dev, z=a.z, transitions=a.transitions, stations=used)
        rows = diagnose(models, data, every=a.every, near=a.near)
        print_table(title, [os.path.basename(f).replace(".npz", "") for f in files], rows)
        if title.startswith("TEST"):
            print_decisions(*decisions(models, data, every=a.every))
    print("\nuniform and persistence are only references; the sensors do not have the true map.")


if __name__ == "__main__":
    main()
