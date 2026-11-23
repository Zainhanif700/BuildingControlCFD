"""
Forecast at ANY point of the room from SENSOR DATA ONLY, version 2026-11-23.
The chain is the one real sensors will use:   sensor file (server format)  ->  reader  ->  mapper  ->  sensor model  ->
forecast of the CO2 in the whole room  ->  the point you choose.
The model sees only the sensor file: the CO2 of the 8 stations of the last 6 minutes, the window speeds and the door states.
For now the sensor file is made from a simulation (synthetic.py, with noise). Later the same step reads the real export.
The simulation is only used (1) to make the sensor file and (2) to draw the true CO2 at the point for comparison.

Make the sensor file and the forecast in one go (the file is made only if it does not exist yet):
    python3 sensor_forecast.py --tag st_chain --sim ../gnot_openfoam_rans/data_v4/A03.npz --xy 7.8 4.6 --out point_A03.png
or find the simulation by its closed windows (exactly these windows closed, all others open):
    python3 sensor_forecast.py --tag st_chain --closed 4,5 --data-dir ../gnot_openfoam_rans/data_v4 --xy 7.8 4.6
The model has to be a direction-blind one trained on direct data (a speed sensor gives no direction), e.g. st_chain.
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stations as st
import synthetic
import readers
import mapper
from diagnose_sensor import load_members, l2        # also puts the folder with common.py and forecast.py on the path
import common  # noqa: F401
from forecast import load_plane, T_OUT, DT
from demo_chain import forecast as chain_forecast, START
import demo_overview as D

OFF = st.CO2_OUTDOOR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="st_chain", help="sensor model in checkpoints/ (direction-blind, trained on direct data)")
    ap.add_argument("--sim", default=None, help="direct data file of one simulation, e.g. .../data_v4/A03.npz")
    ap.add_argument("--closed", default=None, help="instead of --sim: the windows that are closed, e.g. 4,5 (needs --data-dir)")
    ap.add_argument("--data-dir", default=None, help="folder with the direct data files, e.g. .../data_v4")
    ap.add_argument("--csv", default=None, help="sensor file (server format); made from the simulation if it does not exist")
    ap.add_argument("--people", type=float, default=40.0)
    ap.add_argument("--noise", type=float, default=st.CO2_NOISE_PPM, help="noise of the synthetic CO2 values [ppm]")
    ap.add_argument("--xy", type=float, nargs=2, default=(7.8, 4.6), metavar=("X", "Y"),
                    help="the point of the room [m] (x 0 to 15.5, y 0 to 9.2); default: the middle of the seating area")
    ap.add_argument("--map-min", type=float, default=9.0, help="time of the room maps [min]")
    ap.add_argument("--out", default="sensor_forecast.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)

    sim = D.pick_file(a.sim, a.closed, a.data_dir)
    name = os.path.splitext(os.path.basename(sim))[0]
    models, ck = load_members(a.tag, dev)
    assert ck["stations"] == [s.name for s in st.STATIONS], "the model was trained with other stations"
    assert ck.get("direction_blind", False), "use a model trained with --direction-blind (a speed sensor has no direction)"
    hist_len = ck.get("hist", 12)

    csv_path = a.csv or f"sensors_{name}.csv"
    if not os.path.exists(csv_path):
        rows = synthetic.station_rows(sim, people=a.people, start=START, noise=a.noise)
        synthetic.write_csv(rows, csv_path)
        print(f"sensor file made from the simulation {name}: {len(rows)} rows -> {csv_path}", flush=True)
    else:
        print(f"using the existing sensor file {csv_path}", flush=True)
    records = readers.read_server_csv(csv_path)             # from here on the model only sees this file

    d = load_plane(sim, z=st.Z_STATION)                     # grid of the room, and the truth for the comparison
    xy = d["xy"]
    xyz = torch.tensor(np.c_[xy, np.full(len(xy), d["z"], np.float32)])
    truth = torch.tensor(d["frames"]) * (a.people / d["N_ref"])
    V_sim = np.asarray(d["V"], float)
    if any(V_sim[w - 1] != 0 for w in st.CLOSED_WINDOWS):
        print(f"NOTE: the windows {st.CLOSED_WINDOWS} have no station; in this simulation one of them is open, "
              f"so the model cannot see it", flush=True)
    ip = int(np.argmin((xy[:, 0] - a.xy[0]) ** 2 + (xy[:, 1] - a.xy[1]) ** 2))
    px, py = float(xy[ip, 0]), float(xy[ip, 1])
    print(f"point shown (nearest grid point): x = {px:.2f} m, y = {py:.2f} m", flush=True)

    ks = list(range(hist_len - 1, truth.shape[0] - T_OUT))
    k_map = min(max(int(round(a.map_min * 60 / DT)) - T_OUT, ks[0]), ks[-1])
    t_fc, fc_pt, errs, maps = [], [], [], {}
    for k in ks:
        m = mapper.model_inputs(records, START.timestamp() + DT * k, a.people)
        V_map = np.concatenate([m["V"], m["doors"]])
        chain = chain_forecast(models, xyz, torch.tensor(m["xyz"]).unsqueeze(0), torch.tensor(m["hist"]).unsqueeze(0),
                               V_map, a.people, hist_len)
        fut = truth[k + 1:k + 1 + T_OUT].T
        t_fc.append((k + T_OUT) * DT / 60.0)
        fc_pt.append(float(chain[ip, -1]) + OFF)
        errs.append(l2(chain, fut))
        if k == k_map:
            maps = {"fc": chain[:, -1].numpy() + OFF, "true": fut[:, -1].numpy() + OFF, "t": (k + T_OUT) * DT / 60.0}
    t_all = np.arange(truth.shape[0]) * DT / 60.0
    true_pt = truth[:, ip].numpy() + OFF
    err_pt = np.abs(np.array(fc_pt) - np.interp(t_fc, t_all, true_pt))
    print(f"scenario {name}, sensor model {a.tag}: mean error of the forecast at this point {err_pt.mean():.1f} ppm "
          f"(largest {err_pt.max():.1f} ppm); mean error of the whole-room map (strict) {100 * np.mean(errs):.1f} %", flush=True)

    fig = plt.figure(figsize=(15, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    Vd = [abs(float(v)) for v in V_sim]                     # an outlet window (-1) is open too
    marks = np.array([[s.x, s.y] for s in st.STATIONS])
    vmin, vmax = float(maps["true"].min()), float(maps["true"].max())
    for j, (val, ttl) in enumerate(((maps["fc"], f"Forecast from the sensors only (the stars), t = {maps['t']:g} min"),
                                    (maps["true"], f"Simulation (reference), t = {maps['t']:g} min"))):
        ax = fig.add_subplot(gs[0, j])
        img, ext = D.to_image(xy, val)
        im = ax.imshow(img, origin="lower", extent=ext, cmap="viridis", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label="CO$_2$ [ppm]", fraction=0.04, pad=0.02)
        D.draw_room(ax, Vd)
        ax.plot(marks[:, 0], marks[:, 1], "k*", ms=9, mfc="yellow", mew=0.8)
        ax.plot(px, py, "x", color="red", ms=12, mew=3)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("x: room length [m]")
        ax.set_ylabel("y: room width [m]")
    ax = fig.add_subplot(gs[1, :])
    ax.axvspan(0, hist_len * DT / 60, color="0.9", label="first 6 min: history only, no forecast yet")
    for s in st.STATIONS:
        rs = sorted((r for r in records if r["station"] == s.id), key=lambda r: r["time"])
        ax.plot([(r["time"] - START.timestamp()) / 60.0 for r in rs], [r["co2"] for r in rs], "-", lw=0.8, alpha=0.6)
    ax.plot([], [], "-", lw=0.8, color="0.4", label="input: the 8 sensors (CO$_2$ with noise)")
    ax.plot(t_all, true_pt, "-", color="black", lw=2.5, label=f"simulation at the red cross (x = {px:.1f} m, y = {py:.1f} m)")
    ax.plot(t_fc, fc_pt, "o", color="#d62728", ms=4, label="forecast at the red cross, made 3 min earlier, from the sensors only")
    ax.set_xlabel("time [min]")
    ax.set_ylabel("CO$_2$ [ppm] (outdoor air: 400 ppm)")
    ax.set_title("Input: the sensor values. Output: the forecast at the chosen point, 3 minutes ahead", fontsize=10)
    ax.legend(fontsize=8.5, loc="best")
    ax.grid(alpha=0.3)
    fig.suptitle(f"Sensor chain, scenario {name}, {a.people:g} people, model {a.tag}: window speeds "
                 f"{[round(float(v), 2) for v in V_sim]} m/s (-1 = outlet window), doors {[int(v) for v in d['doors']]} (1 = open)\n"
                 f"mean error at the chosen point {err_pt.mean():.1f} ppm; the sensor values come from the simulation "
                 f"(no real measurements yet)", fontsize=11)
    fig.savefig(a.out, dpi=130)
    plt.close(fig)
    print(f"figure saved: {a.out}")


if __name__ == "__main__":
    main()
