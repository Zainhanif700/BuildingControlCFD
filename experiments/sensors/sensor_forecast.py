"""
Forecast at ANY point of the room from SENSOR DATA ONLY, version 2026-11-23.
How the model works (as the real system will do it), at every step of 30 s:
    INPUT : the sensor values of the last 6 minutes (CO2 of the 8 stations, window speeds, door states)
    OUTPUT: the CO2 of the whole room 3 minutes ahead, and from it the CO2 at the point you choose
Chain: sensor file (server format) -> reader -> mapper -> sensor model -> forecast of the whole room -> your point.
The model sees only the sensor file. For now the file is made from a simulation (synthetic.py, with noise); later the
same step reads the real export. The simulation is also used to draw the true CO2 for comparison.
The figure has the layout of demo_overview.py: three room maps (simulation, forecast from the sensors, difference),
the average CO2 in the seating area, and the CO2 at your point over the whole run with the sensor inputs.

    python3 sensor_forecast.py --tag st_lecture --sim ../gnot_openfoam_rans/data_v4/N24.npz --xy 7.8 4.6
or find the simulation by its closed windows:
    python3 sensor_forecast.py --tag st_lecture --closed 4,5 --data-dir ../gnot_openfoam_rans/data_v4 --xy 7.8 4.6
The model has to be a direction-blind one trained on direct data (a speed sensor gives no direction), e.g. st_lecture.
"""
import argparse
import os

import numpy as np
import torch

import stations as st
import synthetic
import readers
import mapper
from diagnose_sensor import load_members, l2        # also puts the folder with common.py and forecast.py on the path
import common
from forecast import load_plane, T_OUT, DT
from demo_chain import forecast as chain_forecast, START
import demo_overview as D
from plot_forecast import WINDOWS, DOORS, ROOM_X, ROOM_Y

OFF = st.CO2_OUTDOOR


def label_room(ax, V):
    """Names of the windows (top wall), the doors (bottom wall) and the sensors in a room plan."""
    for k, (x0, x1, _, _) in enumerate(WINDOWS):
        ax.text((x0 + x1) / 2, ROOM_Y[1] + (0.95 if k in (2, 5) else 0.35), f"W{k + 1}", ha="center", va="bottom",
                fontsize=8, color="#0b7fb5" if abs(V[k]) > 0 else "0.45", weight="bold" if abs(V[k]) > 0 else None)
    for k, (x0, x1, _, _) in enumerate(DOORS):
        ax.text((x0 + x1) / 2, ROOM_Y[0] - 0.45, f"D{k + 1}", ha="center", va="top", fontsize=8, color="darkorange",
                weight="bold")
    for s in st.STATIONS:
        ax.annotate(s.name.replace("S-", ""), (s.x, s.y),
                    xytext=(0, (-24 if s.number in (3, 6) else -13) if s.kind == "window" else 13),
                    textcoords="offset points", ha="center", fontsize=7)
    ax.set_ylim(ROOM_Y[0] - 1.6, ROOM_Y[1] + 1.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="st_lecture", help="sensor model in checkpoints/ (direction-blind, trained on direct data)")
    ap.add_argument("--sim", default=None, help="direct data file of one simulation, e.g. .../data_v4/N24.npz")
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
    V_sim = [float(v) for v in d["V"]] + [float(v) for v in d["doors"]]
    if any(V_sim[w - 1] != 0 for w in st.CLOSED_WINDOWS):
        print(f"NOTE: the windows {st.CLOSED_WINDOWS} have no station; in this simulation one of them is open, "
              f"so the model cannot see it", flush=True)
    ip = int(np.argmin((xy[:, 0] - a.xy[0]) ** 2 + (xy[:, 1] - a.xy[1]) ** 2))
    px, py = float(xy[ip, 0]), float(xy[ip, 1])
    print(f"point shown (nearest grid point): x = {px:.2f} m, y = {py:.2f} m", flush=True)
    (x0, x1), (y0, y1), _ = common.SEAT_BOX
    sm = torch.tensor((xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1))

    ks = list(range(hist_len - 1, truth.shape[0] - T_OUT))
    k_map = min(max(int(round(a.map_min * 60 / DT)) - T_OUT, ks[0]), ks[-1])
    tt, fc_pt, em, bm, es, bs, s_fc, s_pers, s_em, s_ep = [], [], [], [], [], [], [], [], [], []
    truth_map = fc_map = None
    for k in ks:
        m = mapper.model_inputs(records, START.timestamp() + DT * k, a.people)          # INPUT: the last 6 min of the sensors
        V_map = np.concatenate([m["V"], m["doors"]])
        chain = chain_forecast(models, xyz, torch.tensor(m["xyz"]).unsqueeze(0), torch.tensor(m["hist"]).unsqueeze(0),
                               V_map, a.people, hist_len)                               # OUTPUT: whole room, next 3 min
        fut = truth[k + 1:k + 1 + T_OUT].T
        base = truth[k].unsqueeze(1).expand_as(fut)                                      # "stays as it is now" (true map)
        tt.append((k + T_OUT) * DT / 60.0)
        fc_pt.append(float(chain[ip, -1]) + OFF)
        em.append(l2(chain + OFF, fut + OFF))
        bm.append(l2(base + OFF, fut + OFF))
        es.append(l2(chain, fut))
        bs.append(l2(base, fut))
        s_fc.append(float(chain[sm, -1].mean()) + OFF)
        s_pers.append(float(base[sm, -1].mean()) + OFF)
        s_em.append(float((chain[sm].mean(0) - fut[sm].mean(0)).abs().mean()))
        s_ep.append(float((base[sm].mean(0) - fut[sm].mean(0)).abs().mean()))
        if k == k_map:
            truth_map, fc_map = fut[:, -1].numpy() + OFF, chain[:, -1].numpy() + OFF
    t_all = np.arange(truth.shape[0]) * DT / 60.0
    true_pt = truth[:, ip].numpy() + OFF
    err_pt = np.abs(np.array(fc_pt) - np.interp(tt, t_all, true_pt))
    print(f"scenario {name}, sensor model {a.tag}: mean error at the point {err_pt.mean():.1f} ppm (largest "
          f"{err_pt.max():.1f} ppm); seating-area average {np.mean(s_em):.1f} ppm; whole-room map (paper definition / strict) "
          f"{100 * np.mean(em):.2f} % / {100 * np.mean(es):.1f} %", flush=True)

    inputs = []
    for s in st.STATIONS:
        rs = sorted((r for r in records if r["station"] == s.id), key=lambda r: r["time"])
        inputs.append(([(r["time"] - START.timestamp()) / 60.0 for r in rs], [r["co2"] for r in rs], s.name))
    err = {"paper_model": np.mean(em), "paper_pers": np.mean(bm), "strict_model": np.mean(es), "strict_pers": np.mean(bs)}
    seat = {"full": truth[:, sm].mean(1).numpy() + OFF, "fc": np.array(s_fc), "pers": np.array(s_pers),
            "box": ((x0, x1), (y0, y1)), "ppm_model": float(np.mean(s_em)), "ppm_pers": float(np.mean(s_ep))}
    marks = np.array([[s.x, s.y] for s in st.STATIONS])
    openw = [f"W{i + 1} {abs(v):.2g} m/s" for i, v in enumerate(V_sim[:8]) if v != 0]
    suptitle = (f"Sensor chain: simulated sensor data of {name} ({a.people:g} people), model {a.tag}\n"
                f"INPUT: the last 6 min of the 8 sensors (stars)  ->  OUTPUT: the CO$_2$ of the room 3 min ahead\n"
                f"open windows: {', '.join(openw) if openw else 'none'}; doors: D1 {'open' if V_sim[8] else 'closed'}, "
                f"D2 {'open' if V_sim[9] else 'closed'}\n"
                f"mean error at the red cross {err_pt.mean():.1f} ppm, of the average in the seating area "
                f"{np.mean(s_em):.1f} ppm (the sensor values come from a simulation, no real measurements yet)")
    D.make_figure(xy, V_sim, name, a.people, a.people, (k_map + T_OUT) * DT / 60.0, truth_map, fc_map, t_all,
                  true_pt.reshape(-1, 1), np.array(tt), np.array(fc_pt).reshape(-1, 1), err, a.out,
                  title_prefix="Sensor chain", marks=marks, seat=seat, show_points=(0,), points=[(px, py)],
                  inputs=inputs, point_name="forecast point", suptitle=suptitle, room_labeler=label_room)
    print(f"figure saved: {a.out}")


if __name__ == "__main__":
    main()
