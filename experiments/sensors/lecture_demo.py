"""
A synthetic lecture through the whole chain, as suggested by Prof. Wagner:
the room starts at about 420 ppm, the students come (the CO2 rises towards 700 and more), some people leave (it falls).
How it is made: CO2 is a passive tracer, so for a flow that does not change the CO2 for a changing number of people is
the response of the simulation (people switched on at time 0) shifted in time and added up for every change of the number
of people. The sensor data of the stations are cut out of this field (server format, noise), go through reader and
mapper, and the trained model forecasts the next 3 minutes every 30 s. The graph shows the average CO2 in the seating
area, the forecast, the simple "stays the same" forecast, the 700 and 800 ppm lines and the warnings.
Use a model that does not need the direction of the air and the number of people:
    train_sensor.py --direction-blind --people-blind (on direct data)
    python lecture_demo.py --data-dir ../gnot_openfoam_rans/data_v4 --scenarios ../gnot_openfoam_rans/scenarios_v4.txt \
        --tag st_lecture --out lecture.png
"""
import argparse
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stations as st
import synthetic
import readers
import mapper
from diagnose_sensor import load_members        # also puts the folder with common.py and forecast.py on the path
import common
from forecast import load_plane, T_OUT
from demo_chain import forecast
from train_forecast import split

START = datetime(2026, 11, 16, 8, 0, tzinfo=timezone.utc)
STEP = 30.0
# (time in s, number of people from then on): a few people early, students come, some leave
PEOPLE = [(0, 5), (300, 25), (600, 45), (1500, 25)]
BASE_EXCESS = 20.0            # ppm above outside air at the beginning (420 ppm)
LIMITS = (700.0, 800.0)


def people_at(t, timeline):
    n = 0.0
    for start, value in sorted(timeline):
        if t >= start:
            n = float(value)
    return n


def smooth_response(frames, width):
    """Spreads the increments of the step response over `width` time steps (a moving average of the increments), so
    that the jump when my CO2 calculation switches from the real flow to the mean flow (150 s) is spread out.
    The response stays 0 at the start. Only for the graph, it changes the data, say so in the caption."""
    if width <= 1:
        return frames
    inc = np.diff(frames, axis=0, prepend=0.0)                       # increments, the first one is the first value
    pad = width // 2
    padded = np.concatenate([np.zeros((pad, inc.shape[1]), inc.dtype), inc, np.repeat(inc[-1:], pad, axis=0)])
    kernel = np.ones(width, np.float32) / width
    smooth = np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(inc.shape[1])], axis=1)
    return np.cumsum(smooth[:len(inc)], axis=0).astype(frames.dtype)


def lecture_excess(frames, n_ref, timeline, base=BASE_EXCESS):
    """CO2 above outside air (time steps, points) for a changing number of people.
    frames: the simulation with n_ref people switched on at time 0."""
    out = np.full(frames.shape, base, np.float32)
    prev = 0.0
    for start, n in sorted(timeline):
        j0 = int(round(start / STEP))
        change = (n - prev) / n_ref
        prev = float(n)
        for k in range(j0, frames.shape[0]):
            out[k] += change * frames[k - j0]
    return out


def run_lecture(models, path, timeline=PEOPLE, hist_len=mapper.N_HIST, margin=50.0, noise=st.CO2_NOISE_PPM, smooth=0):
    """Forecast every 30 s through the whole chain. Returns the times, the seating-area averages and the warnings.
    smooth: width in time steps for smoothing the step response (0 = off, see smooth_response)."""
    d = load_plane(path, z=st.Z_STATION)
    frames = smooth_response(d["frames"], smooth)
    n_ref = d["N_ref"]
    excess = lecture_excess(frames, n_ref, timeline)
    xyz = torch.tensor(np.c_[d["xy"], np.full(len(d["xy"]), d["z"], np.float32)])
    (x0, x1), (y0, y1), _ = common.SEAT_BOX
    seat = (d["xy"][:, 0] >= x0) & (d["xy"][:, 0] <= x1) & (d["xy"][:, 1] >= y0) & (d["xy"][:, 1] <= y1)
    truth = excess[:, seat].mean(1) + st.CO2_OUTDOOR                       # seating-area average in ppm

    rows = synthetic.station_rows(path, start=START, noise=noise, excess=excess)
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "lecture.csv")
        synthetic.write_csv(rows, csv_path)
        records = readers.read_server_csv(csv_path)

    ks, fc_seat = [], []
    for k in range(hist_len - 1, frames.shape[0] - T_OUT):
        n = people_at(STEP * k, timeline)
        m = mapper.model_inputs(records, START.timestamp() + STEP * k, n)
        fc = forecast(models, xyz, torch.tensor(m["xyz"]).unsqueeze(0), torch.tensor(m["hist"]).unsqueeze(0),
                      np.concatenate([m["V"], m["doors"]]), n, hist_len)
        ks.append(k)
        fc_seat.append(fc[torch.tensor(seat)].mean(0).numpy() + st.CO2_OUTDOOR)   # forecast for steps 1..6 ahead
    ks, fc_seat = np.array(ks), np.array(fc_seat)
    future = np.array([truth[k + 1:k + 1 + T_OUT] for k in ks])                # what really happens in the next 3 min
    out = {"t": STEP * np.arange(frames.shape[0]), "truth": truth, "ks": ks, "fc": fc_seat, "future": future,
           "now": truth[ks], "people": np.array([people_at(STEP * k, timeline) for k in range(frames.shape[0])]),
           "margin": margin, "name": os.path.basename(path).replace(".npz", ""), "smooth": smooth}
    for lim in LIMITS:
        out[f"warn_{lim:g}"] = fc_seat.max(1) >= lim - margin                  # the model warns
        out[f"real_{lim:g}"] = future.max(1) >= lim                            # it really passes the limit in 3 min
    return out


def summary(r):
    """Lead time of the first warning before the CO2 first passes each limit, and the false alarms."""
    lines = []
    for lim in LIMITS:
        warn, real = r[f"warn_{lim:g}"], r[f"real_{lim:g}"]
        cross = np.where(r["truth"] >= lim)[0]
        if len(cross) == 0:
            lines.append(f"{lim:g} ppm: the CO2 never reaches it in this lecture (peak {r['truth'].max():.0f} ppm); "
                         f"{int(warn.sum())} warning steps (false alarms)")
            continue
        t_cross = STEP * cross[0]
        w = np.where(warn)[0]
        t_warn = STEP * r["ks"][w[0]] if len(w) else None
        lead = "no warning before" if t_warn is None else f"first warning at {t_warn / 60:.1f} min, {(t_cross - t_warn) / 60:+.1f} min before"
        lines.append(f"{lim:g} ppm: reached at {t_cross / 60:.1f} min; {lead}; found {int((warn & real).sum())} of "
                     f"{int(real.sum())} steps with a real event in the next 3 min, {int((warn & ~real).sum())} false alarm steps")
    return lines


def plot(r, path):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    t_min = r["t"] / 60.0
    ax.plot(t_min, r["truth"], color="black", lw=2, label="real average CO$_2$ in the seating area")
    tt = (STEP * (r["ks"] + T_OUT)) / 60.0                                       # time that the forecasts are for (3 min ahead)
    ax.plot(tt, r["now"], color="gray", lw=1.2, ls="--", label="simple forecast: stays as it is now")
    ax.plot(tt, r["fc"][:, -1], color="#d62728", lw=1.6, label="model forecast, made 3 minutes earlier")
    for lim, col in zip(LIMITS, ("#e6a817", "#d62728")):
        ax.axhline(lim, color=col, ls=":", lw=1)
        ax.annotate(f"{lim:g} ppm", xy=(1.0, lim), xycoords=("axes fraction", "data"), xytext=(5, 0),
                    textcoords="offset points", va="center", ha="left", color=col, fontsize=9, annotation_clip=False)
        warn = r[f"warn_{lim:g}"]
        for k in r["ks"][warn]:
            ax.axvspan(STEP * k / 60.0, STEP * (k + 1) / 60.0, ymin=0.0, ymax=0.04 if lim == LIMITS[0] else 0.08,
                       color=col, alpha=0.8)
    ax.set_ylabel("CO$_2$ [ppm]")
    extra = f", step response smoothed over {r['smooth']} steps" if r.get("smooth", 0) > 1 else ""
    ax.set_title(f"Synthetic lecture (scenario {r['name']}): forecast and warning, margin {r['margin']:g} ppm{extra}")
    ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.95, edgecolor="none", facecolor="white")
    ax2.step(t_min, r["people"], where="post", color="#1f77b4")
    ax2.set_ylabel("people")
    ax2.set_xlabel("time [min]\ncoloured bars at the bottom of the upper graph: the model warns (yellow: 700 ppm, red: 800 ppm)")
    fig.tight_layout(rect=(0, 0, 0.94, 1))                 # room on the right for the labels of the limits
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scenario", default=None, help="name of the scenario, e.g. N24 (default: the first test scenario "
                    "that fits the plan and whose seating-area average reaches 700 to 900 ppm)")
    ap.add_argument("--margin", type=float, default=50.0)
    ap.add_argument("--smooth", type=int, default=0, help="smooth the step response over this many steps of 30 s "
                    "(the data have a jump at 150 s, where my CO2 calculation switches to the mean flow); 0 = off")
    ap.add_argument("--out", default="lecture.png")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    models, ck = load_members(a.tag, torch.device(a.device))
    assert ck["stations"] == [s.name for s in st.STATIONS], "the model was trained with other stations"
    assert ck.get("direction_blind", False) and ck.get("people_blind", False), \
        "train the model with --direction-blind --people-blind"
    train_files, test_files = split(a.data_dir, a.scenarios, with_s00=False)
    files = [f for f in train_files + test_files if os.path.basename(f).replace(".npz", "") == a.scenario] if a.scenario \
        else [f for f in test_files if all(np.load(f)["V"][w - 1] == 0 for w in st.CLOSED_WINDOWS)]
    chosen = None
    (x0, x1), (y0, y1), _ = common.SEAT_BOX
    for f in files:
        d = load_plane(f, z=st.Z_STATION)
        seat = (d["xy"][:, 0] >= x0) & (d["xy"][:, 0] <= x1) & (d["xy"][:, 1] >= y0) & (d["xy"][:, 1] <= y1)
        peak = lecture_excess(d["frames"], d["N_ref"], PEOPLE)[:, seat].mean(1).max() + st.CO2_OUTDOOR
        print(f"  {os.path.basename(f)}: peak of the lecture about {peak:.0f} ppm (average in the seating area)")
        if chosen is None and (a.scenario or 700.0 <= peak <= 900.0):
            chosen = f
    chosen = chosen or files[0]
    r = run_lecture(models, chosen, margin=a.margin, hist_len=ck.get("hist", 12), smooth=a.smooth)
    print(f"\nlecture with the scenario {r['name']}; people: {PEOPLE}")
    for line in summary(r):
        print("  " + line)
    plot(r, a.out)
    print(f"graph saved: {a.out}")


if __name__ == "__main__":
    main()
