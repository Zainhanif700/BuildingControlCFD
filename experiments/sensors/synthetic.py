"""
Makes sensor data from one simulation file, in the same columns as the sensor server (table sensor_data).
The server stores every value as an integer: value * 1,000,000.
Usage: python synthetic.py --data ../gnot_openfoam_rans/data_v3/N10.npz --people 45 --out N10_sensors.csv
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone

import numpy as np

import stations as st

COLUMNS = ["time", "station_id", "scd41_co2", "scd41_temperature", "scd41_humidity",
           "pav1015v_velocity", "vl53l1x_distance", "battery_voltage"]
FIX = 1_000_000


def people_over_time(t, people):
    """people is one number, or a list of (start time in s, number), e.g. [(0, 25), (1200, 45)]."""
    if np.isscalar(people):
        return np.full(len(t), float(people))
    n = np.zeros(len(t))
    for start, value in sorted(people):
        n[t >= start] = value
    return n


def station_rows(data_file, people=20, z=st.Z_STATION, noise=st.CO2_NOISE_PPM, seed=0,
                 start=datetime(2026, 11, 16, 8, 0, tzinfo=timezone.utc), excess=None):
    """excess: optional (time steps, plane points) array with the CO2 above outside air in ppm, already for the wanted
    number of people (e.g. a lecture made with lecture_demo.py). Then `people` is not used."""
    d = np.load(data_file)
    P, C, t = d["P"], d["C"], d["t_c"].astype(float)
    V = d["V"].astype(float)
    doors = d["doors"].astype(float) if "doors" in d.files else np.ones(2)
    layers = np.unique(P[:, 2])
    on_plane = np.abs(P[:, 2] - layers[np.argmin(np.abs(layers - z))]) < 1e-5
    xy, frames = P[on_plane, :2], C[:, on_plane]
    n_people = people_over_time(t, people)
    if excess is not None:
        assert excess.shape == frames.shape, "excess must have the shape (time steps, plane points) of the file"
        frames, n_people = np.asarray(excess, float), np.full(len(t), float(d["N_ref"]))
    rng = np.random.default_rng(seed)

    rows = []
    for s in st.STATIONS:
        cell = int(np.argmin((xy[:, 0] - s.x) ** 2 + (xy[:, 1] - s.y) ** 2))
        # the CO2 of people who arrive later is not exact here: it scales the whole run with the current number
        co2 = st.CO2_OUTDOOR + frames[:, cell] * n_people / float(d["N_ref"]) + rng.normal(0.0, noise, len(t))
        if s.kind == "window":
            is_open = V[s.number - 1] != 0
            # the air speed sensor gives the size of the speed, not the direction. An outlet window of the simulation
            # (value -1) gets the size 1, like the model uses it. A closed window gives no speed.
            speed = abs(V[s.number - 1]) if is_open else None
        else:
            is_open = doors[s.number - 1] > 0
            speed = None
        dist = st.DIST_OPEN_MM if is_open else st.DIST_CLOSED_MM
        for k in range(len(t)):
            rows.append({
                "time": (start + timedelta(seconds=float(t[k]))).isoformat(),
                "station_id": s.id,
                "scd41_co2": int(round(co2[k] * FIX)),
                "scd41_temperature": "",
                "scd41_humidity": "",
                "pav1015v_velocity": "" if speed is None else int(round(speed * FIX)),
                "vl53l1x_distance": int(round(dist * FIX)),
                "battery_voltage": 3700 * FIX,
            })
    rows.sort(key=lambda r: (r["time"], r["station_id"]))
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--people", type=float, default=20)
    ap.add_argument("--noise", type=float, default=st.CO2_NOISE_PPM)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = station_rows(a.data, a.people, noise=a.noise)
    write_csv(rows, a.out)
    print(f"{len(rows)} rows from {len(st.STATIONS)} stations -> {a.out}")
