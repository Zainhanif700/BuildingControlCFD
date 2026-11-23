"""
Overview figure, version 2026-11-23: the whole forecast of the SENSOR model for ONE test scenario in ONE image.
The sensor model only sees the CO2 at the sensor positions (stars) and the window speeds and doors; it forecasts the
CO2 of the whole room at 1.6 m height. Same figure as demo_overview.py (maps, four points over the whole run).
Nothing is trained here. Run it in experiments/sensors:
    python3 sensor_overview.py --tag fin_8real --file <data_transitions_v4>/N14.npz --out sensor_overview_N14.png
Models: fin_8real (the 8 real stations), fin_v1 (+ 1 virtual sensor in the middle), fin_v16 (+ 16 virtual sensors).
"""
import argparse
import os

import numpy as np
import torch

import stations as st
from sensor_model import SensorPlaneData
from diagnose_sensor import load_members        # also puts the folder with common.py and forecast.py on the path
import common  # noqa: F401
import demo_overview as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="folder name in checkpoints/, e.g. fin_8real")
    ap.add_argument("--file", default=None, help="transition file of one scenario, e.g. .../data_transitions_v4/N14.npz")
    ap.add_argument("--closed", default=None, help="instead of --file: the windows that are closed, e.g. 4,5 (needs --data-dir)")
    ap.add_argument("--data-dir", default=None, help="folder with the transition files, e.g. data_transitions_v4")
    ap.add_argument("--xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="show the forecast at this point of the room [m] instead of --point")
    ap.add_argument("--people", type=float, nargs=2, default=(40.0, 40.0), metavar=("NOW", "BEFORE"))
    ap.add_argument("--map-t0-min", type=float, default=6.0, help="the maps show the forecast for t0 + 3 min")
    ap.add_argument("--point", type=int, default=2, choices=(0, 1, 2, 3, 4),
                    help="which of the four points to show in the lower panel (default 2, the middle of the seating area); 0 = all four")
    ap.add_argument("--out", default="sensor_overview.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    args.file = D.pick_file(args.file, args.closed, args.data_dir)
    models, ck = load_members(args.tag, dev)
    used = st.by_name(ck["stations"])                      # the stations the members were trained with
    data = SensorPlaneData([args.file], dev, z=1.6, transitions=True, stations=used,
                           hist_len=ck.get("hist", 12), t0_min=ck.get("t0_min", 0))
    n_now, n_before = args.people
    name = os.path.splitext(os.path.basename(args.file))[0]
    print(f"{len(models)} members, model {args.tag}, stations: {ck['stations']}", flush=True)
    points = [tuple(args.xy)] if args.xy else D.POINTS
    show = (0,) if args.xy else (None if args.point == 0 else (args.point - 1,))
    r = D.collect(models, data, n_now, n_before, args.map_t0_min, points=points)
    print("point shown (nearest grid point): " + ", ".join(f"x = {x:.2f} m, y = {y:.2f} m" for x, y in
                                                           [r["points_used"][i] for i in (show or range(len(points)))]))
    marks = data.xyz[data.hist_idx, :2].cpu().numpy()      # the plane points where the sensor values come from
    D.make_figure(r["xy"], r["V"], name, n_now, n_before, r["t_map"], r["truth_map"], r["fc_map"], r["tfull"],
                  r["full_pts"], r["tt"], r["pp"], r["err"], args.out,
                  title_prefix=f"Sensor model ({args.tag}, sees only the {len(marks)} stars)", marks=marks, seat=r["seat"],
                  show_points=show, points=points)
    D.print_errors(name, r)
    print(f"figure saved: {args.out}")


if __name__ == "__main__":
    main()
