"""
Test of the whole chain with a trained model, the way real sensor data will go through it:
    simulation file -> sensor file in the server format (synthetic.py) -> reader -> mapper -> trained model -> forecast
The forecast is compared with the simulated truth. For comparison the same model also gets the history straight from the
simulation (no sensor file, no noise): if both errors are about the same, the chain does not lose anything.
Use a model that does not need the direction of the air (train_sensor.py --direction-blind), trained on direct data:
    python demo_chain.py --data-dir ../gnot_openfoam_rans/data_v4 --scenarios ../gnot_openfoam_rans/scenarios_v4.txt \
        --tag st_chain
"""
import argparse
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import torch

import stations as st
import synthetic
import readers
import mapper
from diagnose_sensor import load_members, l2        # also puts the folder with common.py and forecast.py on the path
import common
from forecast import load_plane, T_OUT
from train_forecast import split

START = datetime(2026, 11, 16, 8, 0, tzinfo=timezone.utc)


def forecast(models, xyz, hx, hh, V, people, hist_len):
    """Ensemble mean forecast (points, 6) for the whole plane."""
    dev = next(models[0].parameters()).device                    # the models may be on the GPU
    xyz, hx, hh = xyz.to(dev), hx.to(dev), hh.to(dev)
    Vt = torch.as_tensor(V, dtype=torch.float32).unsqueeze(0).to(dev)
    Nt = torch.tensor([[float(people)]], device=dev)
    outs = []
    with torch.no_grad():
        for m in models:
            parts = []
            for i in range(0, xyz.shape[0], 4096):
                q = xyz[i:i + 4096].unsqueeze(0)
                mu, _ = m(q, torch.zeros(1, q.shape[1], hist_len, device=dev), hx, hh, Vt, Nt)
                parts.append(mu[0])
            outs.append(torch.cat(parts))
    return torch.stack(outs).mean(0).cpu()


def run_case(models, path, people, steps, hist_len=mapper.N_HIST, start=START, noise=st.CO2_NOISE_PPM):
    """For some times of one scenario file: error of the chain, error with the history straight from the simulation,
    error of persistence, and the error of the seating-area mean in ppm."""
    d = load_plane(path, z=st.Z_STATION)
    xyz = torch.tensor(np.c_[d["xy"], np.full(len(d["xy"]), d["z"], np.float32)])
    frames = torch.tensor(d["frames"]) * (people / d["N_ref"])               # CO2 above outside air for this many people
    cells = [int(np.argmin((d["xy"][:, 0] - s.x) ** 2 + (d["xy"][:, 1] - s.y) ** 2)) for s in st.STATIONS]
    (x0, x1), (y0, y1), _ = common.SEAT_BOX
    seat = torch.tensor((d["xy"][:, 0] >= x0) & (d["xy"][:, 0] <= x1) & (d["xy"][:, 1] >= y0) & (d["xy"][:, 1] <= y1))
    rows = synthetic.station_rows(path, people=people, start=start, noise=noise)
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "sensors.csv")
        synthetic.write_csv(rows, csv_path)
        records = readers.read_server_csv(csv_path)

    V_true = np.concatenate([np.abs(d["V"]), d["doors"]]).astype(np.float32)    # what a speed sensor can give
    out = []
    for k in steps:
        m = mapper.model_inputs(records, start.timestamp() + 30.0 * k, people)
        V_map = np.concatenate([m["V"], m["doors"]])
        hx = torch.tensor(m["xyz"]).unsqueeze(0)
        fut = frames[k + 1:k + 1 + T_OUT].T
        chain = forecast(models, xyz, hx, torch.tensor(m["hist"]).unsqueeze(0), V_map, people, hist_len)
        hh_true = frames[k - hist_len + 1:k + 1][:, cells].T.unsqueeze(0)
        direct = forecast(models, xyz, hx, hh_true, V_true, people, hist_len)
        pers = frames[k].unsqueeze(1).expand_as(fut)
        out.append({"k": k, "chain": l2(chain, fut), "direct": l2(direct, fut), "persistence": l2(pers, fut),
                    "seat_ppm": float((chain[seat].mean(0) - fut[seat].mean(0)).abs().mean()),
                    "v_ok": bool(np.allclose(V_map, V_true, atol=1e-4))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--people", type=float, default=45.0)
    ap.add_argument("--steps", type=int, nargs="+", default=[20, 30, 40])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    models, ck = load_members(a.tag, torch.device(a.device))
    assert ck["stations"] == [s.name for s in st.STATIONS], "the model was trained with other stations"
    assert ck.get("direction_blind", False), "train the model with --direction-blind (a speed sensor has no direction)"
    _, test_files = split(a.data_dir, a.scenarios, with_s00=False)
    hist_len = ck.get("hist", 12)

    print(f"{len(models)} members, {a.people:g} people, forecast from steps {a.steps} (30 s each)")
    print("  scenario   step    chain  straight  persistence  seating-area mean   mapper speeds right")
    all_rows = []
    for f in test_files:
        V = np.load(f)["V"]
        if any(V[w - 1] != 0 for w in st.CLOSED_WINDOWS):        # a window without a station cannot be seen
            print(f"  {os.path.basename(f).replace('.npz', ''):9s} skipped: a window without a station is open")
            continue
        for r in run_case(models, f, a.people, a.steps, hist_len):
            all_rows.append(r)
            print(f"  {os.path.basename(f).replace('.npz', ''):9s} {r['k']:5d}  {100 * r['chain']:6.1f}   {100 * r['direct']:6.1f}"
                  f"   {100 * r['persistence']:9.1f}   {r['seat_ppm']:11.1f} ppm   {'yes' if r['v_ok'] else 'NO'}")
    mean = lambda key: 100 * np.mean([r[key] for r in all_rows])
    print(f"\nmean (strict error in %): chain {mean('chain'):.1f}, straight from the simulation {mean('direct'):.1f}, "
          f"persistence {mean('persistence'):.1f}; seating-area mean error {np.mean([r['seat_ppm'] for r in all_rows]):.1f} ppm")
    print(f"mapper speeds right in {sum(r['v_ok'] for r in all_rows)} of {len(all_rows)} cases")


if __name__ == "__main__":
    main()
