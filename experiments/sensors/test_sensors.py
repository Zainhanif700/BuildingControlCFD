"""
Small tests for the sensor code. They run on the CPU in a few seconds and need no real data:
    python test_sensors.py
A small fake simulation file is made first, then every step of the chain is checked.
"""
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import torch

import stations as st
import synthetic
import readers
import mapper
from sensor_model import SensorForecastGNOT, SensorPlaneData
from forecast import nll, evaluate

V_TRUE = np.array([0.5, 0, 1.0, 0, 0, 0.8, 0, 0], np.float32)     # W1, W3, W6 open
DOORS_TRUE = np.array([1, 0], np.float32)


def fake_file(path, seed=0):
    """A coarse plane (0.5 m) with a CO2 field that grows in time and is different at every point."""
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.arange(0.25, 15.5, 0.5), np.arange(0.25, 9.1, 0.5))
    P = np.c_[x.ravel(), y.ravel(), np.full(x.size, 1.58)].astype(np.float32)
    t = np.arange(0.0, 1800.0 + 1, 30.0)
    shape = 50 + 30 * np.sin(P[:, 0] / 3.0 + seed) + 20 * np.cos(P[:, 1] / 2.0) + rng.normal(0, 2, len(P))
    C = (t[:, None] / 1800.0) * shape[None, :]
    np.savez(path, P=P, C=C.astype(np.float32), t_c=t.astype(np.float32), V=V_TRUE, doors=DOORS_TRUE,
             N_ref=np.float32(20.0), plane=np.ones(len(P), bool))


def test_chain(tmp):
    data = os.path.join(tmp, "F01.npz")
    fake_file(data)
    start = datetime(2026, 11, 16, 8, 0, tzinfo=timezone.utc)
    rows = synthetic.station_rows(data, people=20, noise=0.0, start=start)
    assert len(rows) == len(st.STATIONS) * 61
    csv_path = os.path.join(tmp, "sensors.csv")
    synthetic.write_csv(rows, csv_path)

    recs = readers.read_server_csv(csv_path)
    assert len(recs) == len(rows)
    t_end = start.timestamp() + 900.0
    m = mapper.model_inputs(recs, t_end, people=20)

    assert np.allclose(m["V"], V_TRUE), m["V"]
    assert np.allclose(m["doors"], DOORS_TRUE), m["doors"]
    assert m["hist"].shape == (len(st.STATIONS), 12)

    # the mapped history must be the simulated CO2 at the station cells
    d = np.load(data)
    for i, s in enumerate(st.STATIONS):
        cell = int(np.argmin((d["P"][:, 0] - s.x) ** 2 + (d["P"][:, 1] - s.y) ** 2))
        true = d["C"][19:31, cell]          # 12 values up to t = 900 s
        assert np.allclose(m["hist"][i], true, atol=1e-3), (s.name, m["hist"][i], true)
    print("ok  synthetic data -> csv -> reader -> mapper gives back the simulation")
    return m


def test_closed_window(tmp):
    """A window whose distance sensor says closed must get speed 0, even if the speed sensor shows something."""
    recs = []
    for s in st.STATIONS:
        for k in range(13):
            recs.append({"time": 30.0 * k, "station": s.id, "co2": 500.0, "speed": 0.4, "distance": 20.0})
    m = mapper.model_inputs(recs, 360.0, people=30)
    assert np.all(m["V"] == 0) and np.all(m["doors"] == 0)
    assert np.allclose(m["hist"], 100.0)
    print("ok  closed windows and doors give speed 0")


def test_missing_data():
    recs = [{"time": 0.0, "station": st.STATIONS[0].id, "co2": 500.0, "speed": None, "distance": 300.0}]
    try:
        mapper.model_inputs(recs, 360.0, people=30)
    except ValueError as e:
        print("ok  missing data is reported:", e)
        return
    raise AssertionError("missing data was not noticed")


def test_model(tmp, m):
    files = []
    for k in range(3):
        f = os.path.join(tmp, f"F{k:02d}.npz")
        fake_file(f, seed=k)
        files.append(f)
    dev = torch.device("cpu")
    tr = SensorPlaneData(files[:2], dev, noise=5.0)
    te = SensorPlaneData(files[2:], dev)
    assert len(tr.hist_idx) == len(st.STATIONS)

    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    model = SensorForecastGNOT(d=32, layers=1, c_scale=50.0)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    first = last = None
    for it in range(150):
        qx, qh, hx, hh, V, N, target = tr.batch(4, 64, gen)
        mean, logvar = model(qx, qh, hx, hh, V, N)
        assert mean.shape == target.shape == (4, 64, 6)
        loss = nll(mean, logvar, target, 50.0)
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = float(loss) if first is None else first
        last = float(loss)
    assert last < first, (first, last)
    print(f"ok  sensor model trains (loss {first:.2f} -> {last:.2f})")

    # the output must not depend on the history at the points where we predict
    model.eval()
    qx, qh, hx, hh, V, N, _ = tr.batch(2, 16, gen)
    a, _ = model(qx, qh, hx, hh, V, N)
    b, _ = model(qx, torch.zeros_like(qh), hx, hh, V, N)
    assert torch.allclose(a, b)
    print("ok  the forecast uses only the station history")

    err, base, n = evaluate([model], te, every=8)
    assert np.isfinite(err) and n > 0
    print(f"ok  evaluation runs ({n} samples)")

    # mapped sensor values go into the model
    V = torch.tensor(np.concatenate([m["V"], m["doors"]])).unsqueeze(0)
    out, _ = model(te.xyz[:10].unsqueeze(0), torch.zeros(1, 10, 12), torch.tensor(m["xyz"]).unsqueeze(0),
                   torch.tensor(m["hist"]).unsqueeze(0), V, torch.tensor([[m["people"]]]))
    assert out.shape == (1, 10, 6) and torch.isfinite(out).all()
    print("ok  mapper output fits into the model")


def test_diagnose(tmp):
    """The diagnosis runs for a saved member and gives finite numbers for every column."""
    import diagnose_sensor as dg
    files = []
    for k in range(2):
        f = os.path.join(tmp, f"D{k:02d}.npz")
        fake_file(f, seed=10 + k)
        files.append(f)
    dev = torch.device("cpu")
    torch.manual_seed(0)
    model = SensorForecastGNOT(d=32, layers=1, c_scale=50.0)
    ck_dir = os.path.join(dg.HERE, "checkpoints", "_test_tmp")
    os.makedirs(ck_dir, exist_ok=True)
    try:
        torch.save({"state": model.state_dict(), "c_scale": 50.0, "d": 32, "layers": 1,
                    "stations": [s.name for s in st.STATIONS], "noise": 0.0}, os.path.join(ck_dir, "member0.pth"))
        models, ck = dg.load_members("_test_tmp", dev)
    finally:
        for f in os.listdir(ck_dir):
            os.remove(os.path.join(ck_dir, f))
        os.rmdir(ck_dir)
    data = SensorPlaneData(files, dev)
    rows = dg.diagnose(models, data, every=6, near=2.0)
    assert len(rows) == 2
    for r in rows:
        assert all(np.isfinite(v) for v in r.values()), r
        assert r["zero"] == 1.0
    print("ok  diagnosis runs for a saved member")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        m = test_chain(tmp)
        test_closed_window(tmp)
        test_missing_data()
        test_model(tmp, m)
        test_diagnose(tmp)
    print("ALL TESTS PASSED")
