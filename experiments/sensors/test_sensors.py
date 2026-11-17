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


def test_anchor(tmp):
    """The anchored model: interpolation of the station values, training, and the same output form."""
    h_xyz = torch.tensor([[[0.0, 0.0, 1.6], [10.0, 0.0, 1.6], [0.0, 8.0, 1.6]]])
    flat = torch.full((1, 3, 12), 120.0)
    q = torch.rand(1, 20, 3) * 10
    assert torch.allclose(SensorForecastGNOT.interpolate(q, h_xyz, flat), torch.full((1, 20, 12), 120.0), atol=1e-3)
    hh = torch.tensor([[[100.0] * 12, [200.0] * 12, [300.0] * 12]])
    at_station = SensorForecastGNOT.interpolate(h_xyz[:, :1], h_xyz, hh)
    assert abs(float(at_station[0, 0, 0]) - 100.0) < 15.0      # at a station the value is close to its own
    mid = SensorForecastGNOT.interpolate(torch.tensor([[[9.0, 0.5, 1.6]]]), h_xyz, hh)
    assert float(mid[0, 0, 0]) > 180.0                          # next to the second station: close to its value
    print("ok  interpolation of the station histories")

    files = []
    for k in range(3):
        f = os.path.join(tmp, f"AN{k:02d}.npz")
        fake_file(f, seed=20 + k)
        files.append(f)
    dev = torch.device("cpu")
    tr = SensorPlaneData(files[:2], dev, noise=5.0)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    model = SensorForecastGNOT(d=32, layers=1, c_scale=50.0, anchor=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    first = last = None
    for it in range(150):
        qx, qh, hx, hh, V, N, target = tr.batch(4, 64, gen)
        mean, logvar = model(qx, qh, hx, hh, V, N)
        assert mean.shape == target.shape == (4, 64, 6)
        loss = nll(mean, logvar, target, 50.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = float(loss) if first is None else first
        last = float(loss)
    assert last < first, (first, last)
    print(f"ok  anchored sensor model trains (loss {first:.2f} -> {last:.2f})")


def test_blind(tmp):
    """The blind model gives the same output whatever the station CO2 is."""
    f = os.path.join(tmp, "BL00.npz")
    fake_file(f, seed=30)
    dev = torch.device("cpu")
    tr = SensorPlaneData([f], dev)
    gen = torch.Generator().manual_seed(0)
    qx, qh, hx, hh, V, N, _ = tr.batch(2, 16, gen)
    for anchor in (False, True):
        torch.manual_seed(0)
        model = SensorForecastGNOT(d=32, layers=1, c_scale=50.0, anchor=anchor, blind=True).eval()
        a, _ = model(qx, qh, hx, hh, V, N)
        b, _ = model(qx, qh, hx, hh * 3 + 100.0, V, N)
        assert torch.allclose(a, b)
    print("ok  the blind model ignores the station CO2")


def test_history_length(tmp):
    """A longer station history: more steps in, later start times, same output form; 12 steps stays as before."""
    files = []
    for k in range(3):
        f = os.path.join(tmp, f"HL{k:02d}.npz")
        fake_file(f, seed=50 + k)
        files.append(f)
    dev = torch.device("cpu")
    base = SensorPlaneData(files[:2], dev)
    assert base.t0s[0][0] == 11
    for hist_len in (24, 40):
        tr = SensorPlaneData(files[:2], dev, noise=5.0, hist_len=hist_len, t0_min=39)
        te = SensorPlaneData(files[2:], dev, hist_len=hist_len, t0_min=39)
        assert tr.t0s[0][0] == 39 and len(tr.t0s[0]) == 61 - 6 - 39
        gen = torch.Generator().manual_seed(0)
        torch.manual_seed(0)
        model = SensorForecastGNOT(d=32, layers=1, c_scale=50.0, hist_len=hist_len)
        qx, qh, hx, hh, V, N, target = tr.batch(4, 32, gen)
        assert hh.shape == (4, len(st.STATIONS), hist_len)
        mean, logvar = model(qx, qh, hx, hh, V, N)
        assert mean.shape == target.shape == (4, 32, 6) and torch.isfinite(mean).all()
        err, base_err, n = evaluate([model], te, every=4)
        assert n > 0 and np.isfinite(err)
    # the anchored model also works with a longer history
    model = SensorForecastGNOT(d=32, layers=1, c_scale=50.0, anchor=True, hist_len=24)
    tr = SensorPlaneData(files[:2], dev, hist_len=24, t0_min=39)
    qx, qh, hx, hh, V, N, target = tr.batch(2, 16, torch.Generator().manual_seed(1))
    assert model(qx, qh, hx, hh, V, N)[0].shape == (2, 16, 6)
    print("ok  longer station history")


def test_station_dropout(tmp):
    """Station dropout: fewer stations in some steps, never fewer than 3, same values for the kept ones."""
    import train_sensor as ts
    hx = torch.rand(2, 8, 3)
    hh = torch.rand(2, 8, 12)
    gen = torch.Generator().manual_seed(0)
    assert ts.drop_stations(hx, hh, 0.0, gen)[0].shape[1] == 8                  # p = 0: nothing changes
    sizes = set()
    for _ in range(200):
        x2, h2 = ts.drop_stations(hx, hh, 1.0, gen)
        sizes.add(x2.shape[1])
        assert 4 <= x2.shape[1] <= 7 and h2.shape[1] == x2.shape[1]
        for j in range(x2.shape[1]):                                             # every kept station is a real one
            assert any(torch.equal(x2[0, j], hx[0, i]) and torch.equal(h2[0, j], hh[0, i]) for i in range(8))
    assert len(sizes) > 1
    model = SensorForecastGNOT(d=32, layers=1, c_scale=50.0)                    # the model accepts any number of stations
    qx = torch.rand(2, 5, 3)
    for n in (4, 8):
        out, _ = model(qx, torch.zeros(2, 5, 12), hx[:, :n], hh[:, :n], torch.zeros(2, 10), torch.full((2, 1), 30.0))
        assert out.shape == (2, 5, 6)
    print("ok  station dropout")


def test_relative_loss(tmp):
    """The relative loss scale follows the size of each sample; the normal one is a single number."""
    import train_sensor as ts
    big = torch.full((1, 8, 6), 200.0)
    calm = torch.full((1, 8, 6), 10.0)
    zero = torch.zeros(1, 8, 6)
    target = torch.cat([big, calm, zero])
    assert ts.loss_scale(target, 50.0, False, 5.0) == 50.0
    s = ts.loss_scale(target, 50.0, True, 5.0)
    assert s.shape == (3, 1, 1) and torch.allclose(s.flatten(), torch.tensor([200.0, 10.0, 5.0]))
    mean = target + 1.0
    logvar = torch.zeros_like(target)
    loss = nll(mean, logvar, target, s)
    assert torch.isfinite(loss)
    # the same absolute error counts 400 times more in the calm sample than in the big one (200 / 10 squared)
    per = [float(nll(mean[i:i + 1], logvar[i:i + 1], target[i:i + 1], s[i:i + 1])) for i in range(2)]
    assert per[1] > per[0]
    print("ok  relative loss scale")


def test_checks(tmp):
    """People-blind model, split by time (seen configuration), random split, error by time."""
    import train_sensor as ts
    import diagnose_sensor as dg
    files = []
    for k in range(4):
        f = os.path.join(tmp, f"CK{k:02d}.npz")
        fake_file(f, seed=60 + k)
        files.append(f)
    dev = torch.device("cpu")
    tr = SensorPlaneData(files[:2], dev)
    gen = torch.Generator().manual_seed(0)
    qx, qh, hx, hh, V, N, _ = tr.batch(2, 16, gen)
    torch.manual_seed(0)
    model = SensorForecastGNOT(d=32, layers=1, c_scale=50.0, people_blind=True).eval()
    a, _ = model(qx, qh, hx, hh, V, N)
    b, _ = model(qx, qh, hx, hh, V, N * 0 + 77.0)
    assert torch.allclose(a, b)                                       # the number of people does not matter
    normal = SensorForecastGNOT(d=32, layers=1, c_scale=50.0)
    assert float(normal(qx, qh, hx, hh, V, N)[0].sum()) != float(normal(qx, qh, hx, hh, V, N * 0 + 77.0)[0].sum())
    early = SensorPlaneData(files[:1], dev, t0_max=20)
    late = SensorPlaneData(files[:1], dev, t0_min=33)
    assert early.t0s[0][0] == 11 and early.t0s[0][-1] == 20 and late.t0s[0][0] == 33 and late.t0s[0][-1] == 54
    names_n = [f"N{i:02d}.npz" for i in range(1, 29)]
    names_a = [f"A{i:02d}.npz" for i in range(1, 41)]
    allf = names_n + names_a
    tr_f, te_f = ts.resplit(allf[:52], allf[52:], seed=3)
    assert len(tr_f) == 52 and len(te_f) == 16 and not set(tr_f) & set(te_f)
    assert sum(f.startswith("N") for f in te_f) == 8 and sum(f.startswith("A") for f in te_f) == 8
    assert ts.resplit(allf[:52], allf[52:], seed=3)[1] == te_f and ts.resplit(allf[:52], allf[52:], seed=4)[1] != te_f
    res = dg.by_time([normal.eval()], SensorPlaneData(files[2:], dev), every=4)
    assert all(np.isfinite(r["model"]) for r in res.values() if r["n"] > 0)
    dg.print_by_time(res)
    # direction-blind: an outlet window (negative speed) looks like an inlet window of the same size
    Vo = torch.zeros(2, 10)
    Vo[:, 1] = -1.0
    Vp = Vo.abs()
    torch.manual_seed(0)
    dmodel = SensorForecastGNOT(d=32, layers=1, c_scale=50.0, direction_blind=True).eval()
    assert torch.allclose(dmodel(qx, qh, hx, hh, Vo, N)[0], dmodel(qx, qh, hx, hh, Vp, N)[0])
    nmodel = SensorForecastGNOT(d=32, layers=1, c_scale=50.0).eval()
    assert not torch.allclose(nmodel(qx, qh, hx, hh, Vo, N)[0], nmodel(qx, qh, hx, hh, Vp, N)[0])
    print("ok  people-blind, time split, random split, error by time, direction-blind")


def test_physics(tmp):
    """The physics token: numbers of the well-mixed balance and a model that uses them."""
    import sensor_model as sm
    V = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0, 1, 0],            # one inlet window at 1 m/s, door 1 open
                      [0.0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])           # nothing open but both doors
    N = torch.tensor([[40.0], [40.0]])
    f = sm.physics_features(V, N)
    assert f.shape == (2, 6) and torch.isfinite(f).all()
    q = 1.0 * 0.61 * 3.15
    assert abs(float(f[0, 0]) - np.log10(q)) < 1e-5                           # log10 of the flow
    assert abs(float(f[0, 2]) - 40 * 3.96 / q / 500.0) < 5e-3                  # CO2 level the room approaches
    assert float(f[1, 0]) == -1.0 and float(f[1, 2]) == 1000.0 / 500.0          # no inlet: flow floor, level capped
    assert float(f[0, 3]) == 1 / 8 and float(f[0, 5]) == 0.5
    files = []
    for k in range(2):
        p = os.path.join(tmp, f"PH{k:02d}.npz")
        fake_file(p, seed=70 + k)
        files.append(p)
    tr = SensorPlaneData(files, torch.device("cpu"))
    gen = torch.Generator().manual_seed(0)
    qx, qh, hx, hh, V2, N2, target = tr.batch(2, 16, gen)
    torch.manual_seed(0)
    model = SensorForecastGNOT(d=32, layers=1, c_scale=50.0, physics=True).eval()
    out, _ = model(qx, qh, hx, hh, V2, N2)
    assert out.shape == (2, 16, 6) and torch.isfinite(out).all()
    out2, _ = model(qx, qh, hx, hh, V2 * 2.0, N2)
    assert not torch.allclose(out, out2)
    plain = SensorForecastGNOT(d=32, layers=1, c_scale=50.0)
    assert sum(p.numel() for p in model.parameters()) > sum(p.numel() for p in plain.parameters())
    print("ok  physics token")


def test_chain_demo(tmp):
    """The whole chain with a saved model: without noise the chain must give exactly the straight forecast."""
    import demo_chain as dc
    import diagnose_sensor as dg
    path = os.path.join(tmp, "DC00.npz")
    fake_file(path, seed=80)
    dev = torch.device("cpu")
    torch.manual_seed(0)
    model = SensorForecastGNOT(d=32, layers=1, c_scale=50.0, direction_blind=True)
    ck_dir = os.path.join(dg.HERE, "checkpoints", "_test_tmp2")
    os.makedirs(ck_dir, exist_ok=True)
    try:
        torch.save({"state": model.state_dict(), "c_scale": 50.0, "d": 32, "layers": 1,
                    "stations": [s.name for s in st.STATIONS], "noise": 0.0, "direction_blind": True},
                   os.path.join(ck_dir, "member0.pth"))
        models, ck = dg.load_members("_test_tmp2", dev)
    finally:
        for f in os.listdir(ck_dir):
            os.remove(os.path.join(ck_dir, f))
        os.rmdir(ck_dir)
    res = dc.run_case(models, path, people=30.0, steps=[20, 30], noise=0.0)
    assert len(res) == 2
    for r in res:
        assert r["v_ok"], r
        assert abs(r["chain"] - r["direct"]) < 1e-3, r                 # reader + mapper keep the history exactly
        assert all(np.isfinite(r[k]) for k in ("chain", "direct", "persistence", "seat_ppm"))
    noisy = dc.run_case(models, path, people=30.0, steps=[20], noise=10.0)[0]
    assert noisy["v_ok"] and np.isfinite(noisy["chain"])
    print("ok  the whole chain (sensor file, reader, mapper, model)")


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

    err, counts, margin_counts = dg.decisions(models, data, every=6, thresholds=(50.0, 100.0))
    assert all(np.isfinite(v).all() for v in err.values()) and len(err["seat_true"]) > 0
    for c in counts.values():
        for name in ("model", "persistence"):
            assert c[name].sum() == len(err["seat_true"])           # every sample is counted once
    for (t, m), c in margin_counts.items():
        assert c.sum() == len(err["seat_true"])
        if m == 0:
            assert (c == counts[t]["model"]).all()                  # margin 0 is the normal warning
    # a larger margin never finds fewer events
    for t in (50.0, 100.0):
        found = [margin_counts[(t, m)][0] for m in (0.0, 25.0, 50.0, 75.0)]
        assert found == sorted(found)
    dg.print_decisions(err, counts, margin_counts)
    print("ok  the decision test runs")


def test_virtual(tmp):
    """Virtual stations: names, positions, and a member trained with them can be loaded by name."""
    for n, count in (("1", 1), ("2", 2), ("4", 4), ("16", 16), ("front", 1), ("back", 1), ("windowside", 1),
                     ("doorside", 1)):
        v = st.virtual_stations(n)
        assert len(v) == count and all(s.kind == "virtual" for s in v)
        assert all(3.0 <= s.x <= 12.5 and 2.0 <= s.y <= 7.0 for s in v)       # inside the seating area
    assert st.virtual_stations(0) == [] and st.virtual_stations("0") == []
    every = [s.name for k in st.VIRTUAL_SETS for s in st.virtual_stations(k)]
    assert len(set(every)) == len(every)                                     # names are unique
    names = [s.name for s in st.STATIONS + st.virtual_stations(2)]
    assert [s.name for s in st.by_name(names)] == names and len(set(names)) == len(names)
    files = []
    for k in range(2):
        f = os.path.join(tmp, f"VS{k:02d}.npz")
        fake_file(f, seed=40 + k)
        files.append(f)
    data = SensorPlaneData(files, torch.device("cpu"), stations=st.by_name(names))
    assert len(data.hist_idx) == len(st.STATIONS) + 2
    print("ok  virtual stations")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        m = test_chain(tmp)
        test_closed_window(tmp)
        test_missing_data()
        test_model(tmp, m)
        test_anchor(tmp)
        test_blind(tmp)
        test_virtual(tmp)
        test_history_length(tmp)
        test_station_dropout(tmp)
        test_relative_loss(tmp)
        test_checks(tmp)
        test_physics(tmp)
        test_chain_demo(tmp)
        test_diagnose(tmp)
    print("ALL TESTS PASSED")
