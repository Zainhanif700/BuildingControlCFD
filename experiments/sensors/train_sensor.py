"""
Trains the sensor version of the forecast model and prints the test error.
Usage: python train_sensor.py --data-dir ../gnot_openfoam_rans/data_transitions_v4 --transitions \
           --scenarios ../gnot_openfoam_rans/scenarios_v4.txt --tag sensors_v4 --noise 10
"""
import argparse
import os
import time

import numpy as np
import torch

import stations as st
from sensor_model import SensorForecastGNOT, SensorPlaneData
from forecast import nll, evaluate
from train_forecast import split, c_scale_of

HERE = os.path.dirname(os.path.abspath(__file__))


def train_one(k, tr, te, a, dev, c_scale):
    torch.manual_seed(1000 + k)
    gen = torch.Generator().manual_seed(2000 + k)
    model = SensorForecastGNOT(d=a.d, layers=a.layers, c_scale=c_scale, anchor=a.anchor, blind=a.blind).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.05)
    start, losses = time.time(), []
    for it in range(1, a.iters + 1):
        qx, qh, hx, hh, V, N, target = tr.batch(a.batch, a.queries, gen)
        mean, logvar = model(qx, qh, hx, hh, V, N)
        loss = nll(mean, logvar, target, c_scale)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss))
        if it % a.log_every == 0 or it == a.iters:
            model.eval()
            e_te, b_te, _ = evaluate([model], te, every=4)
            model.train()
            print(f"  m{k} it {it:6d}  loss {np.mean(losses):+.4f}  test {100 * e_te:.2f}%  "
                  f"(persistence {100 * b_te:.2f}%)  {time.time() - start:.0f} s", flush=True)
            losses = []
    torch.save({"state": model.state_dict(), "c_scale": c_scale, "d": a.d, "layers": a.layers,
                "stations": [s.name for s in st.STATIONS], "noise": a.noise, "anchor": a.anchor,
                "blind": a.blind},
               os.path.join(a.out, f"member{k}.pth"))
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--transitions", action="store_true")
    ap.add_argument("--anchor", action="store_true", help="interpolate the station histories to every point (see sensor_model.py)")
    ap.add_argument("--blind", action="store_true", help="test: the model does not get the station CO2")
    ap.add_argument("--n-train", type=int, default=0, help="use only this many (randomly chosen) training scenarios, 0 = all")
    ap.add_argument("--z", type=float, default=1.6)
    ap.add_argument("--noise", type=float, default=10.0, help="noise on the station CO2 in training [ppm]")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--queries", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=5000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.out = os.path.join(HERE, "checkpoints", a.tag)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)

    train_files, test_files = split(a.data_dir, a.scenarios, with_s00=False)
    if a.n_train:
        keep = np.sort(np.random.default_rng(0).choice(len(train_files), a.n_train, replace=False))
        train_files = [train_files[i] for i in keep]
    tr =SensorPlaneData(train_files, dev, z=a.z, transitions=a.transitions, noise=a.noise)
    te = SensorPlaneData(test_files, dev, z=a.z, transitions=a.transitions)
    c_scale = c_scale_of(tr)
    print(f"{len(train_files)} train / {len(test_files)} test scenarios, {len(tr.hist_idx)} stations, "
          f"{tr.xyz.shape[0]} plane points, device {dev}", flush=True)

    models = [train_one(k, tr, te, a, dev, c_scale) for k in range(a.members)]
    err, base, n = evaluate(models, te)
    print(f"\ntest: ensemble {100 * err:.2f}%  persistence {100 * base:.2f}%  ({n} samples)")
    print("note: persistence here uses the true CO2 map of now, which the sensors do not have")


if __name__ == "__main__":
    main()
