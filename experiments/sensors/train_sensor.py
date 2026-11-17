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


def loss_scale(target, c_scale, relative, floor):
    """Scale of the loss. Normal: one number for all samples (ppm). Relative: the size of each sample's own signal,
    so that calm scenarios with a small CO2 rise count as much as strong ones (like in the relative l2 error)."""
    if not relative:
        return c_scale
    return target.pow(2).mean((1, 2), keepdim=True).sqrt().clamp_min(floor).detach()


def resplit(train_files, test_files, seed):
    """A random split of all scenarios with 8 N and 8 A scenarios for the test, like the fixed split."""
    allf = list(train_files) + list(test_files)
    names = [os.path.basename(f) for f in allf]
    n_idx = [i for i, n in enumerate(names) if n.startswith("N")]
    a_idx = [i for i, n in enumerate(names) if n.startswith("A")]
    rng = np.random.default_rng(seed)
    test = set(rng.choice(n_idx, 8, replace=False).tolist()) | set(rng.choice(a_idx, 8, replace=False).tolist())
    return [f for i, f in enumerate(allf) if i not in test], [f for i, f in enumerate(allf) if i in test]


def drop_stations(hx, hh, p, gen):
    """With probability p hide some stations (between half and all but one) for this training step.
    The model must then not rely on one exact station. hx: (B, S, 3), hh: (B, S, steps)."""
    S = hx.shape[1]
    if p <= 0 or S < 4 or float(torch.rand(1, generator=gen)) >= p:
        return hx, hh
    n_keep = int(torch.randint(max(3, S // 2), S, (1,), generator=gen))
    keep = torch.sort(torch.randperm(S, generator=gen)[:n_keep]).values.to(hx.device)
    return hx[:, keep], hh[:, keep]


def train_one(k, tr, te, a, dev, c_scale):
    torch.manual_seed(1000 + k)
    gen = torch.Generator().manual_seed(2000 + k)
    model = SensorForecastGNOT(d=a.d, layers=a.layers, c_scale=c_scale, anchor=a.anchor, blind=a.blind,
                               hist_len=a.hist, people_blind=a.people_blind, physics=a.physics).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.05)
    start, losses = time.time(), []
    for it in range(1, a.iters + 1):
        qx, qh, hx, hh, V, N, target = tr.batch(a.batch, a.queries, gen)
        hx, hh = drop_stations(hx, hh, a.station_dropout, gen)
        mean, logvar = model(qx, qh, hx, hh, V, N)
        loss = nll(mean, logvar, target, loss_scale(target, c_scale, a.relative_loss, a.rel_floor))
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
                "stations": a.stations, "noise": a.noise, "anchor": a.anchor,
                "blind": a.blind, "hist": a.hist, "t0_min": a.t0_min, "people_blind": a.people_blind,
                "physics": a.physics},
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
    ap.add_argument("--virtual", default="0", choices=["0"] + list(st.VIRTUAL_SETS),
                    help="what-if test: add a set of virtual stations in the seating area (not in the real room)")
    ap.add_argument("--relative-loss", action="store_true",
                    help="weigh every training sample by the size of its own signal (like the relative error)")
    ap.add_argument("--rel-floor", type=float, default=5.0, help="smallest signal size [ppm] used for that weighting")
    ap.add_argument("--wd", type=float, default=1e-4, help="weight decay of the optimizer")
    ap.add_argument("--station-dropout", type=float, default=0.0,
                    help="chance per training step that some stations are hidden (against overfitting)")
    ap.add_argument("--hist", type=int, default=12, help="number of 30 s steps of station history the model gets")
    ap.add_argument("--t0-min", type=int, default=0,
                    help="first time step used (so that runs with different histories use the same samples)")
    ap.add_argument("--n-train", type=int, default=0, help="use only this many (randomly chosen) training scenarios, 0 = all")
    ap.add_argument("--physics", action="store_true",
                    help="give the model the well-mixed room balance (fresh-air flow, final CO2 level, time constant)")
    ap.add_argument("--people-blind", action="store_true",
                    help="test: the model is not told the number of people (the real system does not know it)")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="0 = the fixed split of the scenario file; otherwise a random split with 8 N and 8 A test scenarios")
    ap.add_argument("--seen-config", action="store_true",
                    help="test: the test scenarios are the training scenarios at LATER times (window setting seen in training)")
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
    if a.split_seed:
        train_files, test_files = resplit(train_files, test_files, a.split_seed)
    if a.n_train:
        keep = np.sort(np.random.default_rng(0).choice(len(train_files), a.n_train, replace=False))
        train_files = [train_files[i] for i in keep]
    used = st.STATIONS + st.virtual_stations(a.virtual)
    a.stations = [s.name for s in used]
    t0_split = 32                 # --seen-config: train on the time steps up to this one, test on the later ones
    if a.seen_config:
        tr = SensorPlaneData(train_files, dev, z=a.z, transitions=a.transitions, noise=a.noise, stations=used,
                             hist_len=a.hist, t0_min=a.t0_min, t0_max=t0_split)
        te = SensorPlaneData(train_files, dev, z=a.z, transitions=a.transitions, stations=used,
                             hist_len=a.hist, t0_min=t0_split + 1)
        test_files = train_files
    else:
        tr = SensorPlaneData(train_files, dev, z=a.z, transitions=a.transitions, noise=a.noise, stations=used,
                             hist_len=a.hist, t0_min=a.t0_min)
        te = SensorPlaneData(test_files, dev, z=a.z, transitions=a.transitions, stations=used,
                             hist_len=a.hist, t0_min=a.t0_min)
    print("test scenarios: " + " ".join(os.path.basename(f).replace(".npz", "") for f in test_files), flush=True)
    c_scale = c_scale_of(tr)
    print(f"{len(train_files)} train / {len(test_files)} test scenarios, {len(tr.hist_idx)} stations, "
          f"{tr.xyz.shape[0]} plane points, device {dev}", flush=True)

    models = [train_one(k, tr, te, a, dev, c_scale) for k in range(a.members)]
    err, base, n = evaluate(models, te)
    print(f"\ntest: ensemble {100 * err:.2f}%  persistence {100 * base:.2f}%  ({n} samples)")
    print("note: persistence here uses the true CO2 map of now, which the sensors do not have")


if __name__ == "__main__":
    main()
