"""
Supervised GNOT on OpenFOAM data (Track 2). Loss = relative MSE of velocity + relative MSE of CO2 on
random (point, time) samples of the extracted datasets (extract_case.py). No PDE residuals.

Phase 1 (pipeline check, ONE case): some dataset times are held out (--holdout-times) and reported
separately -- a first, weak generalisation signal (interpolation in time). Fitting the training times
well only shows that the pipeline works; generalisation to new window settings needs held-out CASES
(Phase 2, --test-data).

Metrics (same definitions as experiments/gnot/openfoam/compare_with_openfoam.py): relative L2 of the
velocity vector (volume / breathing plane), relative L2 of CO2 (plane / volume), CO2 mass ratio.

Usage (training env, from experiments/gnot_openfoam):
  PYTHONUNBUFFERED=1 python3 train_supervised.py --data data/W1_1ms_dx0.1.npz --tag p1 2>&1 | tee logs/p1.log
"""
import argparse
import math
import os
import time

import numpy as np
import torch

import common
from common import CKPT_DIR
from model import SupervisedGNOT


def load(path, dev):
    d = np.load(path)
    ds = {k: d[k] for k in d.files}
    ds["name"] = os.path.basename(path)
    for k in ("P", "U", "C", "t", "V"):
        ds[k + "_d"] = torch.tensor(ds[k], dtype=torch.float32, device=dev)
    ds["plane_d"] = torch.tensor(ds["plane"], device=dev)
    return ds


def predict(model, ds, ti, dev, n_people=None, batch=16384):
    """Full field at dataset time index ti -> (U (n,3), C (n,)) as torch tensors."""
    P = ds["P_d"]
    n = P.shape[0]
    t = float(ds["t"][ti])
    N = float(ds["N_ref"]) if n_people is None else n_people
    Us, Cs = [], []
    with torch.no_grad():
        for i in range(0, n, batch):
            Q = P[i:i + batch]
            b = Q.shape[0]
            u, v, w, C = model(Q[:, 0:1], Q[:, 1:2], Q[:, 2:3], torch.full((b, 1), t, device=dev),
                               ds["V_d"].view(1, -1).expand(b, -1), torch.full((b, 1), N, device=dev))
            Us.append(torch.cat([u, v, w], 1))
            Cs.append(C.squeeze(1))
    return torch.cat(Us), torch.cat(Cs)


def rel(a, b):
    den = torch.linalg.norm(b)
    return float("nan") if den == 0 else (torch.linalg.norm(a - b) / den).item()


def metrics(model, ds, ti, dev):
    U, C = predict(model, ds, ti, dev)
    Ur, Cr = ds["U_d"][ti], ds["C_d"][ti]
    pl = ds["plane_d"]
    return {"vel_vol": rel(U, Ur), "vel_plane": rel(U[pl], Ur[pl]),
            "co2_plane": rel(C[pl], Cr[pl]), "co2_vol": rel(C, Cr),
            "mass": (C.sum() / Cr.sum()).item() if Cr.sum() > 0 else float("nan")}


def report(model, sets, dev, holdout):
    for ds, label in sets:
        for ti, t in enumerate(ds["t"]):
            t = float(t)
            if t in (30.0, 60.0, 120.0) or t in holdout:
                m = metrics(model, ds, ti, dev)
                kind = "held-out time" if (t in holdout and label == "train") else label
                print(f"    {ds['name']:28s} t={t:5.0f} [{kind:13s}] velocity {100 * m['vel_vol']:5.1f}% vol "
                      f"{100 * m['vel_plane']:5.1f}% plane | CO2 {100 * m['co2_plane']:5.1f}% plane "
                      f"{100 * m['co2_vol']:5.1f}% vol, mass {m['mass']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="training datasets (npz)")
    ap.add_argument("--test-data", nargs="*", default=[], help="held-out CASES (Phase 2)")
    ap.add_argument("--holdout-times", type=float, nargs="*", default=[30.0, 90.0],
                    help="dataset times never used for training (Phase 1: interpolation-in-time check)")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(args.seed)
    out_dir = os.path.join(CKPT_DIR, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    if any(f.endswith(".pth") for f in os.listdir(out_dir)):
        raise SystemExit(f"{out_dir} already has checkpoints -- use a new --tag")

    train = [load(p, dev) for p in args.data]
    test = [load(p, dev) for p in args.test_data]
    hold = set(float(t) for t in args.holdout_times)
    tr_idx = [[i for i, t in enumerate(ds["t"]) if float(t) not in hold] for ds in train]
    for ds, ti in zip(train, tr_idx):
        assert ti, f"{ds['name']}: no training times left"
    # loss scales: mean squared target over the training samples (-> relative MSE, both terms O(1))
    u2 = np.mean([float((ds["U_d"][ti] ** 2).sum(-1).mean()) for ds, ti in zip(train, tr_idx)])
    c2 = np.mean([float((ds["C_d"][ti] ** 2).mean()) for ds, ti in zip(train, tr_idx)])
    print(f"train: {[ds['name'] for ds in train]} (times {[float(ds['t'][i]) for i in tr_idx[0]]}); "
          f"held-out times {sorted(hold)}; test cases {[ds['name'] for ds in test]}; "
          f"scales |u|^2 {u2:.3e}, C^2 {c2:.3e}; device {dev}")

    model = SupervisedGNOT().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_at = lambda it: args.lr * 0.9 ** (it / 2000)       # same schedule as Track 1 (Expert's Guide)
    t0 = time.time()
    for it in range(args.iters + 1):
        for gpar in opt.param_groups:
            gpar["lr"] = lr_at(it)
        k = int(torch.randint(len(train), (1,)).item())    # one case per step
        ds = train[k]
        ti = torch.tensor(tr_idx[k], device=dev)[torch.randint(len(tr_idx[k]), (args.batch,), device=dev)]
        pi = torch.randint(ds["P_d"].shape[0], (args.batch,), device=dev)
        Q = ds["P_d"][pi]
        t = ds["t_d"][ti].unsqueeze(1)
        V = ds["V_d"].view(1, -1).expand(args.batch, -1)
        N = torch.full((args.batch, 1), float(ds["N_ref"]), device=dev)
        u, v, w, C = model(Q[:, 0:1], Q[:, 1:2], Q[:, 2:3], t, V, N)
        Ur, Cr = ds["U_d"][ti, pi], ds["C_d"][ti, pi]
        L_u = ((torch.cat([u, v, w], 1) - Ur) ** 2).sum(1).mean() / u2
        L_c = ((C.squeeze(1) - Cr) ** 2).mean() / c2
        loss = L_u + L_c
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 200 == 0:
            print(f"[{it:6d}/{args.iters}] loss {loss.item():.4e} (u {L_u.item():.3e}, C {L_c.item():.3e}) "
                  f"lr {lr_at(it):.2e} | {(it + 1) / (time.time() - t0):.1f} it/s", flush=True)
        if it % args.eval_every == 0 and it > 0 or it == args.iters:
            print(f"  eval at iter {it}:")
            report(model, [(d, "train") for d in train] + [(d, "TEST case") for d in test], dev, hold)
            torch.save({"iter": it, "model_state": model.state_dict(), "args": vars(args),
                        "scales": (u2, c2), "train": args.data, "holdout_times": sorted(hold)},
                       os.path.join(out_dir, f"sup_{args.tag}_iter{it}.pth"))
    print("done")


if __name__ == "__main__":
    main()
