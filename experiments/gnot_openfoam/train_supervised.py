"""
Supervised training of GNOT on the laminar OpenFOAM data.
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


def load(path, dev, store="auto"):
    """Load the dataset files onto the device."""
    d = np.load(path)
    ds = {k: d[k] for k in d.files}
    ds["name"] = os.path.basename(path)
    if "t" in ds:
        ds["t_u"], ds["t_c"] = ds["t"], ds["t"]
    t_u, t_c = [float(v) for v in ds["t_u"]], [float(v) for v in ds["t_c"]]
    umap = []
    for t in t_c:
        if t <= t_u[-1] + 1e-6:
            hits = [i for i, s in enumerate(t_u) if abs(s - t) < 1e-6]
            assert hits, f"{ds['name']}: CO2 time {t} has no velocity snapshot"
            umap.append(hits[0])
        else:
            umap.append(len(t_u) - 1)
    ds["t"] = ds["t_c"]
    sdev = store if store != "auto" else dev
    ds["U_d"] = torch.tensor(ds["U"][umap], dtype=torch.float32, device=sdev)
    for k in ("P", "C", "t", "V"):
        ds[k + "_d"] = torch.tensor(ds[k], dtype=torch.float32, device=sdev)
    ds["plane_d"] = torch.tensor(ds["plane"], device=sdev)
    ds["dev"] = dev
    return ds


def predict(model, ds, ti, dev, n_people=None, batch=16384):
    """Full field at dataset time index ti -> (U (n,3), C (n,)) as torch tensors."""
    P = ds["P_d"].to(dev)
    n = P.shape[0]
    t = float(ds["t"][ti])
    N = float(ds["N_ref"]) if n_people is None else n_people
    Us, Cs = [], []
    with torch.no_grad():
        for i in range(0, n, batch):
            Q = P[i:i + batch]
            b = Q.shape[0]
            u, v, w, C = model(Q[:, 0:1], Q[:, 1:2], Q[:, 2:3], torch.full((b, 1), t, device=dev),
                               ds["V_d"].to(dev).view(1, -1).expand(b, -1), torch.full((b, 1), N, device=dev))
            Us.append(torch.cat([u, v, w], 1))
            Cs.append(C.squeeze(1))
    return torch.cat(Us), torch.cat(Cs)


def rel(a, b):
    den = torch.linalg.norm(b)
    return float("nan") if den == 0 else (torch.linalg.norm(a - b) / den).item()


def metrics(model, ds, ti, dev):
    U, C = predict(model, ds, ti, dev)
    Ur, Cr = ds["U_d"][ti].to(dev), ds["C_d"][ti].to(dev)
    pl = ds["plane_d"].to(dev)
    return {"vel_vol": rel(U, Ur), "vel_plane": rel(U[pl], Ur[pl]),
            "co2_plane": rel(C[pl], Cr[pl]), "co2_vol": rel(C, Cr),
            "mass": (C.sum() / Cr.sum()).item() if Cr.sum() > 0 else float("nan")}


def report(model, sets, dev, holdout):
    for ds, label in sets:
        for ti, t in enumerate(ds["t"]):
            t = float(t)
            if t in (30.0, 60.0, 120.0, 600.0, 1800.0) or t in holdout:
                m = metrics(model, ds, ti, dev)
                kind = "held-out time" if (t in holdout and label == "train") else label
                print(f"    {ds['name']:28s} t={t:5.0f} [{kind:13s}] velocity {100 * m['vel_vol']:5.1f}% vol "
                      f"{100 * m['vel_plane']:5.1f}% plane | CO2 {100 * m['co2_plane']:5.1f}% plane "
                      f"{100 * m['co2_vol']:5.1f}% vol, mass {m['mass']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", default=[], help="training datasets (npz)")
    ap.add_argument("--scenarios", default=None,
                    help="Phase 2: take train/test cases from this file (scenarios.txt) -> data/<name>.npz")
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
    ap.add_argument("--data-on-cpu", action="store_true", help="keep datasets in CPU RAM (many cases)")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(args.seed)
    out_dir = os.path.join(CKPT_DIR, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    if any(f.endswith(".pth") for f in os.listdir(out_dir)):
        raise SystemExit(f"{out_dir} already has checkpoints -- use a new --tag")

    if args.scenarios:
        rows = [l.split() for l in open(args.scenarios) if l.strip() and not l.startswith("#")]
        path = lambda n: os.path.join(common.DATA_DIR, n + ".npz")
        missing = [r[0] for r in rows if not os.path.isfile(path(r[0]))]
        if missing:
            raise SystemExit(f"not extracted yet: {missing}")
        args.data = [path(r[0]) for r in rows if r[1] == "train"]
        args.test_data = [path(r[0]) for r in rows if r[1] == "test"]
        assert not set(args.data) & set(args.test_data), "a test case is also in the training set"
    if not args.data:
        raise SystemExit("give --data or --scenarios")
    store = "cpu" if args.data_on_cpu else "auto"
    train = [load(p, dev, store) for p in args.data]
    test = [load(p, dev, store) for p in args.test_data]
    hold = set(float(t) for t in args.holdout_times)
    tr_idx = [[i for i, t in enumerate(ds["t"]) if float(t) not in hold] for ds in train]
    for ds, ti in zip(train, tr_idx):
        assert ti, f"{ds['name']}: no training times left"
    u2 = np.mean([float((ds["U_d"][ti] ** 2).sum(-1).mean()) for ds, ti in zip(train, tr_idx)])
    c2 = np.mean([float((ds["C_d"][ti] ** 2).mean()) for ds, ti in zip(train, tr_idx)])
    print(f"train: {[ds['name'] for ds in train]} (times {[float(ds['t'][i]) for i in tr_idx[0]]}); "
          f"held-out times {sorted(hold)}; test cases {[ds['name'] for ds in test]}; "
          f"scales |u|^2 {u2:.3e}, C^2 {c2:.3e}; device {dev}")

    t_horizon = max(float(ds["t"][-1]) for ds in train + test)
    model = SupervisedGNOT(t_horizon=max(t_horizon, 120.0)).to(dev)
    print(f"time horizon {model.t_horizon:g} s")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_at = lambda it: args.lr * 0.9 ** (it / 2000)
    t0 = time.time()
    for it in range(args.iters + 1):
        for gpar in opt.param_groups:
            gpar["lr"] = lr_at(it)
        k = int(torch.randint(len(train), (1,)).item())
        ds = train[k]
        sdev = ds["P_d"].device
        ti = torch.tensor(tr_idx[k], device=sdev)[torch.randint(len(tr_idx[k]), (args.batch,), device=sdev)]
        pi = torch.randint(ds["P_d"].shape[0], (args.batch,), device=sdev)
        Q = ds["P_d"][pi].to(dev)
        t = ds["t_d"][ti].unsqueeze(1).to(dev)
        V = ds["V_d"].to(dev).view(1, -1).expand(args.batch, -1)
        N = torch.full((args.batch, 1), float(ds["N_ref"]), device=dev)
        u, v, w, C = model(Q[:, 0:1], Q[:, 1:2], Q[:, 2:3], t, V, N)
        Ur, Cr = ds["U_d"][ti, pi].to(dev), ds["C_d"][ti, pi].to(dev)
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
            torch.save({"iter": it, "model_state": model.state_dict(), "args": vars(args), "t_horizon": model.t_horizon,
                        "scales": (u2, c2), "train": args.data, "holdout_times": sorted(hold)},
                       os.path.join(out_dir, f"sup_{args.tag}_iter{it}.pth"))
            print(f"  eval at iter {it}:")
            report(model, [(d, "train") for d in train[:3]] + [(d, "TEST case") for d in test], dev, hold)
    print("done")


if __name__ == "__main__":
    main()
