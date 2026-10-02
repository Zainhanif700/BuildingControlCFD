"""
Train the CO2 forecasting operator (forecast.py) as in Bian & Shi (2025): last 6 min -> next 3 min on the
breathing plane, Gaussian NLL, AdamW, ensemble of 5 members (different seeds), l2 error of eq. 12.

Works on both datasets (same scenario names / split):
  laminar  --data-dir ../gnot_openfoam/data      (test run while the RANS dataset is computed)
  RANS     --data-dir data                        (the real target, option B2)

Usage (training env, from experiments/gnot_openfoam_rans):
  python3 train_forecast.py --data-dir ../gnot_openfoam/data --tag lam_fc --members 5 --iters 20000
Output: checkpoints/<tag>/member<k>.pth, logs to stdout; at the end the ensemble test errors
(per case and mean) and the persistence baseline ('CO2 stays as it is').
"""
import argparse
import os
import time

import numpy as np
import torch

import common  # noqa: F401
from forecast import ForecastGNOT, PlaneData, nll, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = os.path.join(HERE, "..", "gnot_openfoam", "scenarios.txt")


def split(data_dir, scen, with_s00):
    rows = [l.split() for l in open(scen) if l.strip() and not l.startswith("#")]
    rows = [r for r in rows if with_s00 or r[2] != "0,0,0,0,0,0,0,0"]
    path = lambda n: os.path.join(data_dir, n + ".npz")
    missing = [r[0] for r in rows if not os.path.isfile(path(r[0]))]
    if missing:
        raise SystemExit(f"not in {data_dir}: {missing}")
    tr = [path(r[0]) for r in rows if r[1] == "train"]
    te = [path(r[0]) for r in rows if r[1] == "test"]
    assert tr and te and not set(tr) & set(te)
    return tr, te


def c_scale_of(data, n_typ=45.0):
    """RMS CO2 of the training frames at a typical occupancy (middle of 10-80)."""
    sq = [float((f * (n_typ / n)).pow(2).mean()) for f, n in zip(data.frames, data.N_ref)]
    return float(np.sqrt(np.mean(sq)))


def train_member(k, tr, te, args, dev, c_scale):
    torch.manual_seed(1000 + k)
    gen = torch.Generator().manual_seed(2000 + k)
    model = ForecastGNOT(d=args.d, layers=args.layers, c_scale=c_scale).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.05)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"member {k}: {n_par / 1e3:.0f}k parameters, c_scale {c_scale:.4g}", flush=True)
    t0, run = time.time(), []
    for it in range(1, args.iters + 1):
        qx, qh, hx, hh, V, N, tg = tr.batch(args.batch, args.queries, gen)
        mean, logvar = model(qx, qh, hx, hh, V, N)
        loss = nll(mean, logvar, tg, c_scale)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run.append(float(loss))
        if it % args.log_every == 0 or it == args.iters:
            msg = f"  m{k} it {it:6d}  NLL {np.mean(run):+.4f}  lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f} s"
            run = []
            if it % args.eval_every == 0 or it == args.iters:
                model.eval()
                e_tr, _, _ = evaluate([model], tr, every=4)
                e_te, b_te, _ = evaluate([model], te, every=4)
                model.train()
                msg += f"  | l2 train {100 * e_tr:.2f}%  test {100 * e_te:.2f}%  (persistence {100 * b_te:.2f}%)"
            print(msg, flush=True)
    path = os.path.join(args.out, f"member{k}.pth")
    torch.save({"state": model.state_dict(), "c_scale": c_scale, "d": args.d, "layers": args.layers,
                "data_dir": args.data_dir, "hist_idx": tr.hist_idx.cpu()}, path)
    print(f"  saved {path}", flush=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--scenarios", default=SCEN)
    ap.add_argument("--with-s00", action="store_true", help="keep S00 (all windows closed); default off as in RANS")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--queries", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out = os.path.join(HERE, "checkpoints", args.tag)
    os.makedirs(args.out, exist_ok=True)
    if any(f.endswith(".pth") for f in os.listdir(args.out)):
        raise SystemExit(f"{args.out} already has checkpoints -- use a new --tag")
    dev = torch.device(args.device)
    trp, tep = split(args.data_dir, args.scenarios, args.with_s00)
    tr = PlaneData(trp, dev, seed=0)
    te = PlaneData(tep, dev, seed=0)
    assert torch.equal(tr.hist_idx, te.hist_idx)
    c_scale = c_scale_of(tr)
    print(f"data {args.data_dir}: {len(trp)} train / {len(tep)} test cases, {tr.xyz.shape[0]} plane points "
          f"(z = {float(tr.xyz[0, 2]):.2f} m), {sum(map(len, tr.t0s))} train samples per occupancy; device {dev}", flush=True)

    models = [train_member(k, tr, te, args, dev, c_scale) for k in range(args.members)]

    print("\nFINAL (every sample, occupancy = N_ref of the data)")
    for name, d in (("train", tr), ("test", te)):
        e1, b, n = evaluate(models[:1], d)
        eE, _, _ = evaluate(models, d)
        print(f"  {name:5s}: member 0 {100 * e1:.2f}%  ensemble({len(models)}) {100 * eE:.2f}%  persistence {100 * b:.2f}%  [{n} samples]")
    for N in (10.0, 80.0):
        eE, b, _ = evaluate(models, te, N=N)
        print(f"  test at N = {N:g}: ensemble {100 * eE:.2f}%  persistence {100 * b:.2f}%")
    print("  per test case (ensemble / persistence):")
    for k, c in enumerate(te.cases):
        sub = PlaneData.__new__(PlaneData)
        sub.__dict__.update(te.__dict__)
        sub.cases, sub.frames, sub.V, sub.N_ref, sub.t0s = [c], [te.frames[k]], [te.V[k]], [te.N_ref[k]], [te.t0s[k]]
        eE, b, _ = evaluate(models, sub)
        print(f"    {c['name']:10s} {100 * eE:6.2f}%  {100 * b:6.2f}%")
    print("Paper (Bian & Shi 2025): l2 train 5.9 %, test 10.9 %.")


if __name__ == "__main__":
    main()
