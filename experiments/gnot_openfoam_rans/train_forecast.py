"""
Trains the forecast ensemble (5 networks) and prints the test errors and the 'CO2 stays the same' baseline.
Usage: python3 train_forecast.py --data-dir data_transitions --transitions --tag rans_tr --z 1.6 --c-offset 400
"""
import argparse
import os
import time

import numpy as np
import torch

import common
from forecast import ForecastGNOT, PlaneData, nll, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = os.path.join(HERE, "scenarios.txt")


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
    """Typical CO2 level of the training data (used to scale the inputs)."""
    sq = []
    for k in range(len(data.cases)):
        f = data.frames[k] * (n_typ / data.N_ref[k])
        if data.H is not None:
            f = f + data.H[k] * (n_typ / data.N_ref[k])
        sq.append(float(f.pow(2).mean()))
    return float(np.sqrt(np.mean(sq)))


def d_scale_of(data, n_typ=45.0):
    """Typical 3-minute change of the CO2 (used to scale the outputs)."""
    sq = []
    for k, (f, n, t0s) in enumerate(zip(data.frames, data.N_ref, data.t0s)):
        f = f * (n_typ / n)
        if data.H is not None:
            f = f + data.H[k] * (n_typ / n)
        for t0 in t0s:
            sq.append(float((f[t0 + 1:t0 + 7] - f[t0:t0 + 1]).pow(2).mean()))
    return float(np.sqrt(np.mean(sq)))


def train_member(k, tr, te, args, dev, c_scale, d_scale):
    torch.manual_seed(1000 + k)
    gen = torch.Generator().manual_seed(2000 + k)
    model = ForecastGNOT(d=args.d, layers=args.layers, c_scale=c_scale, d_scale=d_scale).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.05)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"member {k}: {n_par / 1e3:.0f}k parameters, c_scale {c_scale:.4g}, d_scale {d_scale:.4g}", flush=True)
    t0, run = time.time(), []
    for it in range(1, args.iters + 1):
        qx, qh, hx, hh, V, N, tg = tr.batch(args.batch, args.queries, gen)
        mean, logvar = model(qx, qh, hx, hh, V, N)
        loss = nll(mean, logvar, tg, d_scale)
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
    torch.save({"state": model.state_dict(), "c_scale": c_scale, "d_scale": d_scale, "d": args.d, "layers": args.layers,
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
    ap.add_argument("--transitions", action="store_true",
                    help="--data-dir holds transition files (make_transitions.py): paper-like runs that start from another case's CO2")
    ap.add_argument("--z", type=float, default=None, help="plane height [m]; default the file's plane (1.6 = paper)")
    ap.add_argument("--c-offset",type=float, default=0.0,
                    help="RANS data: 400 -> also report the paper's metric (absolute ppm); laminar data: 0")
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out = os.path.join(HERE, "checkpoints", args.tag)
    os.makedirs(args.out, exist_ok=True)
    if any(f.endswith(".pth") for f in os.listdir(args.out)):
        raise SystemExit(f"{args.out} already has checkpoints -- use a new --tag")
    dev = torch.device(args.device)
    trp, tep = split(args.data_dir, args.scenarios, args.with_s00)
    tr = PlaneData(trp, dev, seed=0, z=args.z, transitions=args.transitions)
    te = PlaneData(tep, dev, seed=0, z=args.z, transitions=args.transitions)
    assert torch.equal(tr.hist_idx, te.hist_idx)
    c_scale, d_scale = c_scale_of(tr), d_scale_of(tr)
    print(f"data {args.data_dir}: {len(trp)} train / {len(tep)} test cases, {tr.xyz.shape[0]} plane points "
          f"(z = {float(tr.xyz[0, 2]):.2f} m), {sum(map(len, tr.t0s))} train samples per occupancy; device {dev}", flush=True)

    models = [train_member(k, tr, te, args, dev, c_scale, d_scale) for k in range(args.members)]

    print("\nFINAL (every sample, occupancy = N_ref of the data)")
    for name, d in (("train", tr), ("test", te)):
        e1, b, n = evaluate(models[:1], d)
        eE, _, _ = evaluate(models, d)
        print(f"  {name:5s}: member 0 {100 * e1:.2f}%  ensemble({len(models)}) {100 * eE:.2f}%  persistence {100 * b:.2f}%  [{n} samples]"
              f"  -> model error = {eE / b:.2f} x persistence (must be < 1 to be useful)")
    if args.c_offset:
        for name, d in (("train", tr), ("test", te)):
            eE, b, _ = evaluate(models, d, offset=args.c_offset)
            print(f"  {name:5s} PAPER METRIC (absolute = {args.c_offset:g} + excess): ensemble {100 * eE:.2f}%  persistence {100 * b:.2f}%")
    for N in (10.0, 80.0):
        eE, b, _ = evaluate(models, te, N=N)
        print(f"  test at N = {N:g}: ensemble {100 * eE:.2f}%  persistence {100 * b:.2f}%")
    print("  per test case (ensemble / persistence):")
    for k, c in enumerate(te.cases):
        sub = PlaneData.__new__(PlaneData)
        sub.__dict__.update(te.__dict__)
        sub.cases, sub.frames, sub.V, sub.N_ref, sub.t0s = [c], [te.frames[k]], [te.V[k]], [te.N_ref[k]], [te.t0s[k]]
        sub.H = [te.H[k]] if te.H is not None else None
        eE, b, _ = evaluate(models, sub)
        print(f"    {c['name']:10s} {100 * eE:6.2f}%  {100 * b:6.2f}%")
    print("Paper (Bian & Shi 2025): l2 train 5.9 %, test 10.9 %.")


if __name__ == "__main__":
    main()
