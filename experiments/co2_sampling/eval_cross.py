"""
Tests trained forecast ensembles on the test settings of several datasets, with the same evaluate() as the
training, e.g. the model trained on data_v2 and the one trained on data_v3, both on the corrected data_v3.
Only reads checkpoints and data.
Usage: python3 eval_cross.py --models A=<ckpt dir>,B=<ckpt dir> --data v2=<data dir>,v3=<data dir>
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RANS = os.path.normpath(os.path.join(HERE, "..", "gnot_openfoam_rans"))
sys.path.insert(0, RANS)
import common
from forecast import ForecastGNOT, PlaneData, evaluate
from train_forecast import split


def load(ckdir, dev):
    models, hist = [], None
    for p in sorted(glob.glob(os.path.join(ckdir, "member*.pth"))):
        ck = torch.load(p, map_location=dev)
        m = ForecastGNOT(d=ck["d"], layers=ck["layers"], c_scale=ck["c_scale"], d_scale=ck["d_scale"],
                         open_flags=ck.get("open_flags", False)).to(dev)
        m.load_state_dict(ck["state"])
        m.eval()
        models.append(m)
        hist = ck["hist_idx"]
    assert models, f"no member*.pth in {ckdir}"
    return models, hist


def one_case(te, k):
    sub = PlaneData.__new__(PlaneData)
    sub.__dict__.update(te.__dict__)
    sub.cases, sub.frames, sub.V, sub.N_ref, sub.t0s = [te.cases[k]], [te.frames[k]], [te.V[k]], [te.N_ref[k]], [te.t0s[k]]
    sub.H = [te.H[k]] if te.H is not None else None
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="name=checkpoint dir,...")
    ap.add_argument("--data", required=True, help="name=data dir,... (test settings from --scenarios)")
    ap.add_argument("--scenarios", default=os.path.join(RANS, "scenarios_v2.txt"))
    ap.add_argument("--transitions", action="store_true")
    ap.add_argument("--z", type=float, default=1.6)
    ap.add_argument("--c-offset", type=float, default=400.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    pairs = lambda s: [tuple(x.split("=", 1)) for x in s.split(",")]
    models = {n: load(d, dev) for n, d in pairs(args.models)}
    rows = []
    for dn, dd in pairs(args.data):
        _, tep = split(dd, args.scenarios, False)
        te = PlaneData(tep, dev, seed=0, z=args.z, transitions=args.transitions)
        for mn, (ms, hist) in models.items():
            assert torch.equal(hist.to(te.hist_idx.device), te.hist_idx), f"history points of {mn} differ from the data"
            with torch.no_grad():
                e, b, n = evaluate(ms, te)
                ep, bp, _ = evaluate(ms, te, offset=args.c_offset)
                per = [evaluate(ms, one_case(te, k)) for k in range(len(te.cases))]
            rows.append((mn, dn, e, b, ep, bp, n, [(c["name"][:3], x[0], x[1]) for c, x in zip(te.cases, per)]))
            print(f"model {mn} on {dn}: ensemble {100 * e:.2f}%  persistence {100 * b:.2f}%  ({e / b:.2f} x)  "
                  f"paper metric {100 * ep:.3f}% / {100 * bp:.3f}%  [{n} samples]", flush=True)
    print("\nper test setting, ensemble error [%] (persistence in brackets)")
    print("model on data | " + " ".join(f"{c:>13s}" for c, _, _ in rows[0][7]))
    for mn, dn, *_, per in rows:
        print(f"{mn + ' on ' + dn:13s} | " + " ".join(f"{100 * e:5.2f} ({100 * b:4.2f})" for _, e, b in per))


if __name__ == "__main__":
    main()
