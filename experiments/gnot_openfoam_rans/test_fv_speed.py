"""
Checks the faster CO2 solver (CUDA graph on the frozen flow): recomputes the CO2 of an existing dataset file
with the old step-by-step loop and with the graph, and compares both with each other and with the stored data.
Usage: python3 test_fv_speed.py data_v2/N12.npz [minutes, default 10]
"""
import argparse
import time

import numpy as np
import torch

from common import N_REF  # first: adds experiments/pinn to the path
import check_co2_with_model_flow as L2
from extract_rans import DT_OUT
from fv_cons import ConsFV
from fv_turb import seat_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("minutes", type=float, nargs="?", default=10.0)
    args = ap.parse_args()
    d = np.load(args.npz)
    V = [float(v) for v in d["V"]]
    g = L2.Grid(round(float(d["dx"]), 6), V)
    fl = g.fluid
    grid = lambda a: (lambda o: (o.__setitem__(fl, a), o)[1])(np.zeros(g.X.shape))
    snaps = [[grid(U[:, 0]), grid(U[:, 1]), grid(U[:, 2]), grid(nu)] for U, nu in zip(d["U_snap"], d["nut_snap"])]
    mean = [grid(d["U_mean"][:, 0]), grid(d["U_mean"][:, 1]), grid(d["U_mean"][:, 2]), grid(d["nut_mean"])]
    t_snap = [float(t) for t in d["t_snap"]]
    T, S = t_snap + [t_snap[-1] + 10.0], snaps + [mean]
    out_t = [float(t) for t in np.arange(DT_OUT, 60 * args.minutes + 1e-9, DT_OUT)]
    doors = [int(x) for x in d["doors"]] if "doors" in d.files else [1, 1]
    res = {}
    for name, graph in (("step by step", False), ("CUDA graph", True)):
        fv = ConsFV(g, seat_source(g, N_REF), V, "cuda", torch.float32, sc_t=float(d["sc_t"]), doors=doors)
        torch.cuda.synchronize()
        t0 = time.time()
        out, dt, n = fv.solve(T, S, out_t, use_graph=graph)
        torch.cuda.synchronize()
        res[name] = (time.time() - t0, np.stack([out[t][fl] for t in out_t]))
        print(f"{name:13s}: {res[name][0]:7.0f} s for {args.minutes:g} min of CO2 ({n} steps of {dt:.4f} s)", flush=True)
    (ta, a), (tb, b) = res["step by step"], res["CUDA graph"]
    stored = d["C"][1:len(out_t) + 1]
    print(f"speed-up {ta / tb:.1f}x")
    print(f"max difference graph vs step by step: {np.abs(a - b).max():.3g} ppm (CO2 up to {a.max():.1f} ppm)")
    print(f"max difference step by step vs stored dataset: {np.abs(a - stored).max():.3g} ppm")
    print(f"max difference graph vs stored dataset: {np.abs(b - stored).max():.3g} ppm")


if __name__ == "__main__":
    main()
