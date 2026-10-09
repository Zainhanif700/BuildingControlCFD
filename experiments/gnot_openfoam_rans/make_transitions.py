"""
Paper-like runs that start from the CO2 of another window setting (no new OpenFOAM runs needed).
Usage: python3 make_transitions.py --shard 0/2   (and --shard 1/2 in parallel)
"""
import argparse
import glob
import os
import time

import numpy as np

import common
from common import N_REF
from extract_rans import DT_OUT, T_CO2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data_transitions")


def plane_mask(P, heights=(1.1, 1.6)):
    zs = np.unique(P[:, 2])
    m = np.zeros(len(P), bool)
    for z in heights:
        m |= np.abs(P[:, 2] - zs[np.argmin(np.abs(zs - z))]) < 1e-5
    return m


def make_one(b_path, a_path, device="cuda"):
    import torch
    import check_co2_with_model_flow as L2
    from fv_cons import ConsFV
    b, a = np.load(b_path), np.load(a_path)
    assert np.allclose(a["P"], b["P"]), "grid mismatch"
    assert "co2_solver" in b.files and str(b["co2_solver"]) == common.DATA_CO2_SOLVER, f"{b_path}: old solver"
    V = [float(v) for v in b["V"]]
    g = L2.Grid(float(b["dx"]), V)
    fl = g.fluid
    grid = lambda x: (lambda o: (o.__setitem__(fl, x), o)[1])(np.zeros(g.X.shape))
    snaps = [[grid(U[:, 0]), grid(U[:, 1]), grid(U[:, 2]), grid(nu)] for U, nu in zip(b["U_snap"], b["nut_snap"])]
    mean = [grid(b["U_mean"][:, 0]), grid(b["U_mean"][:, 1]), grid(b["U_mean"][:, 2]), grid(b["nut_mean"])]
    t_snap = [float(t) for t in b["t_snap"]]
    T = t_snap + [t_snap[-1] + 10.0]
    S = snaps + [mean]
    out_t = [float(t) for t in np.arange(DT_OUT, T_CO2 + 1e-9, DT_OUT)]
    c0 = a["C"][-1].astype(np.float64)
    fv = ConsFV(g, np.zeros(g.X.shape), V, device, torch.float32, sc_t=float(b["sc_t"]))
    H, _, n = fv.solve(T, S, out_t, c0=grid(c0))
    m = plane_mask(b["P"])
    Hp = np.stack([c0[m]] + [H[t][fl][m] for t in out_t]).astype(np.float32)
    assert np.allclose(b["t_c"], np.array([0.0] + out_t, np.float32)), "time grid differs from the dataset"
    return dict(P=b["P"][m], V=b["V"], N_ref=np.float32(N_REF), t_c=b["t_c"], H=Hp, S=b["C"][:, m],
                from_case=np.array(os.path.basename(a_path)[:-4]), sc_t=b["sc_t"], co2_solver=b["co2_solver"]), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default="0/1", help="i/n: this process does cases i, i+n, ... (run n processes in parallel)")
    ap.add_argument("--expect", type=int, default=39, help="number of dataset files (the random pairing depends on it)")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(HERE, "data", "S*.npz")))
    if len(files) != args.expect:
        raise SystemExit(f"found {len(files)} dataset files, expected {args.expect} -- wait until the dataset is complete")
    names = [os.path.basename(f)[:-4] for f in files]
    rng = np.random.default_rng(args.seed)
    os.makedirs(OUT, exist_ok=True)
    print(f"{len(files)} cases -> {OUT}", flush=True)
    for i, (f, nb) in enumerate(zip(files, names)):
        others = [k for k in range(len(files)) if k != i]
        a = files[others[int(rng.integers(len(others)))]]
        out = os.path.join(OUT, f"{nb}.npz")
        si, sn = (int(x) for x in args.shard.split("/"))
        if i % sn != si:
            continue
        if os.path.exists(out):
            print(f"{nb}: exists -- skipped", flush=True)
            continue
        t0 = time.time()
        d, n = make_one(f, a, args.device)
        with open(out + ".tmp", "wb") as fh:
            np.savez_compressed(fh, **d)
        os.replace(out + ".tmp", out)
        h = d["H"]
        print(f"{nb}: start from {d['from_case']} (mean {h[0].mean():.1f} ppm) -> after 3 min {h[6].mean():.1f}, "
              f"10 min {h[20].mean():.1f}, 30 min {h[-1].mean():.1f} ppm (source off); {n} steps, {time.time() - t0:.0f} s",
              flush=True)


if __name__ == "__main__":
    main()
