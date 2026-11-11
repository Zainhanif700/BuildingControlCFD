"""
Which way of building the dataset CO2 comes closest to OpenFOAM's own CO2? All methods use our dataset
solver; only the flow given to it differs. Needs a case with 1-s saves (run_sampling_test.sh + continue_case.sh).
Only reads the case.
Usage: python3 compare_methods.py --case <case> [--t-split 300]
"""
import argparse
import os
import time

import numpy as np

import stream_co2 as S
import common
from common import N_REF, PPM_OUTDOOR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--t-split", type=float, default=300.0, help="loop test: real flow up to here, then loop")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None, help="run only these methods (indices in the list, e.g. 1,2)")
    args = ap.parse_args()
    import torch
    from fv_cons import ConsFV
    from fv_turb import seat_source

    meta, V, g, doors, times, field, td = S.read_case(args.case)
    t_end = times[-1]
    fl = g.fluid
    check = [t for t in (150.0, args.t_split, t_end) if t in td]
    check = sorted(set(check))
    ref = {t: field(t, "s")[fl] for t in check}
    src = seat_source(g, N_REF)
    print(f"{os.path.basename(os.path.abspath(args.case))}: V = {V}, doors {doors}; {len(times)} saves to {t_end:g} s; "
          f"compare at {check} s", flush=True)

    t0 = time.time()
    fv07 = ConsFV(g, src, V, args.device, torch.float32, sc_t=common.SC_T, doors=doors)
    bank = S.FlowBank(fv07)
    z = np.zeros(g.X.shape, np.float32)
    bank.add(0.0, 0.0, np.zeros(g.X.shape + (3,), np.float32), z)
    acc1 = {"u": 0.0, "nut": 0.0, "n": 0}
    acc10 = {"u": 0.0, "nut": 0.0, "n": 0}
    for t in times:
        u, nut = field(t, "U").astype(np.float32), field(t, "nut").astype(np.float32)
        bank.add(t, t, u, nut)
        if t >= 150.0:
            for acc, ok in ((acc1, True), (acc10, abs(t / 10 - round(t / 10)) < 1e-6)):
                if ok:
                    acc["u"] = acc["u"] + u.astype(np.float64)
                    acc["nut"] = acc["nut"] + nut.astype(np.float64)
                    acc["n"] += 1
    for key, acc in (("mean", acc1), ("mean10", acc10)):
        bank.add(key, 1e3, (acc["u"] / acc["n"]).astype(np.float32), (acc["nut"] / acc["n"]).astype(np.float32))
    print(f"  {len(times)} saves read and made divergence-free in {time.time() - t0:.0f} s "
          f"(mean flow from {acc1['n']} saves at 1 s and {acc10['n']} saves at 10 s, 150-{t_end:g} s)", flush=True)

    t10 = [t for t in times if t <= 150.0 and abs(t / 10 - round(t / 10)) < 1e-6]
    t1 = [t for t in times if t <= 150.0]
    split = [t for t in times if t <= args.t_split]

    hybrid = lambda k: 0.3 if isinstance(k, str) else 0.7
    methods = [
        ("dataset now: 10-s saves to 150 s, then mean, Sc 0.3", S.map_b2(t10, "mean10"), 0.3, check),
        ("10-s saves to 150 s Sc 0.7, then mean Sc 0.3", S.map_b2(t10, "mean10"), hybrid, check),
        ("1-s saves to 150 s Sc 0.7, then mean Sc 0.3", S.map_b2(t1, "mean"), hybrid, check),
        ("1-s saves to 150 s, then mean, Sc 0.3", S.map_b2(t1, "mean"), 0.3, check),
        ("1-s saves to 150 s, then mean, Sc 0.7", S.map_b2(t1, "mean"), 0.7, check),
        (f"1-s saves to {args.t_split:g} s, then loop 150-{args.t_split:g} s, Sc 0.7",
         S.map_loop(split, 150.0), 0.7, check),
        (f"1-s saves to {t_end:g} s (real flow, upper limit), Sc 0.7", S.interp_keys(times), 0.7, check),
    ]
    if args.only:
        methods = [methods[int(i)] for i in args.only.split(",")]
    kz = {zz: int(np.argmin(np.abs(g.c1d[2] - zz))) for zz in (1.1, 1.6)}
    rows = []
    for name, flow_at, sct, at in methods:
        fv = fv07
        t1_ = time.time()
        out = S.run(fv, bank, flow_at, at, sct)
        cells = []
        for t in at:
            c, r = out[t][fl], ref[t]
            plane = []
            for zz, k in kz.items():
                m = fl[:, :, k]
                a, b = out[t][:, :, k][m] + PPM_OUTDOOR, field(t, "s")[:, :, k][m] + PPM_OUTDOOR
                plane.append(100 * np.linalg.norm(a - b) / np.linalg.norm(b))
            cells.append(f"{100 * np.linalg.norm(c - r) / np.linalg.norm(r):5.1f} % {c.sum() / r.sum():5.3f} {plane[0]:4.1f}/{plane[1]:3.1f}")
        rows.append((name, cells))
        print(f"  {name}: done in {time.time() - t1_:.0f} s", flush=True)
    print("\nCO2 vs OpenFOAM: error (all cells) / total ours:OpenFOAM / paper metric at 1.1/1.6 m [%]")
    print(f"{'method':62s} | " + " | ".join(f"{t:5g} s" + " " * 18 for t in check))
    for name, cells in rows:
        print(f"{name:62s} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
