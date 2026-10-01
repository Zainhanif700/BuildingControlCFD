"""
Steady RANS pilot (simpleFoam) vs the transient RANS pilot (pimpleFoam, 0-600 s), window 1 at 1 m/s:

 1. convergence: last initial residuals of the steady run
 2. flow: steady field vs the TIME AVERAGE of the transient run from --avg-start (both k-omega SST);
    mean speed, door split, mean nut
 3. CO2 (30-min approach of the dataset: steady flow frozen from t = 0, our solver fv_turb) against
      * OpenFOAM's CO2 on the real fluctuating flow (scalarTransport of the transient pilot) and
      * fv_turb on the transient time-averaged flow (the 'run 2' of check_mean_flow_co2.py)
    The difference to OpenFOAM is expected around 25 % (the measured natural scatter of the fluctuating
    flow); the steady field should be CLOSE to the time-averaged one.
Usage (training env, from experiments/gnot_openfoam_rans):
  python3 compare_steady_pilot.py --steady cases/W1_1ms_steady_dx0.1 --transient cases/W1_1ms_rans_dx0.1
"""
import argparse
import os
import re

import numpy as np

import common
from common import N_REF, BREATHING_Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steady", required=True)
    ap.add_argument("--transient", required=True)
    ap.add_argument("--avg-start", type=float, default=150.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import torch
    import check_co2_with_model_flow as L2
    from compare_with_openfoam import read_internal, time_dirs, read_patch_sum
    from compare_rans_pilot import load_case, to_grid
    from fv_turb import TurbFV, seat_source
    st, tr = os.path.abspath(args.steady), os.path.abspath(args.transient)
    ms, Vs, g, Cs, i_s, tds = load_case(st, read_internal, time_dirs, L2)
    mt, Vt, gt, Ct, i_t, tdt = load_case(tr, read_internal, time_dirs, L2)
    assert Vs == Vt and g.n == gt.n, "same windows and grid required"
    fl = g.fluid
    kz = int(np.argmin(np.abs(g.c1d[2] - BREATHING_Z)))
    pl = fl[:, :, kz]
    vec = lambda case, d, CC, ii: np.nan_to_num(to_grid(g, ii, read_internal(os.path.join(case, d, "U"), 3, len(CC)))) * fl[..., None]
    sca = lambda case, d, f, CC, ii: np.nan_to_num(to_grid(g, ii, read_internal(os.path.join(case, d, f), 1, len(CC)).reshape(-1))) * fl

    # 1. convergence
    it_last = max(tds)
    logp = os.path.join(st, "log.simpleFoam")
    txt = open(logp).read() if os.path.isfile(logp) else ""
    conv = "converged (residualControl met)" if "SIMPLE solution converged" in txt else "NOT converged within the iteration limit"
    res = {}
    for f in ("Ux", "Uy", "Uz", "p", "k", "omega"):
        m = re.findall(rf"Solving for {f}, Initial residual = ([0-9.eE+-]+)", txt)
        if m:
            res[f] = float(m[-1])
    print(f"steady case {st}: last saved iteration {it_last:g}; {conv}")
    print("  last initial residuals: " + ", ".join(f"{k} {v:.1e}" for k, v in res.items()))

    # 2. flow
    Us = vec(st, tds[it_last], Cs, i_s)
    nus = sca(st, tds[it_last], "nut", Cs, i_s)
    avg_t = [t for t in sorted(tdt) if t >= args.avg_start]
    Um = np.mean([vec(tr, tdt[t], Ct, i_t) for t in avg_t], axis=0)
    num = np.mean([sca(tr, tdt[t], "nut", Ct, i_t) for t in avg_t], axis=0)
    dv = np.linalg.norm(Us[fl] - Um[fl]) / np.linalg.norm(Um[fl])
    dp = np.linalg.norm(Us[:, :, kz][pl] - Um[:, :, kz][pl]) / np.linalg.norm(Um[:, :, kz][pl])
    phi = os.path.join(st, tds[it_last], "phi")
    d1, d2 = read_patch_sum(phi, "door1"), read_patch_sum(phi, "door2")
    splits = []
    for t in avg_t:
        ph = os.path.join(tr, tdt[t], "phi")
        if os.path.isfile(ph) or os.path.isfile(ph + ".gz"):
            a, b = read_patch_sum(ph, "door1"), read_patch_sum(ph, "door2")
            splits.append(a / (a + b))
    print(f"\nflow: steady vs transient time average ({len(avg_t)} saves from {args.avg_start:g} s)")
    print(f"  velocity difference {100 * dv:.1f} % (volume) / {100 * dp:.1f} % (breathing plane)")
    print(f"  mean speed steady {np.mean(np.linalg.norm(Us[fl], axis=1)):.4f} / average {np.mean(np.linalg.norm(Um[fl], axis=1)):.4f} m/s")
    print(f"  door split steady {d1 / (d1 + d2):.3f} / transient mean {np.mean(splits):.3f} (range {min(splits):.3f}-{max(splits):.3f})")
    print(f"  mean nut steady {nus[fl].mean():.2e} / average {num[fl].mean():.2e} m^2/s")

    # 3. CO2
    src = seat_source(g, N_REF)
    t_chk = [t for t in (120.0, 300.0, 600.0) if t in tdt]
    stdy = [Us[..., 0], Us[..., 1], Us[..., 2], nus]
    mean = [Um[..., 0], Um[..., 1], Um[..., 2], num]
    run_s, _, _ = TurbFV(g, src, args.device, torch.float32).solve([0.0, 1.0], [stdy, stdy], t_chk)
    run_m, _, _ = TurbFV(g, src, args.device, torch.float32).solve([0.0, 1.0], [mean, mean], t_chk)
    print(f"\nCO2 on frozen flow from t = 0 (fv_turb) vs OpenFOAM CO2 on the real fluctuating flow")
    print(f"{'t':>5s} | {'steady flow':>22s} | {'time-avg flow':>22s} | steady vs avg | mean excess OF / steady / avg [ppm]")
    for t in t_chk:
        ref = sca(tr, tdt[t], "s", Ct, i_t)
        e = lambda c: (np.linalg.norm(c[fl] - ref[fl]) / np.linalg.norm(ref[fl]),
                       np.linalg.norm(c[:, :, kz][pl] - ref[:, :, kz][pl]) / np.linalg.norm(ref[:, :, kz][pl]))
        es, em = e(run_s[t]), e(run_m[t])
        dsm = np.linalg.norm(run_s[t][fl] - run_m[t][fl]) / np.linalg.norm(run_m[t][fl])
        print(f"{t:5.0f} | {100 * es[0]:6.1f}% {100 * es[1]:6.1f}% plane | {100 * em[0]:6.1f}% {100 * em[1]:6.1f}% plane | "
              f"{100 * dsm:11.1f}%  | {ref[fl].mean():6.1f} / {run_s[t][fl].mean():6.1f} / {run_m[t][fl].mean():6.1f}")
    print("\nReading: converged + steady close to the time average -> steady RANS represents the mean flow;"
          "\n         CO2 ~25 % from OpenFOAM is the expected natural scatter (same as the time-averaged flow).")


if __name__ == "__main__":
    main()
