"""
Extract one transient k-omega SST case (0-600 s, saves every 10 s) into a dataset file (option B2):

  flow   real snapshots 0-T_AVG (start-up), then the TIME AVERAGE of U and nut over [T_AVG, end],
         frozen for the rest of the 30 min (the flow is statistically steady but fluctuates; the
         average is the reproducible part -- pilot: a single realisation differs by ~25 %)
  CO2    fv_turb.py (D_CO2 + nut / Sc_t, seated source, ppm excess over 400 ppm outdoor), every 30 s
         to 1800 s (as in Bian & Shi 2025), N_REF people (C is exactly linear in N)
  check  CO2 at the last OpenFOAM time vs OpenFOAM's own CO2 (scalarTransport on the real flow), for
         (a) fv_turb on all real snapshots and (b) the B2 flow -> per-case quality numbers
  kept   the snapshots 0-T_AVG and the average flow, so the CO2 can be RECOMPUTED later (e.g. with the
         real seating area) without a new OpenFOAM run: --recompute <npz>

Usage (training env, from experiments/gnot_openfoam_rans):
  python3 extract_rans.py --case cases/S04_dx0.1 --name S04
"""
import argparse
import os
import time

import numpy as np

import common
from common import N_REF, BREATHING_Z, DATA_DIR, SEAT_BOX

T_AVG = 150.0
T_CO2 = 1800.0
DT_OUT = 30.0


def co2_b2(g, src, t_snap, snaps, mean, device, solver="old", sct=common.SC_T, V=None):
    """solver 'old' = fv_turb (advective form, as extracted); 'cons' = fv_cons (mass-conserving, needs V)"""
    import torch
    T = list(t_snap) + [t_snap[-1] + 10.0]
    S = list(snaps) + [mean]
    out_t = [float(t) for t in np.arange(DT_OUT, T_CO2 + 1e-9, DT_OUT)]
    if solver == "cons":
        from fv_cons import ConsFV
        fv = ConsFV(g, src, V, device, torch.float32, sc_t=sct)
    else:
        from fv_turb import TurbFV
        fv = TurbFV(g, src, device, torch.float32, sc_t=sct)
    out, dt, n = fv.solve(T, S, out_t, log_every=50000)
    return out_t, out, dt, n


def extract(case, name, out_dir=DATA_DIR, device="cuda"):
    import torch
    import check_co2_with_model_flow as L2
    from compare_with_openfoam import read_internal, time_dirs
    from compare_rans_pilot import load_case, to_grid
    from fv_turb import TurbFV, seat_source
    case = os.path.abspath(case)
    meta, V, g, Cc, idx, td = load_case(case, read_internal, time_dirs, L2)
    fl = g.fluid
    times = sorted(t for t in td if t > 0)
    t_end = times[-1]
    assert t_end > T_AVG + 100, f"case saved only to {t_end:g} s -- needs the run to ~600 s"
    z = np.zeros(g.X.shape)
    snaps = {0.0: [z, z, z, z]}
    for t in times:
        u = np.nan_to_num(to_grid(g, idx, read_internal(os.path.join(case, td[t], "U"), 3, len(Cc)))) * fl[..., None]
        nut = np.nan_to_num(to_grid(g, idx, read_internal(os.path.join(case, td[t], "nut"), 1, len(Cc)).reshape(-1))) * fl
        snaps[t] = [u[..., 0], u[..., 1], u[..., 2], nut]
    avg_t = [t for t in times if t >= T_AVG]
    mean = [np.mean([snaps[t][k] for t in avg_t], axis=0) for k in range(4)]
    um = np.stack(mean[:3], -1)[fl]
    fluct = float(np.mean([np.linalg.norm(np.stack(snaps[t][:3], -1)[fl] - um) / np.linalg.norm(um) for t in avg_t]))
    speeds = [float(np.mean(np.linalg.norm(np.stack(snaps[t][:3], -1)[fl], axis=1))) for t in avg_t]
    t_snap = [0.0] + [t for t in times if t <= T_AVG]
    src = seat_source(g, N_REF)

    t0 = time.time()
    out_t, C, dt, n = co2_b2(g, src, t_snap, [snaps[t] for t in t_snap], mean, device)
    # per-case check at the last OpenFOAM time against OpenFOAM's own CO2 on the real flow
    real, _, _ = TurbFV(g, src, device, torch.float32).solve([0.0] + times, [snaps[t] for t in [0.0] + times], [t_end])
    ref = np.nan_to_num(to_grid(g, idx, read_internal(os.path.join(case, td[t_end], "s"), 1, len(Cc)).reshape(-1))) * fl
    b2 = C[t_end] if t_end in C else None
    if b2 is None:   # t_end not on the 30-s grid: integrate the B2 flow to t_end once more
        b2 = TurbFV(g, src, device, torch.float32).solve(t_snap + [T_AVG + 10.0], [snaps[t] for t in t_snap] + [mean], [t_end])[0][t_end]
    rel = lambda c: float(np.linalg.norm(c[fl] - ref[fl]) / np.linalg.norm(ref[fl]))
    chk = {"err_real_vs_OF": rel(real[t_end]), "err_B2_vs_OF": rel(b2),
           "mass_real_vs_OF": float(real[t_end][fl].sum() / ref[fl].sum()), "mass_B2_vs_OF": float(b2[fl].sum() / ref[fl].sum())}
    # the paper's metric: plane l2 of the ABSOLUTE ppm (400 + excess), at 1.1 m and at 1.6 m (paper plane)
    for zz in (1.1, 1.6):
        k_ = int(np.argmin(np.abs(g.c1d[2] - zz)))
        m_ = fl[:, :, k_]
        r_ = ref[:, :, k_][m_] + common.PPM_OUTDOOR
        chk["abs_err_B2_z" + f"{zz:g}".replace(".", "p")] = float(np.linalg.norm(b2[:, :, k_][m_] + common.PPM_OUTDOOR - r_) / np.linalg.norm(r_))
    print(f"{name}: V = {V}; flow fluctuation around the mean {100 * fluct:.1f} %, mean speed {np.mean(speeds):.3f} m/s "
          f"(+-{100 * np.std(speeds) / np.mean(speeds):.1f} %); CO2: {n} steps, dt {dt:.4f} s, {time.time() - t0:.0f} s wall")
    print(f"  check at {t_end:g} s vs OpenFOAM CO2: real flow {100 * chk['err_real_vs_OF']:.1f} % (mass {chk['mass_real_vs_OF']:.3f}), "
          f"B2 flow {100 * chk['err_B2_vs_OF']:.1f} % (mass {chk['mass_B2_vs_OF']:.3f}); paper metric (absolute ppm, plane) "
          f"B2 {100 * chk['abs_err_B2_z1p1']:.1f} % at 1.1 m, {100 * chk['abs_err_B2_z1p6']:.1f} % at 1.6 m")

    P = np.stack([g.X[fl], g.Y[fl], g.Z[fl]], 1).astype(np.float32)
    kz = int(np.argmin(np.abs(g.c1d[2] - BREATHING_Z)))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, name + ".npz")
    np.savez_compressed(
        out, P=P, plane=np.abs(P[:, 2] - g.c1d[2][kz]) < 1e-5, z_plane=np.float32(g.c1d[2][kz]),
        V=np.array(V, np.float32), N_ref=np.float32(N_REF), dx=np.float32(float(meta["dx"])),
        t_c=np.array([0.0] + out_t, np.float32),
        C=np.stack([np.zeros(int(fl.sum()), np.float32)] + [C[t][fl].astype(np.float32) for t in out_t]),
        t_snap=np.array(t_snap, np.float32),
        U_snap=np.stack([np.stack(snaps[t][:3], -1)[fl].astype(np.float32) for t in t_snap]),
        nut_snap=np.stack([snaps[t][3][fl].astype(np.float32) for t in t_snap]),
        U_mean=um.astype(np.float32), nut_mean=mean[3][fl].astype(np.float32),
        t_avg=np.float32(T_AVG), t_end_of=np.float32(t_end), fluctuation=np.float32(fluct),
        seat_box=np.array(SEAT_BOX, np.float32), seat_box_placeholder=common.SEAT_BOX_IS_PLACEHOLDER,
        **{k: np.float32(v) for k, v in chk.items()})
    print(f"  saved {out} ({os.path.getsize(out) / 1e6:.0f} MB): C every {DT_OUT:g} s to {T_CO2:g} s (ppm excess over "
          f"{common.PPM_OUTDOOR:g} ppm, {N_REF:g} people)" + ("  [SEAT_BOX placeholder]" if common.SEAT_BOX_IS_PLACEHOLDER else ""))
    return out


def recompute(npz, device="cuda", solver="old", sct=common.SC_T):
    """New CO2 from the stored flow (other SEAT_BOX, solver or Sc_t); overwrites C and records the settings.
    Atomic: written to a temporary file first, then renamed (a crash never leaves a broken file)."""
    import check_co2_with_model_flow as L2
    from fv_turb import seat_source
    d = dict(np.load(npz))
    V = [float(v) for v in d["V"]]
    g = L2.Grid(float(d["dx"]), V)
    fl = g.fluid
    grid = lambda a: (lambda o: (o.__setitem__(fl, a), o)[1])(np.zeros(g.X.shape))
    snaps = [[grid(U[:, 0]), grid(U[:, 1]), grid(U[:, 2]), grid(nu)] for U, nu in zip(d["U_snap"], d["nut_snap"])]
    mean = [grid(d["U_mean"][:, 0]), grid(d["U_mean"][:, 1]), grid(d["U_mean"][:, 2]), grid(d["nut_mean"])]
    t0 = time.time()
    out_t, C, _, _ = co2_b2(g, seat_source(g, N_REF), [float(t) for t in d["t_snap"]], snaps, mean, device,
                            solver=solver, sct=sct, V=V)
    d["C"] = np.stack([np.zeros(int(fl.sum()), np.float32)] + [C[t][fl].astype(np.float32) for t in out_t])
    d["seat_box"] = np.array(SEAT_BOX, np.float32)
    d["seat_box_placeholder"] = common.SEAT_BOX_IS_PLACEHOLDER
    d["co2_solver"] = np.array(solver)
    d["sc_t"] = np.float32(sct)
    tmp = npz + ".tmp"                      # not *.npz, so no checker ever picks it up
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **d)
    os.replace(tmp, npz)
    print(f"recomputed CO2 in {npz}: solver {solver}, Sc_t {sct:g}, SEAT_BOX {SEAT_BOX} ({time.time() - t0:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--recompute", default=None, help="npz: recompute the CO2 from the stored flow")
    ap.add_argument("--solver", default="old", choices=["old", "cons"], help="(recompute) CO2 solver")
    ap.add_argument("--sct", type=float, default=common.SC_T, help="(recompute) turbulent Schmidt number")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.recompute:
        recompute(args.recompute, args.device, args.solver, args.sct)
    else:
        if not (args.case and args.name):
            raise SystemExit("give --case and --name (or --recompute <npz>)")
        extract(args.case, args.name, device=args.device)


if __name__ == "__main__":
    main()
