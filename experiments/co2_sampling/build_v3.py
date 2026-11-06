"""
Builds one data_v3 file: our CO2 solver on the real OpenFOAM flow saved every 1 s for 0-150 s
(turbulent Schmidt number 0.7, as in OpenFOAM), then on the 150-600 s mean flow from the data_v2 file
(Schmidt number 0.3), CO2 every 30 s to 30 min. Same keys as data_v2, so the training reads it unchanged.
Only reads the case and the data_v2 file; writes <out>/<name>.npz.
Usage: python3 build_v3.py --name N14 --case cases/N14ws1_dx0.08 --v2 ../gnot_openfoam_rans/data_v2/N14.npz --out data_v3
"""
import argparse
import os
import time

import numpy as np

import stream_co2 as S
import common
from common import N_REF, PPM_OUTDOOR, BREATHING_Z, SEAT_BOX

T_REAL = 150.0
T_CO2 = 1800.0
DT_OUT = 30.0
SCT_REAL, SCT_MEAN = 0.7, 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--case", required=True, help="OpenFOAM case with saves every 1 s to (at least) 150 s")
    ap.add_argument("--v2", required=True, help="data_v2 file of the same setting (its 150-600 s mean flow)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import torch
    from fv_cons import ConsFV
    from fv_turb import seat_source

    out_path = os.path.join(args.out, args.name + ".npz")
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists -- delete it first to rebuild")
    t0 = time.time()
    meta, V, g, doors, times, field, td = S.read_case(args.case, t_max=T_REAL)
    fl = g.fluid
    assert len(times) >= 150 and abs(times[-1] - T_REAL) < 1e-6, f"need saves every 1 s to {T_REAL:g} s, got {len(times)} to {times[-1]:g} s"
    d2 = np.load(args.v2)
    P = np.stack([g.X[fl], g.Y[fl], g.Z[fl]], 1).astype(np.float32)
    assert np.allclose(d2["P"], P), "grid of the case and the data_v2 file differ"
    assert np.allclose(d2["V"], V), f"different settings: {d2['V']} vs {V}"
    assert [int(x) for x in d2["doors"]] == doors, "different doors"

    src = seat_source(g, N_REF)
    fv = ConsFV(g, src, V, args.device, torch.float32, sc_t=SCT_REAL, doors=doors)
    bank = S.FlowBank(fv)
    grid = lambda x: (lambda o: (o.__setitem__(fl, x), o)[1])(np.zeros(g.X.shape, np.float32))
    bank.add(0.0, 0.0, np.zeros(g.X.shape + (3,), np.float32), np.zeros(g.X.shape, np.float32))
    U_snap, nut_snap = [np.zeros((fl.sum(), 3), np.float32)], [np.zeros(fl.sum(), np.float32)]
    for t in times:
        u, nut = field(t, "U").astype(np.float32), field(t, "nut").astype(np.float32)
        bank.add(t, t, u, nut)
        U_snap.append(u[fl])
        nut_snap.append(nut[fl])
    um = np.stack([grid(d2["U_mean"][:, a]) for a in range(3)], -1)
    bank.add("mean", 1e3, um, grid(d2["nut_mean"]))
    t_read = time.time() - t0

    out_t = [float(t) for t in np.arange(DT_OUT, T_CO2 + 1e-9, DT_OUT)]
    sct = lambda k: SCT_MEAN if isinstance(k, str) else SCT_REAL
    C = S.run(fv, bank, S.map_b2(times, "mean"), sorted(set(out_t + [T_REAL])), sct)

    ref = field(T_REAL, "s")
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    chk = {"err_real_vs_OF_150": rel(C[T_REAL][fl], ref[fl]), "mass_real_vs_OF_150": float(C[T_REAL][fl].sum() / ref[fl].sum())}
    for zz in (1.1, 1.6):
        k = int(np.argmin(np.abs(g.c1d[2] - zz)))
        m = fl[:, :, k]
        chk["abs_err_150_z" + f"{zz:g}".replace(".", "p")] = rel(C[T_REAL][:, :, k][m] + PPM_OUTDOOR, ref[:, :, k][m] + PPM_OUTDOOR)

    kz = int(np.argmin(np.abs(g.c1d[2] - BREATHING_Z)))
    os.makedirs(args.out, exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh, P=P, plane=np.abs(P[:, 2] - g.c1d[2][kz]) < 1e-5, z_plane=np.float32(g.c1d[2][kz]),
            V=np.array(V, np.float32), N_ref=np.float32(N_REF), dx=np.float32(float(meta["dx"])), doors=np.array(doors, np.int8),
            t_c=np.array([0.0] + out_t, np.float32),
            C=np.stack([np.zeros(int(fl.sum()), np.float32)] + [C[t][fl].astype(np.float32) for t in out_t]),
            t_snap=np.array([0.0] + times, np.float32), U_snap=np.stack(U_snap), nut_snap=np.stack(nut_snap),
            U_mean=d2["U_mean"], nut_mean=d2["nut_mean"], t_avg=np.float32(T_REAL), t_end_of=d2["t_end_of"],
            fluctuation=d2["fluctuation"], seat_box=np.array(SEAT_BOX, np.float32),
            seat_box_placeholder=common.SEAT_BOX_IS_PLACEHOLDER, co2_solver=np.array(common.DATA_CO2_SOLVER),
            sc_t_real=np.float32(SCT_REAL), sc_t_mean=np.float32(SCT_MEAN),
            method=np.array("v3: 1-s OpenFOAM saves 0-150 s (Sc 0.7), then 150-600 s mean flow from data_v2 (Sc 0.3)"),
            **{k: np.float32(v) for k, v in chk.items()})
    os.replace(tmp, out_path)
    print(f"{args.name}: V = {V}; at 150 s vs OpenFOAM CO2: {100 * chk['err_real_vs_OF_150']:.1f} % (total "
          f"{chk['mass_real_vs_OF_150']:.3f}), paper metric {100 * chk['abs_err_150_z1p1']:.1f}/{100 * chk['abs_err_150_z1p6']:.1f} %; "
          f"mean at 30 min {C[T_CO2][fl].mean():.1f} ppm (data_v2: {float(d2['C'][-1].mean()):.1f}); "
          f"read {t_read:.0f} s, total {time.time() - t0:.0f} s; saved {out_path} ({os.path.getsize(out_path) / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
