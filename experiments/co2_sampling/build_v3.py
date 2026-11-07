"""
Builds one data_v3 file and its transition file.
Dataset: our CO2 solver on the real OpenFOAM flow saved every 1 s for 0-150 s (turbulent Schmidt number 0.7,
as in OpenFOAM), then on the 150-600 s mean flow from the data_v2 file (Schmidt number 0.3), CO2 every 30 s.
After --splice s (default 20 min) the frames are taken from data_v2: from there on both use the same mean flow and
differ by <= 0.15 % (measured on 8 settings), so computing them again changes nothing.
Transition: the same flow, no people, starting from the 30-min CO2 of the partner setting (same random pairing as
make_transitions.py; start field from data_v2, identical to v3 at 30 min within 0.1 %); after --splice from data_transitions_v2.
Same keys as before, so the training reads both files unchanged. Only reads the case and the v2 files.
Usage: python3 build_v3.py --name N14 --case <1-s case> --v2-dir <data_v2> --out <data_v3>
                           --trans-v2-dir <data_transitions_v2> --trans-out <data_transitions_v3> [--only-transition]
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


def partner(name, names, seed=0):
    """Partner setting of the transition run, drawn exactly as in make_transitions.py."""
    rng = np.random.default_rng(seed)
    for i, nb in enumerate(names):
        others = [k for k in range(len(names)) if k != i]
        a = names[others[int(rng.integers(len(others)))]]
        if nb == name:
            return a
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--case", required=True, help="OpenFOAM case with saves every 1 s to (at least) 150 s")
    ap.add_argument("--v2-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trans-v2-dir", required=True)
    ap.add_argument("--trans-out", required=True)
    ap.add_argument("--splice", type=float, default=1200.0, help="[s] take the frames after this from the v2 files")
    ap.add_argument("--only-transition", action="store_true", help="the data_v3 file exists: build only the transition")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import torch
    from fv_cons import ConsFV
    from fv_turb import seat_source
    from make_transitions import plane_mask

    out_path = os.path.join(args.out, args.name + ".npz")
    tr_path = os.path.join(args.trans_out, args.name + ".npz")
    if os.path.exists(tr_path):
        raise SystemExit(f"{tr_path} exists -- delete it first to rebuild")
    if os.path.exists(out_path) != args.only_transition:
        raise SystemExit(f"{out_path} {'missing' if args.only_transition else 'exists'} -- "
                         f"{'drop' if args.only_transition else 'use'} --only-transition")
    t0 = time.time()
    meta, V, g, doors, times, field, td = S.read_case(args.case, t_max=T_REAL)
    fl = g.fluid
    assert len(times) >= 150 and abs(times[-1] - T_REAL) < 1e-6, f"need saves every 1 s to {T_REAL:g} s, got {len(times)} to {times[-1]:g} s"
    d2 = np.load(os.path.join(args.v2_dir, args.name + ".npz"))
    P = np.stack([g.X[fl], g.Y[fl], g.Z[fl]], 1).astype(np.float32)
    assert np.allclose(d2["P"], P), "grid of the case and the data_v2 file differ"
    assert np.allclose(d2["V"], V), f"different settings: {d2['V']} vs {V}"
    assert [int(x) for x in d2["doors"]] == doors, "different doors"
    out_t = [float(t) for t in np.arange(DT_OUT, T_CO2 + 1e-9, DT_OUT)]
    t_c = np.array([0.0] + out_t, np.float32)
    assert np.allclose(d2["t_c"], t_c), "data_v2 time grid differs"
    calc_t = [t for t in out_t if t <= args.splice + 1e-9]
    k_spl = len(calc_t) + 1

    fv = ConsFV(g, seat_source(g, N_REF), V, args.device, torch.float32, sc_t=SCT_REAL, doors=doors)
    bank = S.FlowBank(fv)
    grid = lambda x: (lambda o: (o.__setitem__(fl, x), o)[1])(np.zeros(g.X.shape, np.float32))
    bank.add(0.0, 0.0, np.zeros(g.X.shape + (3,), np.float32), np.zeros(g.X.shape, np.float32))
    U_snap, nut_snap = [np.zeros((fl.sum(), 3), np.float32)], [np.zeros(fl.sum(), np.float32)]
    for t in times:
        u, nut = field(t, "U").astype(np.float32), field(t, "nut").astype(np.float32)
        bank.add(t, t, u, nut)
        U_snap.append(u[fl])
        nut_snap.append(nut[fl])
    bank.add("mean", 1e3, np.stack([grid(d2["U_mean"][:, a]) for a in range(3)], -1), grid(d2["nut_mean"]))
    t_read = time.time() - t0
    sct = lambda k: SCT_MEAN if isinstance(k, str) else SCT_REAL
    flow = S.map_b2(times, "mean")

    if args.only_transition:
        Cfull = np.load(out_path)["C"]
        msg = f"{args.name}: data_v3 file exists"
    else:
        t1 = time.time()
        C = S.run(fv, bank, flow, sorted(set(calc_t + [T_REAL])), sct)
        ref = field(T_REAL, "s")
        rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
        chk = {"err_real_vs_OF_150": rel(C[T_REAL][fl], ref[fl]), "mass_real_vs_OF_150": float(C[T_REAL][fl].sum() / ref[fl].sum())}
        for zz in (1.1, 1.6):
            k = int(np.argmin(np.abs(g.c1d[2] - zz)))
            m = fl[:, :, k]
            chk["abs_err_150_z" + f"{zz:g}".replace(".", "p")] = rel(C[T_REAL][:, :, k][m] + PPM_OUTDOOR, ref[:, :, k][m] + PPM_OUTDOOR)
        chk["jump_at_splice_vs_v2"] = rel(C[calc_t[-1]][fl], d2["C"][k_spl - 1])
        Cfull = np.concatenate([np.zeros((1, int(fl.sum())), np.float32),
                                np.stack([C[t][fl].astype(np.float32) for t in calc_t]), d2["C"][k_spl:].astype(np.float32)])
        kz = int(np.argmin(np.abs(g.c1d[2] - BREATHING_Z)))
        os.makedirs(args.out, exist_ok=True)
        with open(out_path + ".tmp", "wb") as fh:
            np.savez_compressed(
                fh, P=P, plane=np.abs(P[:, 2] - g.c1d[2][kz]) < 1e-5, z_plane=np.float32(g.c1d[2][kz]),
                V=np.array(V, np.float32), N_ref=np.float32(N_REF), dx=np.float32(float(meta["dx"])), doors=np.array(doors, np.int8),
                t_c=t_c, C=Cfull, t_snap=np.array([0.0] + times, np.float32), U_snap=np.stack(U_snap), nut_snap=np.stack(nut_snap),
                U_mean=d2["U_mean"], nut_mean=d2["nut_mean"], t_avg=np.float32(T_REAL), t_end_of=d2["t_end_of"],
                fluctuation=d2["fluctuation"], seat_box=np.array(SEAT_BOX, np.float32),
                seat_box_placeholder=common.SEAT_BOX_IS_PLACEHOLDER, co2_solver=np.array(common.DATA_CO2_SOLVER),
                sc_t_real=np.float32(SCT_REAL), sc_t_mean=np.float32(SCT_MEAN), splice_s=np.float32(args.splice),
                method=np.array(f"v3: 1-s OpenFOAM saves 0-150 s (Sc 0.7), then 150-600 s mean flow from data_v2 (Sc 0.3); "
                                f"frames after {args.splice:g} s from data_v2"),
                **{k: np.float32(v) for k, v in chk.items()})
        os.replace(out_path + ".tmp", out_path)
        msg = (f"{args.name}: V = {V}; at 150 s vs OpenFOAM CO2: {100 * chk['err_real_vs_OF_150']:.1f} % (total "
               f"{chk['mass_real_vs_OF_150']:.3f}), paper metric {100 * chk['abs_err_150_z1p1']:.1f}/{100 * chk['abs_err_150_z1p6']:.1f} %; "
               f"jump to v2 at {args.splice:g} s {100 * chk['jump_at_splice_vs_v2']:.2f} %; dataset {time.time() - t1:.0f} s")

    names = sorted(os.path.basename(f)[:-4] for f in os.listdir(args.v2_dir) if f.startswith("N") and f.endswith(".npz"))
    assert len(names) == 28, f"expected 28 data_v2 files, found {len(names)}"
    a_name = partner(args.name, names)
    tv2 = np.load(os.path.join(args.trans_v2_dir, args.name + ".npz"))
    assert str(tv2["from_case"]) == a_name, f"pairing differs from data_transitions_v2: {a_name} vs {tv2['from_case']}"
    c0 = np.load(os.path.join(args.v2_dir, a_name + ".npz"))["C"][-1].astype(np.float32)
    m = plane_mask(P)
    assert np.allclose(tv2["P"], P[m]) and np.allclose(tv2["t_c"], t_c), "data_transitions_v2 grid or times differ"
    t1 = time.time()
    fv0 = ConsFV(g, np.zeros(g.X.shape), V, args.device, torch.float32, sc_t=SCT_REAL, doors=doors)
    H = S.run(fv0, bank, flow, calc_t, sct, c0=grid(c0))
    Hp = np.concatenate([c0[m][None], np.stack([H[t][fl][m] for t in calc_t]), tv2["H"][k_spl:]]).astype(np.float32)
    jump = float(np.linalg.norm(Hp[k_spl - 1] - tv2["H"][k_spl - 1]) / np.linalg.norm(tv2["H"][k_spl - 1]))
    os.makedirs(args.trans_out, exist_ok=True)
    with open(tr_path + ".tmp", "wb") as fh:
        np.savez_compressed(fh, P=P[m], V=np.array(V, np.float32), N_ref=np.float32(N_REF), t_c=t_c, H=Hp, S=Cfull[:, m],
                            from_case=np.array(a_name), co2_solver=np.array(common.DATA_CO2_SOLVER),
                            sc_t_real=np.float32(SCT_REAL), sc_t_mean=np.float32(SCT_MEAN), doors=np.array(doors, np.int8),
                            splice_s=np.float32(args.splice), jump_at_splice_vs_v2=np.float32(jump),
                            method=np.array(f"v3 transition: start = 30-min CO2 of {a_name} (data_v2), flow as data_v3; "
                                            f"frames after {args.splice:g} s from data_transitions_v2"))
    os.replace(tr_path + ".tmp", tr_path)
    print(f"{msg}; transition from {a_name}: jump to v2 at {args.splice:g} s {100 * jump:.2f} %, {time.time() - t1:.0f} s; "
          f"read {t_read:.0f} s, total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
