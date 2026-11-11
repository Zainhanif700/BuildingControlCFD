"""
Dataset and transition files for the all-openings settings (same method as data_v3, but the mean flow comes from
the setting's own run, because these settings have no data_v2 file).
Dataset (needs the OpenFOAM case of run_full_case.sh): our CO2 solver on the real flow saved every 1 s for 0-150 s
(turbulent Schmidt number 0.7), then on the mean of the saves from 150 to 600 s (Schmidt number 0.3), CO2 every 30 s
to 30 min; checked against OpenFOAM's own CO2 at 150 and 600 s.
Transition (needs only the dataset files): the same flow, no people, starting from the 30-min CO2 of a partner
setting drawn from all settings of the scenario file.
Same keys as before, so the training reads the files unchanged. Only reads the case and the dataset files.
Usage: python3 build_v4.py --name A01 --case <case> --out <data_v4> --scen <scenarios_v4.txt> --trans-out <dir> [--no-transition]
       python3 build_v4.py --name A01 --out <data_v4> --scen <scenarios_v4.txt> --trans-out <dir> --only-transition
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


def scenario_names(path):
    return sorted(l.split()[0] for l in open(path) if l.strip() and not l.startswith("#"))


def partner(name, names, seed=0):
    """Partner of the transition run, drawn in the same way as in make_transitions.py."""
    rng = np.random.default_rng(seed)
    for i, nb in enumerate(names):
        others = [k for k in range(len(names)) if k != i]
        a = names[others[int(rng.integers(len(others)))]]
        if nb == name:
            return a
    raise KeyError(name)


def sct_of(k):
    return SCT_MEAN if isinstance(k, str) else SCT_REAL


def build_dataset(args, out_path, dev):
    import torch
    from fv_cons import ConsFV
    from fv_turb import seat_source
    t0 = time.time()
    meta, V, g, doors, times, field, td = S.read_case(args.case)
    fl = g.fluid
    t_real = [t for t in times if t <= T_REAL + 1e-6]
    T_OF = args.t_of
    t_mean = [t for t in times if t >= T_REAL - 1e-6 and t <= T_OF + 1e-6]
    assert len(t_real) >= 150 and abs(t_real[-1] - T_REAL) < 1e-6, f"need saves every 1 s to 150 s, got {len(t_real)}"
    assert len(t_mean) >= 31 and abs(t_mean[-1] - T_OF) < 1e-6, f"need saves from 150 to {T_OF:g} s, got {len(t_mean)} to {t_mean[-1]:g} s"
    fv = ConsFV(g, seat_source(g, N_REF), V, dev, torch.float32, sc_t=SCT_REAL, doors=doors)
    bank = S.FlowBank(fv)
    grid = lambda x: (lambda o: (o.__setitem__(fl, x), o)[1])(np.zeros(g.X.shape, np.float32))
    bank.add(0.0, 0.0, np.zeros(g.X.shape + (3,), np.float32), np.zeros(g.X.shape, np.float32))
    U_snap, nut_snap = [np.zeros((fl.sum(), 3), np.float32)], [np.zeros(fl.sum(), np.float32)]
    U_m, nut_m = {}, {}
    for t in sorted(set(t_real) | set(t_mean)):
        u, nut = field(t, "U").astype(np.float32), field(t, "nut").astype(np.float32)
        if t <= T_REAL + 1e-6:
            bank.add(t, t, u, nut)
            U_snap.append(u[fl])
            nut_snap.append(nut[fl])
        if t >= T_REAL - 1e-6:
            U_m[t], nut_m[t] = u[fl], nut[fl]
    um = np.mean([U_m[t].astype(np.float64) for t in t_mean], 0).astype(np.float32)
    nm = np.mean([nut_m[t].astype(np.float64) for t in t_mean], 0).astype(np.float32)
    fluct = float(np.mean([np.linalg.norm(U_m[t] - um) / np.linalg.norm(um) for t in t_mean]))
    del U_m, nut_m
    bank.add("mean", 1e3, np.stack([grid(um[:, a]) for a in range(3)], -1), grid(nm))
    t_read = time.time() - t0

    out_t = [float(t) for t in np.arange(DT_OUT, T_CO2 + 1e-9, DT_OUT)]
    C = S.run(fv, bank, S.map_b2(t_real, "mean"), sorted(set(out_t + [T_REAL, T_OF])), sct_of)
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    chk = {}
    for tag, t in (("150", T_REAL), ("end", T_OF)):
        ref = field(t, "s")
        chk[f"err_vs_OF_{tag}"] = rel(C[t][fl], ref[fl])
        chk[f"mass_vs_OF_{tag}"] = float(C[t][fl].sum() / ref[fl].sum())
        for zz in (1.1, 1.6):
            k = int(np.argmin(np.abs(g.c1d[2] - zz)))
            m = fl[:, :, k]
            chk[f"abs_err_{tag}_z" + f"{zz:g}".replace(".", "p")] = rel(C[t][:, :, k][m] + PPM_OUTDOOR, ref[:, :, k][m] + PPM_OUTDOOR)
    P = np.stack([g.X[fl], g.Y[fl], g.Z[fl]], 1).astype(np.float32)
    kz = int(np.argmin(np.abs(g.c1d[2] - BREATHING_Z)))
    os.makedirs(args.out, exist_ok=True)
    with open(out_path + ".tmp", "wb") as fh:
        np.savez_compressed(
            fh, P=P, plane=np.abs(P[:, 2] - g.c1d[2][kz]) < 1e-5, z_plane=np.float32(g.c1d[2][kz]),
            V=np.array(V, np.float32), N_ref=np.float32(N_REF), dx=np.float32(float(meta["dx"])), doors=np.array(doors, np.int8),
            t_c=np.array([0.0] + out_t, np.float32),
            C=np.stack([np.zeros(int(fl.sum()), np.float32)] + [C[t][fl].astype(np.float32) for t in out_t]),
            t_snap=np.array([0.0] + t_real, np.float32), U_snap=np.stack(U_snap), nut_snap=np.stack(nut_snap),
            U_mean=um, nut_mean=nm, t_avg=np.float32(T_REAL), t_end_of=np.float32(T_OF), fluctuation=np.float32(fluct),
            seat_box=np.array(SEAT_BOX, np.float32), seat_box_placeholder=common.SEAT_BOX_IS_PLACEHOLDER,
            co2_solver=np.array(common.DATA_CO2_SOLVER), sc_t_real=np.float32(SCT_REAL), sc_t_mean=np.float32(SCT_MEAN),
            method=np.array(f"v4: 1-s OpenFOAM saves 0-150 s (Sc 0.7), then mean of the 150-{T_OF:g} s saves of the same run (Sc 0.3)"),
            **{k: np.float32(v) for k, v in chk.items()})
    os.replace(out_path + ".tmp", out_path)
    print(f"{args.name}: V = {V}, doors {doors}; vs OpenFOAM CO2 at 150 s {100 * chk['err_vs_OF_150']:.1f} % "
          f"(total {chk['mass_vs_OF_150']:.3f}, paper {100 * chk['abs_err_150_z1p6']:.2f} %), at {T_OF:g} s "
          f"{100 * chk['err_vs_OF_end']:.1f} % (total {chk['mass_vs_OF_end']:.3f}, paper {100 * chk['abs_err_end_z1p6']:.2f} %); "
          f"fluctuation {100 * fluct:.0f} %; mean at 30 min {C[T_CO2][fl].mean():.1f} ppm; read {t_read:.0f} s, "
          f"total {time.time() - t0:.0f} s", flush=True)


def build_transition(args, out_path, tr_path, dev):
    import torch
    import check_co2_with_model_flow as L2
    from fv_cons import ConsFV
    from make_transitions import plane_mask
    t0 = time.time()
    b = np.load(out_path)
    V = [float(v) for v in b["V"]]
    doors = [int(x) for x in b["doors"]]
    g = L2.Grid(round(float(b["dx"]), 6), V)
    fl = g.fluid
    assert np.allclose(b["P"], np.stack([g.X[fl], g.Y[fl], g.Z[fl]], 1)), "grid of the dataset file differs"
    grid = lambda x: (lambda o: (o.__setitem__(fl, x), o)[1])(np.zeros(g.X.shape, np.float32))
    fv0 = ConsFV(g, np.zeros(g.X.shape), V, dev, torch.float32, sc_t=SCT_REAL, doors=doors)
    bank = S.FlowBank(fv0)
    t_snap = [float(t) for t in b["t_snap"]]
    assert t_snap[0] == 0.0 and abs(t_snap[-1] - T_REAL) < 1e-6
    for t, U, nu in zip(t_snap, b["U_snap"], b["nut_snap"]):
        bank.add(t, t, np.stack([grid(U[:, a]) for a in range(3)], -1), grid(nu))
    bank.add("mean", 1e3, np.stack([grid(b["U_mean"][:, a]) for a in range(3)], -1), grid(b["nut_mean"]))
    a_name = partner(args.name, scenario_names(args.scen))
    a_path = os.path.join(args.out, a_name + ".npz")
    assert os.path.exists(a_path), f"partner {a_name} has no dataset file yet ({a_path})"
    c0 = np.load(a_path)["C"][-1].astype(np.float32)
    t_c = b["t_c"].astype(float)
    out_t = [float(t) for t in t_c[1:]]
    H = S.run(fv0, bank, S.map_b2(t_snap[1:], "mean"), out_t, sct_of, c0=grid(c0))
    m = plane_mask(b["P"])
    Hp = np.concatenate([c0[m][None], np.stack([H[t][fl][m] for t in out_t])]).astype(np.float32)
    os.makedirs(args.trans_out, exist_ok=True)
    with open(tr_path + ".tmp", "wb") as fh:
        np.savez_compressed(fh, P=b["P"][m], V=b["V"], N_ref=np.float32(N_REF), t_c=b["t_c"], H=Hp, S=b["C"][:, m],
                            from_case=np.array(a_name), co2_solver=b["co2_solver"], sc_t_real=np.float32(SCT_REAL),
                            sc_t_mean=np.float32(SCT_MEAN), doors=b["doors"],
                            method=np.array(f"v4 transition: start = 30-min CO2 of {a_name}, flow of this setting, full 30 min"))
    os.replace(tr_path + ".tmp", tr_path)
    print(f"{args.name}: transition from {a_name}: mean {Hp[0].mean():.1f} ppm at the start, {Hp[6].mean():.1f} after 3 min, "
          f"{Hp[-1].mean():.2f} after 30 min; {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--case", default=None, help="OpenFOAM case of run_full_case.sh (dataset only)")
    ap.add_argument("--out", required=True, help="dataset folder (all settings, also the partners)")
    ap.add_argument("--scen", required=True, help="scenario file: all settings, for the transition partners")
    ap.add_argument("--trans-out", required=True)
    ap.add_argument("--no-transition", action="store_true")
    ap.add_argument("--only-transition", action="store_true")
    ap.add_argument("--t-of", type=float, default=600.0, help="end of the OpenFOAM run [s] (test: 450)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out_path = os.path.join(args.out, args.name + ".npz")
    tr_path = os.path.join(args.trans_out, args.name + ".npz")
    if not args.only_transition:
        if os.path.exists(out_path):
            raise SystemExit(f"{out_path} exists -- delete it first to rebuild")
        if not args.case:
            raise SystemExit("--case is needed to build the dataset file")
        build_dataset(args, out_path, args.device)
    if not args.no_transition:
        if os.path.exists(tr_path):
            raise SystemExit(f"{tr_path} exists -- delete it first to rebuild")
        build_transition(args, out_path, tr_path, args.device)


if __name__ == "__main__":
    main()
