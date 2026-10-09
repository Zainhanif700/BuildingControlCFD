"""
Diagnostic: what does the network add to the built-in base flow?
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--code-dir", default=os.path.dirname(HERE))
    ap.add_argument("--times", type=float, nargs="*", default=[60.0, 120.0])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    code_dir = os.path.abspath(os.path.expanduser(args.code_dir))
    sys.path.insert(0, code_dir)
    sys.path.insert(1, HERE)
    import torch
    from gnot_model import GNOTOperator, check_checkpoint_compat
    from point_sampler import ROOM_X, ROOM_Y, V_MAX, BREATHING_HEIGHT
    from throughflow import through_flow_potential
    from check_physics_consistency import evaluate
    import check_co2_with_model_flow as L2
    from compare_with_openfoam import read_internal, time_dirs

    case = os.path.abspath(os.path.expanduser(args.case))
    meta = dict(line.split(None, 1) for line in open(os.path.join(case, "scenario.txt")).read().splitlines())
    V = [float(v) for v in meta["V"].split()]
    dx = float(meta["dx"])
    dev = args.device
    ck = os.path.abspath(os.path.expanduser(args.checkpoint))
    ckpt = torch.load(ck, map_location=dev)
    check_checkpoint_compat(ckpt, ck)
    nu_case, nu_ckpt = float(meta.get("nu", "0.01")), float(ckpt.get("nu", 0.01))
    if abs(nu_case - nu_ckpt) > 1e-12:
        raise SystemExit(f"viscosity mismatch: checkpoint nu = {nu_ckpt:g}, OpenFOAM case nu = {nu_case:g}")
    model = GNOTOperator().to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    from gnot_model import door_jet_speed
    s = float(door_jet_speed(torch.tensor([V])).item())
    print(f"{ck} (version={ckpt.get('version')}); scenario {meta['name']} V={V}; correction scale s(V) = {s:.3f} m/s")

    g = L2.Grid(dx, V)
    C = read_internal(os.path.join(case, "0", "C"), 3)
    idx = [np.clip(np.floor((C[:, a] - (ROOM_X[0], ROOM_Y[0], 0.0)[a]) / g.h[a]).astype(int), 0, g.n[a] - 1)
           for a in range(3)]
    fl = g.fluid
    P = np.stack([g.X[fl], g.Y[fl], g.Z[fl]], 1)
    kz = int(np.argmin(np.abs(g.c1d[2] - BREATHING_HEIGHT)))
    plane = (np.abs(P[:, 2] - g.c1d[2][kz]) < 1e-9)
    tdirs = dict(time_dirs(case))

    def bp_velocity(t):
        out = []
        Vt = torch.tensor([V], dtype=torch.float32, device=dev)
        with torch.no_grad():
            alpha = model.door_split(torch.full((1, 1), t, device=dev), Vt).item()
        for i in range(0, len(P), 20000):
            Q = torch.tensor(P[i:i + 20000], dtype=torch.float32, device=dev)
            x, y, z = (Q[:, j:j + 1].clone().requires_grad_(True) for j in range(3))
            n = x.shape[0]
            chi, psi = through_flow_potential(x, y, z, torch.full((n, 1), t, device=dev), Vt.expand(n, -1),
                                              torch.full((n, 1), alpha, device=dev))
            gp = torch.autograd.grad(psi.sum(), (x, y), retain_graph=True)
            gc = torch.autograd.grad(chi.sum(), (y, z))
            out.append(torch.cat([gp[1], gc[1] - gp[0], -gc[0]], 1).detach().cpu().numpy().astype(float))
        return np.concatenate(out), alpha

    print(f"\n{'t':>5s} | {'|u_Bp|':>7s} {'|u_corr|':>8s} {'|u_OF|':>7s} {'|d|':>7s} | "
          f"{'size |corr|/|d|':>15s} {'alignment':>9s} {'achieved':>8s} | region")
    for t in args.times:
        if t not in tdirs:
            print(f"{t:5.0f} | OpenFOAM field not saved -- skipped")
            continue
        U_of = np.full(g.X.shape + (3,), np.nan)
        U_of[idx[0], idx[1], idx[2]] = read_internal(os.path.join(case, tdirs[t], "U"), 3, len(C))
        u_of = U_of[fl]
        f = evaluate(model, dev, P, t, V, 20.0)
        u_m = np.stack([f["u"], f["v"], f["w"]], 1)
        u_bp, alpha = bp_velocity(t)
        u_corr = u_m - u_bp
        d = u_of - u_bp
        for name, m in (("whole room", np.ones(len(P), bool)), ("breathing plane", plane)):
            rms = lambda a: float(np.sqrt(np.mean(np.sum(a[m] ** 2, axis=1))))
            dot = float(np.mean(np.sum(u_corr[m] * d[m], axis=1)))
            nc, nd = rms(u_corr), rms(d)
            print(f"{t:5.0f} | {rms(u_bp):7.3f} {nc:8.3f} {rms(u_of):7.3f} {nd:7.3f} | "
                  f"{nc / nd:15.3f} {dot / max(nc * nd, 1e-30):9.3f} {dot / max(nd * nd, 1e-30):8.3f} | {name}")
    print("\nReading: alignment near +1 but size << 1  -> right direction, too small (scaling, e.g. s(V));"
          "\n         alignment near 0 or negative    -> the correction does not move towards the NS solution"
          "\n                                            (loss/training), whatever its size.")


if __name__ == "__main__":
    main()
