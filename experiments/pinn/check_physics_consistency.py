"""
Check 3: physics checks of a trained model without a reference (mass balance, boundary conditions, CO2 growth).
Usage: python3 check_physics_consistency.py [checkpoint]
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch

from gnot_model import GNOTOperator, check_checkpoint_compat
from point_sampler import (ROOM_X, ROOM_Y, ROOM_Z, WINDOWS, DOORS, COLUMNS, NUM_WINDOWS,
                           BREATHING_HEIGHT, TAU_RAMP, T_MAX, V_MAX)
from train_gnot import NU, RHO, DIFFUSIVITY, EMISSION_PER_PERSON, SIGMA, SOURCE_X, SOURCE_Y

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "milestones", "v13_fullocc", "gnot_v13_fullocc_final.pth")
N_PEOPLE = 20.0
TIMES = (60.0, 120.0)


def scenarios(n_random, seed=0):
    z = [0.0] * NUM_WINDOWS

    def only(**kw):
        V = list(z)
        for k, v in kw.items():
            V[int(k[1:]) - 1] = float(v)
        return V
    out = [
        ("closed", list(z)),
        ("W1 1m/s", only(W1=1)),
        ("W1 3m/s", only(W1=3)),
        ("W1 5m/s", only(W1=5)),
        ("W4 3m/s", only(W4=3)),
        ("W8 3m/s", only(W8=3)),
        ("W1+W8 2m/s", only(W1=2, W8=2)),
        ("W4+W5 2m/s", only(W4=2, W5=2)),
        ("all 1m/s", [1.0] * NUM_WINDOWS),
        ("all 3m/s", [3.0] * NUM_WINDOWS),
        ("all 5m/s", [5.0] * NUM_WINDOWS),
        ("W1 0.2m/s", only(W1=0.2)),
        ("all 0.2m/s", [0.2] * NUM_WINDOWS),
    ]
    rng = np.random.default_rng(seed)
    for i in range(n_random):
        V = rng.uniform(0.0, V_MAX, NUM_WINDOWS)
        if i % 2 == 1:
            V = V * (rng.random(NUM_WINDOWS) >= 0.5)
            if V.max() == 0:
                V[rng.integers(NUM_WINDOWS)] = rng.uniform(0.5, V_MAX)
        out.append((f"random{i + 1}{'p' if i % 2 else 'u'}", np.round(V, 2).tolist()))
    return out


def in_any_column(x, y):
    inside = np.zeros(np.shape(x), dtype=bool)
    for cx, cy, r, _, _ in COLUMNS:
        inside |= (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    return inside


def _grid1d(lo, hi, h):
    n = max(1, int(math.ceil((hi - lo) / h)))
    d = (hi - lo) / n
    return lo + (np.arange(n) + 0.5) * d, d


def plane_patch(axis, value, r1, r2, h1, h2, normal):
    """Midpoint-rule patch on the plane coord[axis] = value"""
    a, da = _grid1d(*r1, h1)
    b, db = _grid1d(*r2, h2)
    A, B = (m.ravel() for m in np.meshgrid(a, b, indexing="ij"))
    other = [k for k in (0, 1, 2) if k != axis]
    P = np.zeros((A.size, 3))
    P[:, axis], P[:, other[0]], P[:, other[1]] = value, A, B
    return P, np.full(A.size, da * db), np.tile(np.asarray(normal, float), (A.size, 1))


def build_surfaces():
    """Surface patches with outward normals (walls, windows, doors, columns)."""
    S = []
    for k, (xlo, xhi, zlo, zhi) in enumerate(WINDOWS):
        S.append(("window", k) + plane_patch(1, ROOM_Y[1], (xlo, xhi), (zlo, zhi), 0.05, 0.1, (0, 1, 0)))
    for k, (xlo, xhi, zlo, zhi) in enumerate(DOORS):
        S.append(("door", k) + plane_patch(1, ROOM_Y[0], (xlo, xhi), (zlo, zhi), 0.05, 0.1, (0, -1, 0)))

    def wall(P, dA, n, keep):
        S.append(("wall", -1, P[keep], dA[keep], n[keep]))

    def wall_with_openings(yval, normal, openings):
        """Wall patches that tile exactly around the window and door openings."""
        ops = sorted(openings)
        edges = [ROOM_X[0]] + [e for o in ops for e in (o[0], o[1])] + [ROOM_X[1]]
        for a, b in zip(edges[0::2], edges[1::2]):
            if b - a > 1e-9:
                P, dA, n = plane_patch(1, yval, (a, b), ROOM_Z, 0.1, 0.1, normal)
                wall(P, dA, n, np.ones(len(P), bool))
        for xlo, xhi, zlo, zhi in ops:
            for z0, z1 in ((ROOM_Z[0], zlo), (zhi, ROOM_Z[1])):
                if z1 - z0 > 1e-9:
                    P, dA, n = plane_patch(1, yval, (xlo, xhi), (z0, z1), 0.05, 0.1, normal)
                    wall(P, dA, n, np.ones(len(P), bool))
    wall_with_openings(ROOM_Y[1], (0, 1, 0), WINDOWS)
    wall_with_openings(ROOM_Y[0], (0, -1, 0), DOORS)
    for val, sgn in ((ROOM_X[0], -1), (ROOM_X[1], 1)):
        P, dA, n = plane_patch(0, val, ROOM_Y, ROOM_Z, 0.1, 0.1, (sgn, 0, 0))
        wall(P, dA, n, np.ones(len(P), bool))
    for val, sgn in ((ROOM_Z[0], -1), (ROOM_Z[1], 1)):
        P, dA, n = plane_patch(2, val, ROOM_X, ROOM_Y, 0.2, 0.2, (0, 0, sgn))
        wall(P, dA, n, ~in_any_column(P[:, 0], P[:, 1]))
    for k, (cx, cy, r, zlo, zhi) in enumerate(COLUMNS):
        th, dth = _grid1d(0.0, 2 * np.pi, 2 * np.pi / 48)
        zz, dz = _grid1d(zlo, zhi, 0.1)
        TH, ZZ = (m.ravel() for m in np.meshgrid(th, zz, indexing="ij"))
        P = np.stack([cx + r * np.cos(TH), cy + r * np.sin(TH), ZZ], 1)
        n = np.stack([-np.cos(TH), -np.sin(TH), np.zeros_like(TH)], 1)
        S.append(("column", k, P, np.full(len(P), r * dth * dz), n))
    return S


def build_volume(h=0.25):
    xs, dx = _grid1d(*ROOM_X, h)
    ys, dy = _grid1d(*ROOM_Y, h)
    zs, dz = _grid1d(*ROOM_Z, h)
    X, Y, Z = (m.ravel() for m in np.meshgrid(xs, ys, zs, indexing="ij"))
    keep = ~in_any_column(X, Y)
    return np.stack([X[keep], Y[keep], Z[keep]], 1), dx * dy * dz


def source(P, n_people):
    d2 = (P[:, 0] - SOURCE_X) ** 2 + (P[:, 1] - SOURCE_Y) ** 2 + (P[:, 2] - BREATHING_HEIGHT) ** 2
    return n_people * EMISSION_PER_PERSON * np.exp(-d2 / SIGMA ** 2)


def evaluate(model, dev, P, t, V, n_people, grad_c=False, dcdt=False, batch=2048):
    """Velocity, CO2, pressure and the CO2 derivatives at the points P."""
    keys = ["u", "v", "w", "c", "p"] + (["cx", "cy", "cz"] if grad_c else []) + (["ct"] if dcdt else [])
    out = {k: [] for k in keys}
    for i in range(0, len(P), batch):
        Q = P[i:i + batch]
        n = len(Q)
        x, y, z = (torch.tensor(Q[:, j:j + 1], dtype=torch.float32, device=dev).requires_grad_(True) for j in range(3))
        tt = torch.full((n, 1), float(t), device=dev).requires_grad_(dcdt)
        VV = torch.tensor(V, dtype=torch.float32, device=dev).view(1, NUM_WINDOWS).expand(n, -1)
        NN = torch.full((n, 1), float(n_people), device=dev)
        A1, A2, A3, C, p = model(x, y, z, tt, VV, NN)
        u, v, w = model.velocity_from_potential(A1, A2, A3, x, y, z)
        vals = {"u": u, "v": v, "w": w, "c": C, "p": p}
        wrt = ([x, y, z] if grad_c else []) + ([tt] if dcdt else [])
        if wrt:
            g = torch.autograd.grad(C, wrt, grad_outputs=torch.ones_like(C), allow_unused=True)
            names = (["cx", "cy", "cz"] if grad_c else []) + (["ct"] if dcdt else [])
            for k, gk in zip(names, g):
                vals[k] = gk if gk is not None else torch.zeros_like(C)
        for k in keys:
            out[k].append(vals[k].detach().cpu().numpy().ravel().astype(float))
    return {k: np.concatenate(a) for k, a in out.items()}


def residuals(model, dev, V, n_people, n_pts, rng, batch=100):
    """Relative Navier-Stokes and CO2 residuals at random interior points."""
    pts = []
    while sum(len(a) for a in pts) < n_pts:
        Q = np.stack([rng.uniform(*ROOM_X, n_pts), rng.uniform(*ROOM_Y, n_pts), rng.uniform(*ROOM_Z, n_pts)], 1)
        pts.append(Q[~in_any_column(Q[:, 0], Q[:, 1])])
    P = np.concatenate(pts)[:n_pts]
    T = rng.uniform(0.0, T_MAX, n_pts)
    acc = {k: 0.0 for k in ("rn", "dn", "rc", "dc")}
    for i in range(0, n_pts, batch):
        n = len(P[i:i + batch])
        x, y, z = (torch.tensor(P[i:i + batch, j:j + 1], dtype=torch.float32, device=dev).requires_grad_(True)
                   for j in range(3))
        t = torch.tensor(T[i:i + batch, None], dtype=torch.float32, device=dev).requires_grad_(True)
        VV = torch.tensor(V, dtype=torch.float32, device=dev).view(1, NUM_WINDOWS).expand(n, -1)
        NN = torch.full((n, 1), float(n_people), device=dev)
        A1, A2, A3, c, p = model(x, y, z, t, VV, NN)
        u, v, w = model.velocity_from_potential(A1, A2, A3, x, y, z)
        def gz(f, s):
            if not f.requires_grad:
                return torch.zeros_like(f)
            r = torch.autograd.grad(f, s, grad_outputs=torch.ones_like(f), create_graph=True, allow_unused=True)[0]
            return torch.zeros_like(f) if r is None else r
        rn2 = dn2 = 0.0
        for q, dp in ((u, gz(p, x)), (v, gz(p, y)), (w, gz(p, z))):
            qx, qy, qz, qt = gz(q, x), gz(q, y), gz(q, z), gz(q, t)
            lap = gz(qx, x) + gz(qy, y) + gz(qz, z)
            conv = u * qx + v * qy + w * qz
            r = qt + conv + dp / RHO - NU * lap
            rn2 = rn2 + r ** 2
            dn2 = dn2 + qt ** 2 + conv ** 2 + (dp / RHO) ** 2 + (NU * lap) ** 2
        cx, cy, cz, ct = gz(c, x), gz(c, y), gz(c, z), gz(c, t)
        lapc = gz(cx, x) + gz(cy, y) + gz(cz, z)
        convc = u * cx + v * cy + w * cz
        d2 = (x - SOURCE_X) ** 2 + (y - SOURCE_Y) ** 2 + (z - BREATHING_HEIGHT) ** 2
        S = NN * EMISSION_PER_PERSON * torch.exp(-d2 / SIGMA ** 2)
        rc = ct + convc - DIFFUSIVITY * lapc - S
        acc["rn"] += rn2.sum().item()
        acc["dn"] += dn2.sum().item()
        acc["rc"] += (rc ** 2).sum().item()
        acc["dc"] += (ct ** 2 + convc ** 2 + (DIFFUSIVITY * lapc) ** 2 + S ** 2).sum().item()
    rel_ns = math.sqrt(acc["rn"] / acc["dn"]) if max(V) > 0 and acc["dn"] > 1e-30 else float("nan")
    return rel_ns, math.sqrt(acc["rc"] / acc["dc"])


def surface_budget(model, dev, surfaces, t, V, n_people):
    ramp = math.tanh(3.0 * t / TAU_RAMP)
    r = {"Q_target": sum(V[k] * ramp * (x1 - x0) * (z1 - z0) for k, (x0, x1, z0, z1) in enumerate(WINDOWS))}
    Q_in = Q_doors = leak_net = leak_gross = closure = 0.0
    F = {"window": 0.0, "door": 0.0, "wall": 0.0, "column": 0.0}
    slip_num = slip_den = 0.0
    win_num = win_den = 0.0
    for kind, k, P, dA, nrm in surfaces:
        f = evaluate(model, dev, P, t, V, n_people, grad_c=True)
        un = f["u"] * nrm[:, 0] + f["v"] * nrm[:, 1] + f["w"] * nrm[:, 2]
        dcdn = f["cx"] * nrm[:, 0] + f["cy"] * nrm[:, 1] + f["cz"] * nrm[:, 2]
        flux = np.sum(un * dA)
        closure += flux
        solid = kind in ("wall", "column") or (kind == "window" and V[k] == 0)
        F["wall" if solid else kind] += np.sum((f["c"] * un - DIFFUSIVITY * dcdn) * dA)
        if kind == "window" and not solid:
            Q_in -= flux
            target = -V[k] * ramp
            e2 = f["u"] ** 2 + (f["v"] - target) ** 2 + f["w"] ** 2
            win_num += np.sum(e2 * dA) / (V[k] * ramp) ** 2
            win_den += np.sum(dA)
        elif kind == "door":
            Q_doors += flux
        else:
            leak_net += flux
            leak_gross += np.sum(np.abs(un) * dA)
            slip_num += np.sum((f["u"] ** 2 + f["v"] ** 2 + f["w"] ** 2) * dA)
            slip_den += np.sum(dA)
    open_v = [V[k] * ramp for k in range(NUM_WINDOWS) if V[k] > 0]
    v_ref = float(np.mean(open_v)) if open_v else float("nan")
    r.update(Q_in=Q_in, Q_doors=Q_doors, leak_net=leak_net, leak_gross=leak_gross, closure=closure,
             wall_slip=math.sqrt(slip_num / slip_den) / v_ref if open_v else float("nan"),
             wall_speed_rms=math.sqrt(slip_num / slip_den),
             win_err=math.sqrt(win_num / win_den) if win_den > 0 else float("nan"),
             F_window=F["window"], F_door=F["door"], F_wall=F["wall"] + F["column"],
             F_out=sum(F.values()))
    return r


def volume_budget(model, dev, vol, dV, t, V, n_people):
    f = evaluate(model, dev, vol, t, V, n_people, dcdt=True)
    return {"E": np.sum(source(vol, n_people)) * dV, "dMdt": np.sum(f["ct"]) * dV, "M": np.sum(f["c"]) * dV}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cpu", help="cpu (default, leaves the GPU to training) or cuda")
    ap.add_argument("--n-res", type=int, default=1000, help="interior points for the residual check (0 = skip)")
    ap.add_argument("--n-random", type=int, default=5, help="random window settings added to the fixed ones")
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    dev = args.device
    ckpt = torch.load(args.checkpoint, map_location=dev)
    check_checkpoint_compat(ckpt, args.checkpoint)
    model = GNOTOperator().to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    version = ckpt.get("version", "unknown")
    out_dir = os.path.join(HERE, "figures", version, "physics_checks")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Checkpoint {args.checkpoint} (version={version}, iter={ckpt.get('iter', '?')}), N={N_PEOPLE:.0f}, "
          f"device={dev}")

    surfaces = build_surfaces()
    vol, dV = build_volume()
    print(f"quadrature: {sum(len(s[2]) for s in surfaces)} surface points, {len(vol)} volume cells "
          f"(total surface {sum(s[3].sum() for s in surfaces):.1f} m^2, volume {len(vol) * dV:.1f} m^3)")

    rows = []
    rng = np.random.default_rng(1)
    for name, V in scenarios(args.n_random):
        t0 = time.time()
        rel_ns, rel_co2 = (residuals(model, dev, V, N_PEOPLE, args.n_res, rng) if args.n_res > 0
                           else (float("nan"), float("nan")))
        for t in TIMES:
            r = {"scenario": name, "V": " ".join(f"{v:g}" for v in V), "t": t, "rel_NS": rel_ns, "rel_CO2": rel_co2}
            r.update(surface_budget(model, dev, surfaces, t, V, N_PEOPLE))
            r.update(volume_budget(model, dev, vol, dV, t, V, N_PEOPLE))
            r["co2_imbalance"] = (r["dMdt"] + r["F_out"] - r["E"]) / r["E"]
            rows.append(r)
        print(f"  {name:12s} done ({time.time() - t0:5.1f} s)", flush=True)

    csv_path = os.path.join(out_dir, "level3_consistency.csv")
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    pct = lambda a: "   -  " if not np.isfinite(a) else f"{a * 100:6.1f}"
    print(f"\nAIR BALANCE at t=60 s [m^3/s] (leak = through walls/floor/ceiling/columns; closure ~0 = quadrature OK)")
    print(f"{'scenario':12s} {'Q_target':>8s} {'Q_in':>7s} {'Q_doors':>7s} {'leak_net':>8s} {'leak_gross':>10s} "
          f"{'closure':>8s} | {'slip%':>6s} {'winErr%':>7s}")
    for r in rows:
        if r["t"] != 60.0:
            continue
        print(f"{r['scenario']:12s} {r['Q_target']:8.3f} {r['Q_in']:7.3f} {r['Q_doors']:7.3f} {r['leak_net']:8.3f} "
              f"{r['leak_gross']:10.3f} {r['closure']:8.3f} | {pct(r['wall_slip'])} {pct(r['win_err']):>7s}")
    print(f"\nCO2 BUDGET (imbalance = share of emitted CO2 created(+)/destroyed(-) by the model) and PDE residuals")
    print(f"{'scenario':12s} {'imb t60%':>8s} {'imb t120%':>9s} | out via doors/windows/walls at t120 (% of E) | "
          f"{'rel_NS':>6s} {'rel_CO2':>7s}")
    by = {}
    for r in rows:
        by.setdefault(r["scenario"], {})[r["t"]] = r
    for name, d in by.items():
        a, b = d[60.0], d[120.0]
        print(f"{name:12s} {pct(a['co2_imbalance']):>8s} {pct(b['co2_imbalance']):>9s} | "
              f"{pct(b['F_door'] / b['E'])} / {pct(b['F_window'] / b['E'])} / {pct(b['F_wall'] / b['E'])}"
              f"{'':17s}| {a['rel_NS']:6.3f} {a['rel_CO2']:7.3f}")
    print("\nReading: 'closed' is the validated calibration row. Open-window rows are suspicious if their")
    print("leakage is a large share of Q_in, win err or slip are large, the CO2 imbalance is far from 0,")
    print("or the relative residuals are clearly above the closed row.")
    print(f"\nTable written to {csv_path}")


if __name__ == "__main__":
    main()
