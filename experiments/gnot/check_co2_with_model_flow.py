"""
LEVEL 2 open-window check: is the model's CO2 consistent with the model's OWN airflow?

For each window setting, the GNOT velocity field is sampled on a finite-volume
grid (at snapshot times, linear in between), and the CO2 equation
    dc/dt + u.grad(c) = D lap(c) + S,     c(t=0) = 0
is solved independently with a standard finite-volume scheme and the SAME
boundary conditions as training:
    walls, floor, ceiling, columns, CLOSED windows : no flux (zero gradient)
    OPEN windows                                   : clean inflow, c = 0
    doors                                          : zero-gradient outflow
Then the GNOT CO2 is compared with this solution (relative L2 on the breathing
plane and over the whole room, at t = 60 and 120 s).

What it shows: whether the CO2 half of the model is right FOR THE FLOW THE MODEL
PREDICTS. It does NOT check the flow itself (that needs level 1, an independent
Navier-Stokes solver). A small error here + a flow that passes level 3 = good
evidence; a large error = the CO2 transport is not consistent with the flow.

Numerics: cell-centred grid (columns = solid cells), advective form u.grad(c)
with second-order upwind differences (first order next to boundaries/solids),
7-point Laplacian with the boundary conditions above, SSP-RK3 time stepping,
dt from the advective (CFL 0.5) and diffusive limits. The advective form is used
because the sampled velocity is not exactly divergence-free on the grid (a
conservative form would then create spurious CO2); velocity components through
solid boundaries are ignored by the zero-gradient ghost values.

SELF-TEST: the 'closed' scenario has zero velocity, so this solver reduces to the
closed-room finite-difference reference -- its breathing-plane error at t = 60 s
must match validate_closed_room.py (z = 1.10 m, t = 60 s) to within ~1%.
GRID CHECK: run once with --dx 0.2 (default) and once with --dx 0.1 on a few
scenarios; the reference is trustworthy where the two agree.

Usage:
    python3 check_co2_with_model_flow.py [checkpoint] [--dx 0.2] [--scenarios closed "W8 3m/s" ...]
Writes figures/<version>/physics_checks/level2_co2_consistency_dx<dx>.csv
GRID CHECK RESULT (v13, 2026-09-27): dx 0.2 vs 0.1 agree within 0.2 percentage points in every
error column and within 0.012 in the mass ratio -> dx = 0.2 is grid-converged for this purpose.
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch

from gnot_model import GNOTOperator, check_checkpoint_compat
from point_sampler import (ROOM_X, ROOM_Y, ROOM_Z, WINDOWS, COLUMNS, BREATHING_HEIGHT, TAU_RAMP, T_MAX,
                           EMISSION_PER_PERSON, CO2_SOURCE_SIGMA)
from train_gnot import DIFFUSIVITY
from check_physics_consistency import evaluate, scenarios, DEFAULT_CKPT
from fd_reference_closed_room import interp, breathing_grid, _fill_solid

HERE = os.path.dirname(os.path.abspath(__file__))
N_PEOPLE = 20.0
T_OUT = (60.0, 120.0)
T_SNAP = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0] + [float(t) for t in range(20, 121, 10)]
DEFAULT_SCENARIOS = ["closed", "W1 3m/s", "W8 3m/s", "W1+W8 2m/s", "all 1m/s", "all 3m/s",
                     "random1u", "random2p"]
SX, SY = (ROOM_X[0] + ROOM_X[1]) / 2, (ROOM_Y[0] + ROOM_Y[1]) / 2


def shift(a, axis, s):
    """b[i] = a[i+s] along axis; out-of-range entries are NaN."""
    b = np.full_like(a, np.nan, dtype=float)
    src, dst = [slice(None)] * 3, [slice(None)] * 3
    if s > 0:
        dst[axis], src[axis] = slice(0, -s), slice(s, None)
    else:
        dst[axis], src[axis] = slice(-s, None), slice(0, s)
    b[tuple(dst)] = a[tuple(src)]
    return b


class Grid:
    def __init__(self, dx_target, V):
        L = [ROOM_X[1] - ROOM_X[0], ROOM_Y[1] - ROOM_Y[0], ROOM_Z[1] - ROOM_Z[0]]
        self.n = [max(4, round(l / dx_target)) for l in L]
        self.h = [l / n for l, n in zip(L, self.n)]
        lo = [ROOM_X[0], ROOM_Y[0], ROOM_Z[0]]
        self.c1d = [lo[a] + (np.arange(self.n[a]) + 0.5) * self.h[a] for a in range(3)]
        X, Y, Z = np.meshgrid(*self.c1d, indexing="ij")
        self.X, self.Y, self.Z = X, Y, Z
        self.fluid = np.ones(X.shape, dtype=bool)
        for cx, cy, r, _, _ in COLUMNS:
            self.fluid &= (X - cx) ** 2 + (Y - cy) ** 2 > r ** 2
        self.S = N_PEOPLE * EMISSION_PER_PERSON * np.exp(
            -((X - SX) ** 2 + (Y - SY) ** 2 + (Z - BREATHING_HEIGHT) ** 2) / CO2_SOURCE_SIGMA ** 2) * self.fluid
        # neighbour availability: nb_ok[(axis, s)] = neighbour at offset s exists and is fluid
        f = self.fluid.astype(float)
        self.nb_ok = {(a, s): (shift(f, a, s) == 1.0) for a in range(3) for s in (-2, -1, 1, 2)}
        # open-window cells on the y = ROOM_Y[1] wall (top y layer): Dirichlet c = 0 there
        xc = self.c1d[0]
        win = np.zeros(self.n[0], dtype=bool)
        for k, (xlo, xhi, _, _) in enumerate(WINDOWS):
            if V[k] > 0:
                win |= (xc >= xlo) & (xc <= xhi)
        self.open_top = np.zeros(X.shape, dtype=bool)
        self.open_top[:, -1, :] = win[:, None] & self.fluid[:, -1, :]
        self.grid_tuple = (ROOM_X[0], ROOM_Y[0], ROOM_Z[0], *self.h, *self.n)

    def neighbour(self, c, axis, s):
        """Value at offset s (|s| = 1) with the boundary rules: fluid neighbour -> its value;
        solid / wall / door / closed window -> c itself (zero gradient); open window -> -c
        (so the face value is 0)."""
        nb = shift(c, axis, s)
        out = np.where(self.nb_ok[(axis, s)], nb, c)
        if axis == 1 and s == 1:
            out = np.where(self.open_top, -c, out)
        return out

    def rhs(self, c, u, v, w):
        D = DIFFUSIVITY
        adv = np.zeros_like(c)
        lap = np.zeros_like(c)
        for a, ua in enumerate((u, v, w)):
            h = self.h[a]
            m1, p1 = self.neighbour(c, a, -1), self.neighbour(c, a, 1)
            lap += (p1 - 2.0 * c + m1) / h ** 2
            back = (c - m1) / h
            fwd = (p1 - c) / h
            ok_b = self.nb_ok[(a, -1)] & self.nb_ok[(a, -2)]
            ok_f = self.nb_ok[(a, 1)] & self.nb_ok[(a, 2)]
            m2, p2 = shift(c, a, -2), shift(c, a, 2)
            back = np.where(ok_b, (3.0 * c - 4.0 * m1 + np.nan_to_num(m2)) / (2.0 * h), back)
            fwd = np.where(ok_f, (-3.0 * c + 4.0 * p1 - np.nan_to_num(p2)) / (2.0 * h), fwd)
            adv += np.where(ua > 0, ua * back, ua * fwd)
        return (D * lap - adv + self.S) * self.fluid


def model_velocity_snapshots(model, dev, g, V):
    P = np.stack([g.X[g.fluid], g.Y[g.fluid], g.Z[g.fluid]], 1)
    snaps = []
    for t in T_SNAP:
        f = evaluate(model, dev, P, t, V, N_PEOPLE)
        uvw = []
        for k in ("u", "v", "w"):
            a = np.zeros(g.X.shape)
            a[g.fluid] = f[k]
            uvw.append(a)
        snaps.append(uvw)
    return snaps


def velocity_at(snaps, t):
    j = int(np.searchsorted(T_SNAP, t, side="right")) - 1
    j = min(max(j, 0), len(T_SNAP) - 2)
    a = (t - T_SNAP[j]) / (T_SNAP[j + 1] - T_SNAP[j])
    a = min(max(a, 0.0), 1.0)
    return [(1 - a) * snaps[j][k] + a * snaps[j + 1][k] for k in range(3)]


def solve_co2(g, snaps):
    c = np.zeros(g.X.shape)
    out, t = {}, 0.0
    umax = max(np.max(np.abs(s[0])) / g.h[0] + np.max(np.abs(s[1])) / g.h[1] + np.max(np.abs(s[2])) / g.h[2]
               for s in snaps)
    dt_adv = 0.5 / umax if umax > 0 else np.inf
    dt_diff = 0.25 / (DIFFUSIVITY * sum(1.0 / h ** 2 for h in g.h))
    dt = min(dt_adv, dt_diff, 0.5)
    n_steps = 0
    for t_end in T_OUT:
        while t < t_end - 1e-9:
            step = min(dt, t_end - t)
            u0 = velocity_at(snaps, t)
            u1 = velocity_at(snaps, t + step)
            uh = velocity_at(snaps, t + 0.5 * step)
            k1 = c + step * g.rhs(c, *u0)                                    # SSP-RK3
            k2 = 0.75 * c + 0.25 * (k1 + step * g.rhs(k1, *u1))
            c = c / 3.0 + 2.0 / 3.0 * (k2 + step * g.rhs(k2, *uh))
            t += step
            n_steps += 1
        out[t_end] = c.copy()
    return out, dt, n_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default=DEFAULT_CKPT)
    ap.add_argument("--dx", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scenarios", nargs="*", default=DEFAULT_SCENARIOS,
                    help="names from check_physics_consistency.scenarios(), or 'all'")
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
          f"dx={args.dx} m")

    all_sc = dict(scenarios(5))
    names = list(all_sc) if args.scenarios == ["all"] else args.scenarios
    unknown = [n for n in names if n not in all_sc]
    if unknown:
        raise SystemExit(f"unknown scenario(s) {unknown}; available: {list(all_sc)}")

    xs, ys, xg, yg, inside = breathing_grid()
    m = ~inside
    rows = []
    for name in names:
        V = all_sc[name]
        t0 = time.time()
        g = Grid(args.dx, V)
        snaps = model_velocity_snapshots(model, dev, g, V)
        fv, dt, n_steps = solve_co2(g, snaps)
        P = np.stack([g.X[g.fluid], g.Y[g.fluid], g.Z[g.fluid]], 1)
        for t in T_OUT:
            ref_vol = fv[t][g.fluid]
            mod_vol = evaluate(model, dev, P, t, V, N_PEOPLE)["c"]
            ref_pl = interp(_fill_solid(fv[t], g.fluid), g.grid_tuple, xg, yg, np.full_like(xg, BREATHING_HEIGHT))
            mod_pl = evaluate(model, dev, np.stack([xg, yg, np.full_like(xg, BREATHING_HEIGHT)], 1), t, V,
                              N_PEOPLE)["c"]
            src_ref = float(interp(_fill_solid(fv[t], g.fluid), g.grid_tuple, SX, SY, BREATHING_HEIGHT))
            src_mod = float(evaluate(model, dev, np.array([[SX, SY, BREATHING_HEIGHT]]), t, V, N_PEOPLE)["c"][0])
            dV = g.h[0] * g.h[1] * g.h[2]
            rows.append({
                "scenario": name, "V": " ".join(f"{v:g}" for v in V), "t": t, "dx": args.dx,
                "plane_rel_L2": np.linalg.norm(mod_pl[m] - ref_pl[m]) / max(np.linalg.norm(ref_pl[m]), 1e-30),
                "volume_rel_L2": np.linalg.norm(mod_vol - ref_vol) / max(np.linalg.norm(ref_vol), 1e-30),
                "source_rel": (src_mod - src_ref) / max(abs(src_ref), 1e-30),
                "mass_model": float(np.sum(mod_vol) * dV), "mass_ref": float(np.sum(ref_vol) * dV),
                "ref_min": float(ref_vol.min()), "dt": dt, "steps": n_steps})
        print(f"  {name:12s} done ({time.time() - t0:5.1f} s, dt={dt:.4f} s, {n_steps} steps)", flush=True)

    # dx in the name, so a grid-check run does not overwrite the main table
    csv_path = os.path.join(out_dir, f"level2_co2_consistency_dx{args.dx:g}.csv")
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"\nLEVEL 2: GNOT CO2 vs finite-volume CO2 transported by GNOT's own flow (dx={args.dx} m, N=20)")
    print(f"{'scenario':12s} {'t':>4s} | {'plane L2%':>9s} {'volume L2%':>10s} {'source%':>8s} | "
          f"{'mass model/ref':>14s} {'ref min':>9s}")
    for r in rows:
        print(f"{r['scenario']:12s} {r['t']:4.0f} | {r['plane_rel_L2'] * 100:9.1f} {r['volume_rel_L2'] * 100:10.1f} "
              f"{r['source_rel'] * 100:+8.1f} | {r['mass_model'] / max(r['mass_ref'], 1e-30):14.3f} "
              f"{r['ref_min']:9.2e}")
    print("\nSELF-TEST: the 'closed' row at t=60 must match validate_closed_room.py (z=1.10 m, t=60 s) within ~1%.")
    print("'ref min' clearly below 0 = the upwind scheme oscillates (use a finer --dx). Mass model/ref far from 1 =")
    print("the model creates/loses CO2 relative to its own flow. Grid check: repeat with --dx 0.1.")
    print(f"\nTable written to {csv_path}")


if __name__ == "__main__":
    main()
