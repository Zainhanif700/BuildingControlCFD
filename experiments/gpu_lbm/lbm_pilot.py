"""
GPU pilot: the classroom flow with a lattice Boltzmann LES (D3Q19, regularized BGK + Smagorinsky) in PyTorch,
and the CO2 from the seats computed in the same time loop. Same room, windows, door, columns and seats as the
OpenFOAM cases. Saves the flow and CO2 on the 8 cm dataset grid, so compare_lbm.py can compare it with OpenFOAM.
Only reads the settings of experiments/gnot_openfoam_rans; writes only to data_lbm/ in this folder.
Usage: python3 lbm_pilot.py --scenario N10 --dx 0.04
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RANS_DIR = os.path.normpath(os.path.join(HERE, "..", "gnot_openfoam_rans"))
sys.path.insert(0, RANS_DIR)
import common
from common import NU_AIR, D_CO2, SC_T, N_REF, SEAT_BOX, PPM_M3S_PER_PERSON, BREATHING_Z
from point_sampler import ROOM_X, ROOM_Y, ROOM_Z, WINDOWS, DOORS, COLUMNS, TAU_RAMP

T_AVG = 150.0

E = np.array([[0, 0, 0],
              [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
              [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
              [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
              [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]])
W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12)
OPP = [int(np.where((E == -e).all(1))[0][0]) for e in E]
PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def read_scenario(name, path=os.path.join(RANS_DIR, "scenarios_v2.txt")):
    for line in open(path):
        p = line.split()
        if p and not p[0].startswith("#") and p[0] == name:
            return [float(v) for v in p[2].split(",")], [int(d) for d in p[3].split(",")]
    raise SystemExit(f"{name} not found in {path}")


class Room:
    """Lattice nodes of the room (cubic cells of about dx), plus one layer of wall nodes around it."""

    def __init__(self, dx, V, doors):
        L = [ROOM_X[1] - ROOM_X[0], ROOM_Y[1] - ROOM_Y[0], ROOM_Z[1] - ROOM_Z[0]]
        self.n = [max(4, round(l / dx)) for l in L]
        self.h = [l / n for l, n in zip(L, self.n)]
        self.L = L
        self.c1d = [(np.arange(n) + 0.5) * h for n, h in zip(self.n, self.h)]
        X, Y, Z = np.meshgrid(*self.c1d, indexing="ij")
        self.X, self.Y, self.Z = X, Y, Z
        fluid = np.ones(X.shape, bool)
        for cx, cy, r, _, _ in COLUMNS:
            fluid &= (X - cx) ** 2 + (Y - cy) ** 2 > r ** 2
        self.fluid = fluid
        xc, zc = self.c1d[0], self.c1d[2]
        e = 0.25 * min(self.h)
        self.door = np.zeros((self.n[0], self.n[2]), bool)
        for (a, b, c, d), op in zip(DOORS, doors):
            if op:
                self.door |= ((xc >= a) & (xc <= b))[:, None] & ((zc >= c - e) & (zc <= d))[None, :]
        self.door &= fluid[:, 0, :]
        self.win = np.zeros((self.n[0], self.n[2]))
        for k, (a, b, _, _) in enumerate(WINDOWS):
            inx = (xc >= a) & (xc <= b)
            if V[k] > 0:
                self.win[inx, :] = V[k] * (b - a) / (inx.sum() * self.h[0])
        self.win *= fluid[:, -1, :]
        (x0, x1), (y0, y1), (z0, z1) = SEAT_BOX
        self.seats = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1) & (Z >= z0) & (Z <= z1) & fluid


class LBM:
    def __init__(self, room, dt, dx, cs_smag, device, dtype=torch.float32):
        self.r, self.dev, self.dt_ = room, device, dtype
        self.dx, self.dt = dx, dt
        nx, ny, nz = room.n
        self.N = (nx + 2, ny + 2, nz + 2)
        T = lambda a, t=dtype: torch.tensor(np.asarray(a), device=device, dtype=t)
        self.e = [tuple(int(c) for c in e) for e in E]
        self.w = T(W)
        self.tau = 3.0 * NU_AIR * dt / dx ** 2 + 0.5
        self.cs2 = cs_smag ** 2
        mom = np.zeros((10, 19))
        mom[0] = 1
        mom[1:4] = E.T
        for k, (a, b) in enumerate(PAIRS):
            mom[4 + k] = E[:, a] * E[:, b]
        self.M = T(mom)
        post = np.zeros((19, 10))
        post[:, 0] = W
        post[:, 1:4] = 3.0 * W[:, None] * E
        for k, (a, b) in enumerate(PAIRS):
            q = E[:, a] * E[:, b] - (1 / 3 if a == b else 0.0)
            post[:, 4 + k] = 4.5 * W * q * (1 if a == b else 2)
        self.P = T(post)

        solid = np.ones(self.N, bool)
        solid[1:-1, 1:-1, 1:-1] = ~room.fluid
        solid[1:-1, 0, 1:-1] &= ~room.door
        self.solid = T(solid, torch.bool)
        self.solid_idx = torch.nonzero(self.solid.reshape(-1)).squeeze(1)
        self.door_idx = [T(np.nonzero(room.door)[a], torch.long) + 1 for a in (0, 1)]
        self.bb = [None] + [torch.roll(self.solid, self.e[i], (0, 1, 2)) & ~self.solid for i in range(1, 19)]
        win = np.zeros(self.N)
        win[1:-1, -1, 1:-1] = room.win * dt / dx
        win = T(win)
        self.inlet = {i: 6.0 * W[i] * torch.roll(win, self.e[i], (0, 1, 2)) for i in range(19) if self.e[i][1] == -1}
        self.f = self.w.view(19, 1, 1, 1).expand(19, *self.N).contiguous()

    def step(self, ramp):
        m = torch.einsum("mq,q...->m...", self.M, self.f)
        rho = m[0]
        u = m[1:4] / rho
        A = torch.empty_like(m[4:])
        q2 = torch.zeros_like(rho)
        for k, (a, b) in enumerate(PAIRS):
            ruu = rho * u[a] * u[b]
            neq = m[4 + k] - ruu - (rho / 3.0 if a == b else 0.0)
            q2 += neq * neq * (1.0 if a == b else 2.0)
            A[k] = neq
        tau_e = 0.5 * (self.tau + torch.sqrt(self.tau ** 2 + 18.0 * math.sqrt(2.0) * self.cs2 * torch.sqrt(q2) / rho))
        keep = 1.0 - 1.0 / tau_e
        for k, (a, b) in enumerate(PAIRS):
            A[k] = rho * u[a] * u[b] + keep * A[k]
        mpost = torch.cat([rho[None], m[1:4], A])
        fp = torch.einsum("qm,m...->q...", self.P, mpost)
        dx_, dz_ = self.door_idx
        ud, rd = u[:, dx_, 1, dz_], rho[dx_, 1, dz_]
        cu = torch.einsum("qd,dn->qn", self.M[1:4].T, ud)
        feq = self.w[:, None] * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * (ud ** 2).sum(0))
        fp[:, dx_, 0, dz_] = feq + fp[:, dx_, 1, dz_] - rd * feq
        for i in range(19):
            fi = torch.roll(fp[i], self.e[i], (0, 1, 2))
            if i:
                fi = torch.where(self.bb[i], fp[OPP[i]], fi)
            if i in self.inlet:
                fi = fi + ramp * self.inlet[i]
            self.f[i] = fi
        self.f.view(19, -1)[:, self.solid_idx] = self.w[:, None]
        nut = (tau_e - self.tau) / 3.0 * self.dx ** 2 / self.dt
        return rho, u * (self.dx / self.dt), nut


def minmod(a, b):
    return torch.where(a * b > 0, torch.sign(a) * torch.minimum(a.abs(), b.abs()), torch.zeros_like(a))


class CO2:
    """Excess CO2 [ppm] on the room nodes: flux form, minmod-limited upwind, fresh air at the windows."""

    def __init__(self, room, device, sc_t):
        self.r, self.dev, self.sc_t = room, device, sc_t
        T = lambda a, t=torch.float32: torch.tensor(np.asarray(a), device=device, dtype=t)
        self.fl = T(room.fluid, torch.bool)
        self.flf = self.fl.float()
        self.h = room.h
        vol = room.seats.sum() * np.prod(room.h)
        self.S = T(np.where(room.seats, N_REF * PPM_M3S_PER_PERSON / vol, 0.0))
        self.door = T(room.door, torch.bool)
        self.win = T(room.win)
        self.open = []
        for a in range(3):
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[a], hi[a] = slice(0, -1), slice(1, None)
            self.open.append(self.fl[tuple(lo)] & self.fl[tuple(hi)])
        self.c = torch.zeros(room.n, device=device)
        self.out = torch.zeros((), device=device, dtype=torch.float64)
        self.added = 0.0
        self.S_total = float(self.S.sum()) * float(np.prod(room.h))
        self.d_max = torch.zeros((), device=device)

    def step(self, u, nut, dt):
        c = self.c
        D = (D_CO2 + torch.clamp(nut, min=0.0) / self.sc_t) * self.flf
        self.d_max = torch.maximum(self.d_max, D.max())
        dc = self.S.clone()
        for a in range(3):
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[a], hi[a] = slice(0, -1), slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            op = self.open[a]
            dif = torch.zeros_like(c)
            dif[hi] = torch.where(op, c[hi] - c[lo], torch.zeros_like(c[hi]))
            dib = torch.zeros_like(c)
            dib[lo] = torch.where(op, c[hi] - c[lo], torch.zeros_like(c[lo]))
            s = minmod(dif, dib)
            uf = torch.where(op, 0.5 * (u[a][lo] + u[a][hi]), torch.zeros_like(c[lo]))
            flux = torch.where(uf > 0, uf * (c + 0.5 * s)[lo], uf * (c - 0.5 * s)[hi])
            flux = flux - torch.where(op, 0.5 * (D[lo] + D[hi]) * (c[hi] - c[lo]) / self.h[a], torch.zeros_like(flux))
            dc[lo] -= flux / self.h[a]
            dc[hi] += flux / self.h[a]
            if a == 1:
                ud = u[1][:, 0, :]
                fd = torch.where(self.door & (ud < 0), ud * c[:, 0, :], torch.zeros_like(ud))
                dc[:, 0, :] += fd / self.h[1]
                self.out = self.out - fd.sum().double() * (self.h[0] * self.h[2] * dt)
        self.c = (c + dt * dc) * self.flf
        self.added += self.S_total * dt


def resample(field, room, g):
    """Trilinear interpolation of room-node fields (C, nx, ny, nz) to the cell centres of grid g."""
    gx, gy, gz = (torch.tensor(2.0 * g.c1d[a] / room.L[a] - 1.0, device=field.device, dtype=field.dtype) for a in range(3))
    X, Y, Z = torch.meshgrid(gx, gy, gz, indexing="ij")
    grid = torch.stack([Z, Y, X], -1)[None]
    return torch.nn.functional.grid_sample(field[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="N10", help="name in scenarios_v2.txt")
    ap.add_argument("--dx", type=float, default=0.04)
    ap.add_argument("--t-end", type=float, default=600.0)
    ap.add_argument("--save-dt", type=float, default=10.0)
    ap.add_argument("--u-lu", type=float, default=0.08, help="lattice speed of the fastest inflow (stability)")
    ap.add_argument("--smagorinsky", type=float, default=0.17)
    ap.add_argument("--sct", type=float, default=SC_T, help="turbulent Schmidt number for the sub-grid CO2 mixing")
    ap.add_argument("--grid-dx", type=float, default=0.08, help="output grid (the dataset grid)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = os.path.abspath(args.out or os.path.join(HERE, "data_lbm", f"{args.scenario}_lbm_dx{args.dx:g}.npz"))
    if not out.startswith(os.path.join(HERE, "data_lbm") + os.sep):
        raise SystemExit(f"--out must be inside {os.path.join(HERE, 'data_lbm')} (keeps the pilot away from the dataset)")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists -- delete it or give another --out")
    V, doors = read_scenario(args.scenario)
    room = Room(args.dx, V, doors)
    assert room.win.max() > 0, "no open window"
    u_door = float(room.win.sum() / room.door.sum())
    umax = max(float(room.win.max()), u_door)
    n_save = math.ceil(args.save_dt * umax / (args.u_lu * args.dx))
    dt = args.save_dt / n_save
    lbm = LBM(room, dt, args.dx, args.smagorinsky, args.device)
    co2 = CO2(room, args.device, args.sct)
    g = Room(args.grid_dx, V, doors)
    fl8 = torch.tensor(g.fluid, device=args.device)
    n_total = round(args.t_end / dt)
    print(f"{args.scenario}: V = {V}, doors {doors}; lattice {room.n} = {np.prod(room.n) / 1e6:.2f} M nodes, "
          f"h = {[round(h, 4) for h in room.h]} m, dt = {dt * 1e3:.3f} ms, tau = {lbm.tau:.6f}, "
          f"window inflow {float(room.win.max()):.3f} m/s, door outflow about {u_door:.3f} m/s; fastest "
          f"{umax * dt / args.dx:.3f} lattice units, {n_total} steps", flush=True)

    t_snap, U_snap, nut_snap, C_snap = [0.0], [], [], []
    zero = torch.zeros((int(fl8.sum()),), device=args.device)
    U_snap.append(torch.zeros((int(fl8.sum()), 3), device=args.device))
    nut_snap.append(zero.clone())
    C_snap.append(zero.clone())
    acc = None
    n_acc = 0
    t0 = time.time()
    for n in range(1, n_total + 1):
        t = n * dt
        ramp = math.tanh(3.0 * t / TAU_RAMP)
        rho, u, nut = lbm.step(ramp)
        u_r = u[:, 1:-1, 1:-1, 1:-1] * co2.flf
        nut_r = nut[1:-1, 1:-1, 1:-1] * co2.flf
        co2.step(u_r, nut_r, dt)
        if t > T_AVG + 1e-9:
            x = torch.cat([u_r, nut_r[None]])
            acc = x.clone() if acc is None else acc + x
            n_acc += 1
        if n % n_save == 0:
            rr = rho[1:-1, 1:-1, 1:-1][co2.fl]
            if not torch.isfinite(rr).all():
                raise SystemExit(f"unstable at t = {t:.1f} s -- try a smaller --u-lu or a finer --dx")
            F = resample(torch.cat([u_r, nut_r[None], co2.c[None]]), room, g)
            t_snap.append(round(t, 6))
            U_snap.append(F[:3].permute(1, 2, 3, 0)[fl8])
            nut_snap.append(F[3][fl8])
            C_snap.append(F[4][fl8])
            mass = float(co2.c.double().sum()) * float(np.prod(room.h))
            dnum = float(co2.d_max) * dt * sum(1.0 / h ** 2 for h in room.h)
            if dnum > 0.5:
                raise SystemExit(f"CO2 step unstable at t = {t:.1f} s (diffusion number {dnum:.2f}) -- use a smaller --u-lu")
            el = time.time() - t0
            print(f"  t = {t:6.1f} s  {n / el:6.1f} steps/s  rho {float(rr.min()):.4f}-{float(rr.max()):.4f}  "
                  f"max |u| {float(u_r.norm(dim=0).max()):.2f} m/s  CO2 in room / (emitted - out) "
                  f"{mass / max(co2.added - float(co2.out), 1e-30):.4f}  diffusion number {dnum:.3f}  "
                  f"({el / 60:.1f} min, ~{el / n * (n_total - n) / 60:.0f} min left)", flush=True)
    wall = time.time() - t0
    mean = resample(acc / n_acc if n_acc else torch.cat([u_r, nut_r[None]]), room, g)
    P = np.stack([g.X[g.fluid], g.Y[g.fluid], g.Z[g.fluid]], 1).astype(np.float32)
    kz = int(np.argmin(np.abs(g.c1d[2] - BREATHING_Z)))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cpu = lambda x: x.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(
        out, P=P, plane=np.abs(P[:, 2] - g.c1d[2][kz]) < 1e-5, z_plane=np.float32(g.c1d[2][kz]),
        V=np.array(V, np.float32), doors=np.array(doors, np.int8), dx=np.float32(args.grid_dx), N_ref=np.float32(N_REF),
        t_snap=np.array(t_snap, np.float32), U_snap=np.stack([cpu(x) for x in U_snap]),
        nut_snap=np.stack([cpu(x) for x in nut_snap]), C_snap=np.stack([cpu(x) for x in C_snap]),
        U_mean=cpu(mean[:3].permute(1, 2, 3, 0)[fl8]), nut_mean=cpu(mean[3][fl8]), t_avg=np.float32(T_AVG),
        lbm_dx=np.float32(args.dx), lbm_n=np.array(room.n), lbm_dt=np.float32(dt), lbm_tau=np.float64(lbm.tau),
        u_lu=np.float32(umax * dt / args.dx), smagorinsky=np.float32(args.smagorinsky), sc_t=np.float32(args.sct),
        steps=np.int64(n_total), wall_s=np.float32(wall), device=np.array(str(torch.cuda.get_device_name(0))
                                                                          if args.device.startswith("cuda") else "cpu"))
    print(f"done: {n_total} steps in {wall / 60:.1f} min ({wall / args.t_end:.2f} s per simulated s); saved {out}")


if __name__ == "__main__":
    main()
