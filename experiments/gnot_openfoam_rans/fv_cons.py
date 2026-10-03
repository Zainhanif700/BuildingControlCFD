"""
Mass-conserving CO2 solver (replaces the advective form of fv_turb.py).

Why: fv_turb.py advects with u.grad(c) using the OpenFOAM cell-centre velocity. That velocity is not
divergence-free on our grid, so CO2 is created/destroyed numerically (pilot: +9.5 % CO2 mass vs
OpenFOAM at 30 min even on the real flow).

What changes:
  1. face velocities = mean of the two cell-centre velocities, then PROJECTED to be exactly
     divergence-free (Poisson solve, CG): walls/closed windows no flow, open windows the fixed inflow
     V * flux_fix * tanh(3t/TAU) (as in OpenFOAM), doors free outflow (p = 0)
  2. advection in FLUX form with these face velocities -> CO2 is conserved exactly
     (only the source adds, only the doors remove, fresh air enters with c = 0)
  3. upwind face values with a minmod limiter (2nd order, no undershoots -> no negative CO2)
Diffusion (D_CO2 + nut/Sc_t), source, time stepping (SSP-RK3) and boundary rules are unchanged.
"""
import math

import numpy as np
import torch

import common  # noqa: F401
from fv_turb import TurbFV, _shift
from point_sampler import WINDOWS, DOORS, TAU_RAMP


def minmod(a, b):
    return torch.where(a * b > 0, torch.sign(a) * torch.minimum(a.abs(), b.abs()), torch.zeros_like(a))


class ConsFV(TurbFV):
    def __init__(self, g, S_grid, V, device="cuda", dtype=torch.float32, doors_open=True, **kw):
        super().__init__(g, S_grid, device, dtype, **kw)
        self.V = [float(v) for v in V]
        nx, ny, nz = g.n
        T = lambda a: torch.tensor(np.asarray(a), device=device)
        fl = self.fluid
        # interior faces (between two fluid cells), face arrays have n+1 entries along their axis
        self.open = []
        for a in range(3):
            shp = list(g.n)
            shp[a] += 1
            m = torch.zeros(shp, dtype=torch.bool, device=device)
            sl_lo, sl_hi, sl_in = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
            sl_lo[a], sl_hi[a], sl_in[a] = slice(0, -1), slice(1, None), slice(1, -1)
            m[tuple(sl_in)] = fl[tuple(sl_lo)] & fl[tuple(sl_hi)]
            self.open.append(m)
        # door faces (y = 0) and open-window faces (y = top): same selection rule as the OpenFOAM patches
        xc, zc = g.c1d[0], g.c1d[2]
        e = 0.25 * min(g.h)
        door = np.zeros((nx, nz), bool)
        if doors_open:
            for (a, b, c, d) in DOORS:
                door |= ((xc >= a) & (xc <= b))[:, None] & ((zc >= c - e) & (zc <= d))[None, :]
        self.door = T(door) & fl[:, 0, :]
        win_speed = np.zeros((nx, nz))
        for k, (a, b, _, _) in enumerate(WINDOWS):
            inx = (xc >= a) & (xc <= b)
            if self.V[k] > 0:
                flux_fix = (b - a) / (inx.sum() * g.h[0])      # as make_openfoam_case.py
                win_speed[inx, :] = self.V[k] * flux_fix
        self.win_speed = T(win_speed).to(torch.float64) * fl[:, -1, :]

    # ---------------- projection ----------------
    def _faces_from_cells(self, u, v, w, t):
        """cell-centre velocity (grid arrays, float64 torch) -> face velocity arrays fx, fy, fz"""
        F = []
        for a, ua in enumerate((u, v, w)):
            shp = list(ua.shape)
            shp[a] += 1
            f = torch.zeros(shp, dtype=torch.float64, device=self.dev)
            lo, hi, inn = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
            lo[a], hi[a], inn[a] = slice(0, -1), slice(1, None), slice(1, -1)
            f[tuple(inn)] = 0.5 * (ua[tuple(lo)] + ua[tuple(hi)])
            f = f * self.open[a]
            F.append(f)
        F[1][:, 0, :] = torch.where(self.door, v[:, 0, :], torch.zeros_like(v[:, 0, :]))     # door: free, start from cell value
        F[1][:, -1, :] = -self.win_speed * math.tanh(3.0 * t / TAU_RAMP)                     # window: fixed inflow (-y)
        return F

    def _div(self, F):
        h = self.h
        return ((F[0][1:] - F[0][:-1]) / h[0] + (F[1][:, 1:] - F[1][:, :-1]) / h[1]
                + (F[2][:, :, 1:] - F[2][:, :, :-1]) / h[2]) * self.fluid

    def _grad(self, p):
        """face gradient of a cell field: interior open faces, door faces (p = 0 at the door, half cell);
        zero on walls and windows (fixed flux)"""
        G = []
        for a in range(3):
            shp = list(p.shape)
            shp[a] += 1
            gr = torch.zeros(shp, dtype=p.dtype, device=self.dev)
            lo, hi, inn = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
            lo[a], hi[a], inn[a] = slice(0, -1), slice(1, None), slice(1, -1)
            gr[tuple(inn)] = (p[tuple(hi)] - p[tuple(lo)]) / self.h[a]
            gr = gr * self.open[a]
            G.append(gr)
        G[1][:, 0, :] = torch.where(self.door, p[:, 0, :] / (0.5 * self.h[1]), torch.zeros_like(p[:, 0, :]))
        return G

    def _lap(self, p):
        return self._div(self._grad(p))

    def project(self, u, v, w, t, tol=1e-10, max_it=5000):
        """-> divergence-free face velocities [fx, fy, fz] (float64) and the remaining relative divergence"""
        u, v, w = (torch.as_tensor(np.asarray(a), device=self.dev, dtype=torch.float64) * self.fluid for a in (u, v, w))
        F = self._faces_from_cells(u, v, w, t)
        b = self._div(F)
        # solve lap(p) = div(F) with CG on the SPD operator -lap, Jacobi preconditioner
        ones = self.fluid.to(torch.float64)
        diag = torch.zeros_like(ones)
        for a in range(3):
            o = self.open[a].to(torch.float64)
            sl_l, sl_r = [slice(None)] * 3, [slice(None)] * 3
            sl_l[a], sl_r[a] = slice(1, None), slice(0, -1)
            diag = diag + (o[tuple(sl_l)] + o[tuple(sl_r)]) / self.h[a] ** 2
        diag[:, 0, :] += self.door.to(torch.float64) * (2.0 / self.h[1] ** 2)
        diag = torch.where(self.fluid, diag, torch.ones_like(diag))
        diag = torch.where(diag > 0, diag, torch.ones_like(diag))
        A = lambda x: -self._lap(x)
        rhs = -b
        x = torch.zeros_like(rhs)
        r = rhs.clone()
        z = r / diag
        p = z.clone()
        rz = (r * z).sum()
        bn = rhs.norm()
        it = 0
        if bn > 0:
            for it in range(1, max_it + 1):
                Ap = A(p)
                alpha = rz / (p * Ap).sum()
                x = x + alpha * p
                r = r - alpha * Ap
                if r.norm() < tol * bn:
                    break
                z = r / diag
                rz_new = (r * z).sum()
                p = z + (rz_new / rz) * p
                rz = rz_new
        G = self._grad(x)
        F = [F[a] - G[a] for a in range(3)]
        F[1][:, -1, :] = -self.win_speed * math.tanh(3.0 * t / TAU_RAMP)   # unchanged by construction; keep exact
        u_scale = max(float(max(f.abs().max() for f in F)), 1e-30)
        rel_div = float(self._div(F).abs().max() * min(self.h) / u_scale)
        return F, rel_div, it

    # ---------------- transport ----------------
    def rhs_c(self, c, fx, fy, fz, D):
        # diffusion: as fv_turb (walls/doors zero gradient, open windows c = 0)
        div = torch.zeros_like(c)
        for a in range(3):
            h = self.h[a]
            m1, p1 = self.neighbour(c, a, -1), self.neighbour(c, a, 1)
            Dp = torch.where(self.nb_ok[(a, 1)], 0.5 * (D + _shift(D, a, 1)), D)
            Dm = torch.where(self.nb_ok[(a, -1)], 0.5 * (D + _shift(D, a, -1)), D)
            div = div + (Dp * (p1 - c) - Dm * (c - m1)) / h ** 2
        # advection: flux form, limited upwind face values
        adv = torch.zeros_like(c)
        for a, f in enumerate((fx, fy, fz)):
            dl = torch.where(self.nb_ok[(a, -1)], c - _shift(c, a, -1), torch.zeros_like(c))
            dr = torch.where(self.nb_ok[(a, 1)], _shift(c, a, 1) - c, torch.zeros_like(c))
            s = minmod(dl, dr)
            lo, hi, inn = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
            lo[a], hi[a], inn[a] = slice(0, -1), slice(1, None), slice(1, -1)
            cL = (c + 0.5 * s)[tuple(lo)]          # value at the face from the cell on its low side
            cR = (c - 0.5 * s)[tuple(hi)]          # ... from the cell on its high side
            flux = torch.zeros_like(f)
            fi = f[tuple(inn)]
            flux[tuple(inn)] = torch.where(fi > 0, fi * cL, fi * cR)
            if a == 1:   # doors (y = 0): outflow carries the cell value, inflow is fresh air (0); windows: inflow, 0
                fd = f[:, 0, :]
                flux[:, 0, :] = torch.where(fd < 0, fd * c[:, 0, :], torch.zeros_like(fd))
            adv = adv + (flux[tuple(hi)] - flux[tuple(lo)]) / self.h[a]
        return (div - adv + self.S) * self.fluid_f

    def solve(self, snap_times, snaps, out_times, c0=None, log_every=None, verbose=False):
        """snaps: [u, v, w, nut] numpy grid arrays at snap_times (as fv_turb). Each snapshot is projected
        once; face velocities and D are linear in time between snapshots, frozen after the last."""
        g = self.g
        S, info = [], []
        for t, s in zip(snap_times, snaps):
            F, rd, it = self.project(s[0], s[1], s[2], t)
            D = self.d_mol + torch.clamp(torch.tensor(np.asarray(s[3]), device=self.dev, dtype=self.dt_), min=0.0) / self.sc_t
            S.append([f.to(self.dt_) for f in F] + [D])
            info.append((t, rd, it))
        self.projection_info = info
        if verbose:
            print("  projection: max relative divergence %.1e, CG iterations max %d" % (max(i[1] for i in info), max(i[2] for i in info)))
        umax = max(sum(float(f.abs().max()) / g.h[a] for a, f in enumerate(s[:3])) for s in S)
        dmax = max(float(s[3].max()) for s in S)
        dt = min(0.5 / umax if umax > 0 else np.inf, 0.25 / (dmax * sum(1.0 / h ** 2 for h in g.h)), 0.5)
        ts = list(snap_times)

        def fields(t):
            if t >= ts[-1]:
                return S[-1]
            j = min(max(int(np.searchsorted(ts, t, side="right")) - 1, 0), len(ts) - 2)
            a = min(max((t - ts[j]) / (ts[j + 1] - ts[j]), 0.0), 1.0)
            return [(1 - a) * S[j][k] + a * S[j + 1][k] for k in range(4)]
        c = torch.zeros(g.X.shape, device=self.dev, dtype=self.dt_) if c0 is None else \
            torch.tensor(np.asarray(c0), device=self.dev, dtype=self.dt_) * self.fluid_f
        out, t, n = {}, 0.0, 0
        for t_end in out_times:
            while t < t_end - 1e-9:
                step = min(dt, t_end - t)
                f0, f1, fh = fields(t), fields(t + step), fields(t + 0.5 * step)
                k1 = c + step * self.rhs_c(c, *f0)
                k2 = 0.75 * c + 0.25 * (k1 + step * self.rhs_c(k1, *f1))
                c = c / 3.0 + 2.0 / 3.0 * (k2 + step * self.rhs_c(k2, *fh))
                t += step
                n += 1
                if log_every and n % log_every == 0:
                    print(f"    FV t = {t:7.1f} s ({n} steps)", flush=True)
            out[t_end] = c.detach().cpu().numpy()
        return out, dt, n
