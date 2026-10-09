"""
Mass-conserving CO2 solver used for the dataset: divergence-free face velocities, flux-form advection with a minmod limiter.
"""
import math

import numpy as np
import torch

import common
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
        self.open = []
        for a in range(3):
            shp = list(g.n)
            shp[a] += 1
            m = torch.zeros(shp, dtype=torch.bool, device=device)
            sl_lo, sl_hi, sl_in = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
            sl_lo[a], sl_hi[a], sl_in[a] = slice(0, -1), slice(1, None), slice(1, -1)
            m[tuple(sl_in)] = fl[tuple(sl_lo)] & fl[tuple(sl_hi)]
            self.open.append(m)
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
                flux_fix = (b - a) / (inx.sum() * g.h[0])
                win_speed[inx, :] = self.V[k] * flux_fix
        self.win_speed = T(win_speed).to(torch.float64) * fl[:, -1, :]

    def _faces_from_cells(self, u, v, w, t):
        """Cell velocities -> face velocities (fixed inflow at the windows, free outflow at the doors)."""
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
        F[1][:, 0, :] = torch.where(self.door, v[:, 0, :], torch.zeros_like(v[:, 0, :]))
        F[1][:, -1, :] = -self.win_speed * math.tanh(3.0 * t / TAU_RAMP)
        return F

    def _div(self, F):
        h = self.h
        return ((F[0][1:] - F[0][:-1]) / h[0] + (F[1][:, 1:] - F[1][:, :-1]) / h[1]
                + (F[2][:, :, 1:] - F[2][:, :, :-1]) / h[2]) * self.fluid

    def _grad(self, p):
        """Face gradient (p = 0 at the doors, zero at walls and windows)."""
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
        """Make the face velocities divergence-free (pressure Poisson solve with CG)."""
        u, v, w = (torch.as_tensor(np.asarray(a), device=self.dev, dtype=torch.float64) * self.fluid for a in (u, v, w))
        F = self._faces_from_cells(u, v, w, t)
        b = self._div(F)
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
        F[1][:, -1, :] = -self.win_speed * math.tanh(3.0 * t / TAU_RAMP)
        u_scale = max(float(max(f.abs().max() for f in F)), 1e-30)
        rel_div = float(self._div(F).abs().max() * min(self.h) / u_scale)
        return F, rel_div, it

    def rhs_c(self, c, fx, fy, fz, D):
        div = torch.zeros_like(c)
        for a in range(3):
            h = self.h[a]
            m1, p1 = self.neighbour(c, a, -1), self.neighbour(c, a, 1)
            Dp = torch.where(self.nb_ok[(a, 1)], 0.5 * (D + _shift(D, a, 1)), D)
            Dm = torch.where(self.nb_ok[(a, -1)], 0.5 * (D + _shift(D, a, -1)), D)
            div = div + (Dp * (p1 - c) - Dm * (c - m1)) / h ** 2
        adv = torch.zeros_like(c)
        for a, f in enumerate((fx, fy, fz)):
            dl = torch.where(self.nb_ok[(a, -1)], c - _shift(c, a, -1), torch.zeros_like(c))
            dr = torch.where(self.nb_ok[(a, 1)], _shift(c, a, 1) - c, torch.zeros_like(c))
            s = minmod(dl, dr)
            lo, hi, inn = [slice(None)] * 3, [slice(None)] * 3, [slice(None)] * 3
            lo[a], hi[a], inn[a] = slice(0, -1), slice(1, None), slice(1, -1)
            cL = (c + 0.5 * s)[tuple(lo)]
            cR = (c - 0.5 * s)[tuple(hi)]
            flux = torch.zeros_like(f)
            fi = f[tuple(inn)]
            flux[tuple(inn)] = torch.where(fi > 0, fi * cL, fi * cR)
            if a == 1:
                fd = f[:, 0, :]
                flux[:, 0, :] = torch.where(fd < 0, fd * c[:, 0, :], torch.zeros_like(fd))
            adv = adv + (flux[tuple(hi)] - flux[tuple(lo)]) / self.h[a]
        return (div - adv + self.S) * self.fluid_f

    def solve(self, snap_times, snaps, out_times, c0=None, log_every=None, verbose=False):
        """CO2 over time for the given flow snapshots (linear in between, frozen after the last)."""
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
