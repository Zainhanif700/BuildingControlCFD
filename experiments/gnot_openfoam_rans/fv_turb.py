"""
CO2 transport with TURBULENT mixing: the verified FV scheme of experiments/gnot_openfoam/fv_torch.py
(same grid, masks, boundary rules, 2nd-order upwind advection, SSP-RK3, frozen flow after the last
snapshot), with a spatially varying diffusivity

    D(x, t) = D_CO2 + nu_t(x, t) / Sc_t        (nu_t from the OpenFOAM k-omega SST run)

in conservative flux form: face diffusivity = mean of the two cells; at walls / closed windows /
doors the face flux is zero (zero gradient, as before); at open-window faces C = 0 with the cell's
own D. With constant D it reduces to fv_torch exactly (test_fv_turb.py). The source is given
explicitly (seated people, common.SEAT_BOX), in ppm/s; the field is the excess over outdoor air.
"""
import numpy as np
import torch

import common
from common import D_CO2, SC_T
from fv_torch import TorchFV, _shift


class TurbFV(TorchFV):
    def __init__(self, g, S_grid, device="cuda", dtype=torch.float32, d_mol=D_CO2, sc_t=SC_T):
        super().__init__(g, device, dtype)
        self.S = torch.tensor(np.asarray(S_grid), device=device, dtype=dtype) * self.fluid_f
        self.d_mol, self.sc_t = float(d_mol), float(sc_t)

    def rhs(self, c, u, v, w, D):
        div = torch.zeros_like(c)
        adv = torch.zeros_like(c)
        for a, ua in enumerate((u, v, w)):
            h = self.h[a]
            m1, p1 = self.neighbour(c, a, -1), self.neighbour(c, a, 1)
            Dp = torch.where(self.nb_ok[(a, 1)], 0.5 * (D + _shift(D, a, 1)), D)
            Dm = torch.where(self.nb_ok[(a, -1)], 0.5 * (D + _shift(D, a, -1)), D)
            div = div + (Dp * (p1 - c) - Dm * (c - m1)) / h ** 2
            back = (c - m1) / h
            fwd = (p1 - c) / h
            ok_b = self.nb_ok[(a, -1)] & self.nb_ok[(a, -2)]
            ok_f = self.nb_ok[(a, 1)] & self.nb_ok[(a, 2)]
            m2, p2 = _shift(c, a, -2), _shift(c, a, 2)
            back = torch.where(ok_b, (3.0 * c - 4.0 * m1 + torch.nan_to_num(m2)) / (2.0 * h), back)
            fwd = torch.where(ok_f, (-3.0 * c + 4.0 * p1 - torch.nan_to_num(p2)) / (2.0 * h), fwd)
            adv = adv + torch.where(ua > 0, ua * back, ua * fwd)
        return (div - adv + self.S) * self.fluid_f

    def solve(self, snap_times, snaps, out_times, c0=None, log_every=None):
        """snaps: list of [u, v, w, nut] numpy arrays (grid shape) at snap_times; returns {t: c}.
        c0: initial field (grid shape), default 0. Time step as fv_torch, with max(D) for diffusion."""
        g = self.g
        umax = max(np.max(np.abs(s[0])) / g.h[0] + np.max(np.abs(s[1])) / g.h[1] + np.max(np.abs(s[2])) / g.h[2]
                   for s in snaps)
        dmax = max(self.d_mol + max(float(np.max(s[3])), 0.0) / self.sc_t for s in snaps)
        dt_adv = 0.5 / umax if umax > 0 else np.inf
        dt_diff = 0.25 / (dmax * sum(1.0 / h ** 2 for h in g.h))
        dt = min(dt_adv, dt_diff, 0.5)
        S = [[torch.tensor(a, device=self.dev, dtype=self.dt_) for a in s[:3]]
             + [self.d_mol + torch.clamp(torch.tensor(s[3], device=self.dev, dtype=self.dt_), min=0.0) / self.sc_t]
             for s in snaps]
        ts = list(snap_times)

        def fields(t):   # u, v, w, D -- linear in time, frozen after the last snapshot (as fv_torch)
            if t >= ts[-1]:
                return S[-1]
            j = int(np.searchsorted(ts, t, side="right")) - 1
            j = min(max(j, 0), len(ts) - 2)
            a = min(max((t - ts[j]) / (ts[j + 1] - ts[j]), 0.0), 1.0)
            return [(1 - a) * S[j][k] + a * S[j + 1][k] for k in range(4)]
        c = torch.zeros(g.X.shape, device=self.dev, dtype=self.dt_) if c0 is None else \
            torch.tensor(np.asarray(c0), device=self.dev, dtype=self.dt_) * self.fluid_f
        out, t, n = {}, 0.0, 0
        for t_end in out_times:
            while t < t_end - 1e-9:
                step = min(dt, t_end - t)
                f0, f1, fh = fields(t), fields(t + step), fields(t + 0.5 * step)
                k1 = c + step * self.rhs(c, *f0)
                k2 = 0.75 * c + 0.25 * (k1 + step * self.rhs(k1, *f1))
                c = c / 3.0 + 2.0 / 3.0 * (k2 + step * self.rhs(k2, *fh))
                t += step
                n += 1
                if log_every and n % log_every == 0:
                    print(f"    FV t = {t:7.1f} s ({n} steps)", flush=True)
            out[t_end] = c.detach().cpu().numpy()
        return out, dt, n


def seat_source(g, n_people):
    """Excess-CO2 source [ppm/s] on the grid: n_people x PPM_M3S_PER_PERSON spread uniformly over the
    fluid cells whose centre lies in SEAT_BOX (same cells as the OpenFOAM cellZone 'seats')."""
    from make_rans_case import seat_mask
    m = seat_mask(g)
    vol = m.sum() * g.h[0] * g.h[1] * g.h[2]
    return np.where(m, n_people * common.PPM_M3S_PER_PERSON / vol, 0.0)
