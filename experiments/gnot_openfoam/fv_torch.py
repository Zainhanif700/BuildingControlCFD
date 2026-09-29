"""
The FV CO2 solver of experiments/gnot/check_co2_with_model_flow.py (Grid.rhs + SSP-RK3 in solve_co2),
ported LINE BY LINE to torch so that 30-minute transports run on the GPU (numpy on one CPU core:
~0.17 s per step at dx = 0.1 -> 3-6 h per case for 1800 s). Same grid, masks, boundary rules,
source, upwind-2nd-order advection, time step rule and velocity interpolation; the Grid object
(masks, source) is taken from the numpy module itself, so geometry cannot drift.
Beyond the last velocity snapshot the flow is FROZEN (velocity_at clamps, as in the numpy code).

verify(): runs both solvers on the same snapshots and reports the max relative difference.
"""
import numpy as np
import torch

import common  # noqa: F401
import check_co2_with_model_flow as L2
from train_gnot import DIFFUSIVITY


def _shift(a, axis, s):
    """torch version of L2.shift: b[i] = a[i+s] along axis, out-of-range entries NaN."""
    b = torch.full_like(a, float("nan"))
    src, dst = [slice(None)] * 3, [slice(None)] * 3
    if s > 0:
        dst[axis], src[axis] = slice(0, -s), slice(s, None)
    else:
        dst[axis], src[axis] = slice(-s, None), slice(0, s)
    b[tuple(dst)] = a[tuple(src)]
    return b


class TorchFV:
    def __init__(self, g, device="cuda", dtype=torch.float64):
        self.g, self.dev, self.dt_ = g, device, dtype
        T = lambda a: torch.tensor(np.asarray(a), device=device)
        self.fluid = T(g.fluid)
        self.fluid_f = self.fluid.to(dtype)
        self.S = T(g.S).to(dtype)
        self.nb_ok = {k: T(v) for k, v in g.nb_ok.items()}
        self.open_top = T(g.open_top)
        self.h = [float(h) for h in g.h]

    def neighbour(self, c, axis, s):
        nb = _shift(c, axis, s)
        out = torch.where(self.nb_ok[(axis, s)], nb, c)
        if axis == 1 and s == 1:
            out = torch.where(self.open_top, -c, out)
        return out

    def rhs(self, c, u, v, w):
        lap = torch.zeros_like(c)
        adv = torch.zeros_like(c)
        for a, ua in enumerate((u, v, w)):
            h = self.h[a]
            m1, p1 = self.neighbour(c, a, -1), self.neighbour(c, a, 1)
            lap = lap + (p1 - 2.0 * c + m1) / h ** 2
            back = (c - m1) / h
            fwd = (p1 - c) / h
            ok_b = self.nb_ok[(a, -1)] & self.nb_ok[(a, -2)]
            ok_f = self.nb_ok[(a, 1)] & self.nb_ok[(a, 2)]
            m2, p2 = _shift(c, a, -2), _shift(c, a, 2)
            back = torch.where(ok_b, (3.0 * c - 4.0 * m1 + torch.nan_to_num(m2)) / (2.0 * h), back)
            fwd = torch.where(ok_f, (-3.0 * c + 4.0 * p1 - torch.nan_to_num(p2)) / (2.0 * h), fwd)
            adv = adv + torch.where(ua > 0, ua * back, ua * fwd)
        return (DIFFUSIVITY * lap - adv + self.S) * self.fluid_f

    def solve(self, snap_times, snaps, out_times, log_every=None):
        """snaps: list of [u, v, w] numpy arrays (grid shape) at snap_times; returns {t: c (numpy)}.
        Time step rule identical to L2.solve_co2."""
        g = self.g
        umax = max(np.max(np.abs(s[0])) / g.h[0] + np.max(np.abs(s[1])) / g.h[1] + np.max(np.abs(s[2])) / g.h[2]
                   for s in snaps)
        dt_adv = 0.5 / umax if umax > 0 else np.inf
        dt_diff = 0.25 / (DIFFUSIVITY * sum(1.0 / h ** 2 for h in g.h))
        dt = min(dt_adv, dt_diff, 0.5)
        S = [[torch.tensor(a, device=self.dev, dtype=self.dt_) for a in s] for s in snaps]
        ts = list(snap_times)

        def vel(t):     # = L2.velocity_at (linear in time, clamped -> frozen after the last snapshot)
            j = int(np.searchsorted(ts, t, side="right")) - 1
            j = min(max(j, 0), len(ts) - 2)
            a = (t - ts[j]) / (ts[j + 1] - ts[j])
            a = min(max(a, 0.0), 1.0)
            return [(1 - a) * S[j][k] + a * S[j + 1][k] for k in range(3)]
        c = torch.zeros(g.X.shape, device=self.dev, dtype=self.dt_)
        out, t, n = {}, 0.0, 0
        for t_end in out_times:
            while t < t_end - 1e-9:
                step = min(dt, t_end - t)
                u0, u1, uh = vel(t), vel(t + step), vel(t + 0.5 * step)
                k1 = c + step * self.rhs(c, *u0)
                k2 = 0.75 * c + 0.25 * (k1 + step * self.rhs(k1, *u1))
                c = c / 3.0 + 2.0 / 3.0 * (k2 + step * self.rhs(k2, *uh))
                t += step
                n += 1
                if log_every and n % log_every == 0:
                    print(f"    FV t = {t:7.1f} s ({n} steps)", flush=True)
            out[t_end] = c.detach().cpu().numpy()
        return out, dt, n


def verify(g, snap_times, snaps, t_check=(30.0, 60.0), device="cuda"):
    """Both solvers, same inputs: max |c_torch - c_numpy| / max |c_numpy| at t_check."""
    L2.T_SNAP = list(snap_times)
    L2.T_OUT = tuple(t_check)
    ref, _, _ = L2.solve_co2(g, snaps)
    got, _, _ = TorchFV(g, device).solve(snap_times, snaps, t_check)
    return {t: float(np.max(np.abs(got[t] - ref[t])) / max(np.max(np.abs(ref[t])), 1e-30)) for t in t_check}
