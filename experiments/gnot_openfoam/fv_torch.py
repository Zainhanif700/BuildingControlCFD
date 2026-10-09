"""
GPU (torch) version of the finite-volume CO2 solver of check_co2_with_model_flow.py; base of the RANS CO2 solvers.
"""
import numpy as np
import torch

import common
import check_co2_with_model_flow as L2
from train_gnot import DIFFUSIVITY


def _shift(a, axis, s):
    """b[i] = a[i + s] along axis (wraps around; every use is masked)."""
    return torch.roll(a, shifts=-s, dims=axis)


class TorchFV:
    def __init__(self, g, device="cuda", dtype=torch.float32):
        """dtype float32 for the dataset (float64 reproduces the numpy solver to round-off)."""
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
        """CO2 over time for the given velocity snapshots (linear in between, frozen after the last)."""
        g = self.g
        umax = max(np.max(np.abs(s[0])) / g.h[0] + np.max(np.abs(s[1])) / g.h[1] + np.max(np.abs(s[2])) / g.h[2]
                   for s in snaps)
        dt_adv = 0.5 / umax if umax > 0 else np.inf
        dt_diff = 0.25 / (DIFFUSIVITY * sum(1.0 / h ** 2 for h in g.h))
        dt = min(dt_adv, dt_diff, 0.5)
        S = [[torch.tensor(a, device=self.dev, dtype=self.dt_) for a in s] for s in snaps]
        ts = list(snap_times)

        def vel(t):
            if t >= ts[-1]:
                return S[-1]
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
    """Compares the numpy and torch solvers on the same input."""
    import time
    L2.T_SNAP = list(snap_times)
    L2.T_OUT = tuple(t_check)
    t0 = time.time()
    ref, _, n = L2.solve_co2(g, snaps)
    out = {"numpy_s_per_step": (time.time() - t0) / n}
    for name, dt in (("float64", torch.float64), ("float32", torch.float32)):
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        got, _, n = TorchFV(g, device, dt).solve(snap_times, snaps, t_check)
        out[f"{name}_s_per_step"] = (time.time() - t0) / n
        for t in t_check:
            out[f"{name} t={t:g}"] = float(np.max(np.abs(got[t] - ref[t])) / max(np.max(np.abs(ref[t])), 1e-30))
    return out
