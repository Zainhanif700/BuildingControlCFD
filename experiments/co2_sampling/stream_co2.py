"""
Our dataset CO2 solver (fv_cons.ConsFV) driven by a flow given as a time map: real OpenFOAM saves, then
either the frozen mean flow (dataset method B2) or a loop over the real saves. Saves are made divergence-free
once and kept on the CPU, so 1-s saves over 600 s fit in memory.
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "gnot_openfoam_rans")))
import common


def read_case(case, t_max=None, every=None):
    """Grid, metadata and a reader for the OpenFOAM saves of a case (times > 0, optionally thinned)."""
    import check_co2_with_model_flow as L2
    from compare_with_openfoam import read_internal, time_dirs
    from compare_rans_pilot import load_case, to_grid
    case = os.path.abspath(case)
    meta, V, g, Cc, idx, td = load_case(case, read_internal, time_dirs, L2)
    times = sorted(t for t in td if 0 < t <= (t_max or np.inf))
    if every:
        times = [t for t in times if abs(t / every - round(t / every)) < 1e-6]
    doors = [int(x) for x in meta.get("doors", "1 1").split()]

    def field(t, name):
        n = 3 if name == "U" else 1
        a = to_grid(g, idx, read_internal(os.path.join(case, td[t], name), n, len(Cc)).reshape(len(Cc), -1))
        a = np.nan_to_num(a).reshape(g.X.shape + ((3,) if n == 3 else ()))
        return a * (g.fluid[..., None] if n == 3 else g.fluid)
    return meta, V, g, doors, times, field, td


class FlowBank:
    """Projected (divergence-free) face velocities and nut of each save, on the CPU."""

    def __init__(self, fv):
        self.fv, self.F, self.nut, self.umax, self.nutmax = fv, {}, {}, 0.0, 0.0

    def add(self, key, t_flow, u, nut):
        F, _, _ = self.fv.project(u[..., 0], u[..., 1], u[..., 2], t_flow)
        self.F[key] = [f.float().cpu() for f in F]
        self.nut[key] = np.clip(nut, 0.0, None).astype(np.float32)
        self.umax = max(self.umax, sum(float(f.abs().max()) / h for f, h in zip(F, self.fv.h)))
        self.nutmax = max(self.nutmax, float(self.nut[key].max()))


def run(fv, bank, flow_at, out_times, sct, log=None):
    """CO2 with SSP-RK3; flow_at(t) -> (key0, key1, weight). Returns {t: CO2 grid} at out_times."""
    import torch
    dev = fv.dev
    d_mol = fv.d_mol
    dt = min(0.5 / bank.umax, 0.25 / ((d_mol + bank.nutmax / sct) * sum(1.0 / h ** 2 for h in fv.h)), 0.5)
    gpu = {}

    def entry(k):
        if k not in gpu:
            if len(gpu) > 4:
                gpu.pop(next(iter(gpu)))
            nut = torch.tensor(bank.nut[k], device=dev)
            gpu[k] = [f.to(dev) for f in bank.F[k]] + [(d_mol + nut / sct).float()]
        return gpu[k]

    def fields(t):
        k0, k1, a = flow_at(t)
        e0 = entry(k0)
        if k1 is None or a <= 0.0:
            return e0
        e1 = entry(k1)
        return [(1 - a) * x + a * y for x, y in zip(e0, e1)]

    c = torch.zeros(fv.g.X.shape, device=dev, dtype=torch.float32)
    out, t, n, t0 = {}, 0.0, 0, time.time()
    for t_end in sorted(out_times):
        while t < t_end - 1e-9:
            step = min(dt, t_end - t)
            f0, f1, fh = fields(t), fields(t + step), fields(t + 0.5 * step)
            k1 = c + step * fv.rhs_c(c, *f0)
            k2 = 0.75 * c + 0.25 * (k1 + step * fv.rhs_c(k1, *f1))
            c = c / 3.0 + 2.0 / 3.0 * (k2 + step * fv.rhs_c(k2, *fh))
            t += step
            n += 1
        out[t_end] = c.detach().cpu().numpy()
        if log:
            print(f"    t = {t_end:6.0f} s  ({n} steps, dt {dt:.4f} s, {time.time() - t0:.0f} s)", flush=True)
    return out


def interp_keys(times):
    """flow_at for t inside the saved range [0, times[-1]] (0 = still air)."""
    ts = [0.0] + list(times)

    def at(t):
        if t >= ts[-1]:
            return ts[-1], None, 0.0
        j = max(int(np.searchsorted(ts, t, side="right")) - 1, 0)
        return ts[j], ts[j + 1], (t - ts[j]) / (ts[j + 1] - ts[j])
    return at


def map_b2(times_real, mean_key="mean", t_switch=10.0):
    """Dataset method B2: real saves up to times_real[-1], then over t_switch s to the frozen mean flow."""
    real = interp_keys(times_real)
    t_r = times_real[-1]

    def at(t):
        if t <= t_r:
            return real(t)
        a = min((t - t_r) / t_switch, 1.0)
        return (t_r, mean_key, a) if a < 1.0 else (mean_key, None, 0.0)
    return at


def map_loop(times, t_loop0):
    """Real saves up to times[-1], then again and again the saves from t_loop0 to times[-1]."""
    real = interp_keys(times)
    t_r = times[-1]
    period = t_r - t_loop0

    def at(t):
        if t <= t_r:
            return real(t)
        return real(t_loop0 + math.fmod(t - t_r, period))
    return at
