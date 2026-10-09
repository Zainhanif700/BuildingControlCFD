"""
Tests of the mass-conserving CO2 solver: no flow, divergence, closed room, mass conservation, no negative CO2.
Usage: python3 test_fv_cons.py
"""
import sys

import numpy as np
import torch

import common
import check_co2_with_model_flow as L2
from fv_turb import TurbFV, seat_source
from fv_cons import ConsFV

FAILED = []


def check(name, ok, msg=""):
    print(f"{'PASS' if ok else 'FAIL'}: {name} {msg}", flush=True)
    if not ok:
        FAILED.append(name)


def swirl(g, amp=0.3):
    """A smooth test flow that is not divergence-free."""
    u = amp * np.sin(g.Y / 2.0) * np.cos(g.Z) * g.fluid
    v = -amp * np.cos(g.X / 3.0) * np.sin(g.Z / 2.0) * g.fluid
    w = 0.3 * amp * np.sin(g.X / 2.5) * np.cos(g.Y / 2.0) * g.fluid
    return u, v, w


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float64
    rng = np.random.default_rng(0)

    V = [1.0, 0, 0, 0, 0, 0, 0, 0.5]
    g = L2.Grid(0.25, V)
    z = np.zeros(g.X.shape)
    nut = rng.uniform(0, 0.02, g.X.shape) * g.fluid
    S = seat_source(g, 20.0)
    a, _, _ = TurbFV(g, S, dev, dt).solve([0.0, 10.0], [[z, z, z, nut]] * 2, [30.0])
    b, _, _ = ConsFV(g, S, [0.0] * 8, dev, dt).solve([0.0, 10.0], [[z, z, z, nut]] * 2, [30.0])
    e = float(np.abs(a[30.0] - b[30.0]).max() / np.abs(a[30.0]).max())
    check("1 no flow: ConsFV == TurbFV", e < 1e-10, f"(max rel. diff {e:.1e})")

    cf = ConsFV(g, S, V, dev, dt)
    u, v, w = swirl(g)
    F, rd, it = cf.project(u, v, w, 100.0)
    inflow = float(cf.win_speed.sum() * g.h[0] * g.h[2])
    door = float(-F[1][:, 0, :].sum() * g.h[0] * g.h[2])
    check("2 projection divergence-free", rd < 1e-7, f"(max rel. divergence {rd:.1e}, {it} CG iterations)")
    check("2 door outflow = window inflow", abs(door - inflow) / inflow < 1e-6,
          f"(in {inflow:.4f}, out {door:.4f} m^3/s; expected V*A = {(1.0 * 0.61 + 0.5 * 0.61) * 3.15:.4f})")

    gc = L2.Grid(0.25, [0.0] * 8)
    zc = np.zeros(gc.X.shape)
    u, v, w = swirl(gc)
    nutc = rng.uniform(0, 0.02, gc.X.shape) * gc.fluid
    snaps = [[u, v, w, nutc]] * 2
    cc = ConsFV(gc, zc, [0.0] * 8, dev, dt, doors_open=False)
    out, _, _ = cc.solve([0.0, 10.0], snaps, [60.0], c0=np.ones(gc.X.shape))
    dev3 = float(np.abs(out[60.0][gc.fluid] - 1.0).max())
    check("3 uniform CO2 stays uniform in a closed swirling room", dev3 < 1e-8,
          f"(max |c - 1| = {dev3:.1e}; projection divergence {max(i[1] for i in cc.projection_info):.1e})")
    Sc = seat_source(gc, 20.0)
    cc = ConsFV(gc, Sc, [0.0] * 8, dev, dt, doors_open=False)
    c0 = rng.uniform(0, 50, gc.X.shape) * gc.fluid
    out, _, _ = cc.solve([0.0, 10.0], snaps, [60.0], c0=c0)
    vol = gc.h[0] * gc.h[1] * gc.h[2]
    m_exp = (c0[gc.fluid].sum() + 60.0 * Sc[gc.fluid].sum()) * vol
    m_got = out[60.0][gc.fluid].sum() * vol
    check("4 closed room: CO2 mass = initial + injected", abs(m_got / m_exp - 1) < 1e-8, f"(ratio {m_got / m_exp:.10f})")

    u, v, w = swirl(g, 1.0)
    out, _, _ = ConsFV(g, S, V, dev, dt).solve([0.0, 10.0], [[z, z, z, nut], [u, v, w, nut]], [120.0])
    c = out[120.0][g.fluid]
    check("5 no negative CO2 (limiter)", c.min() >= -1e-6 * c.max(), f"(min {c.min():.2e}, max {c.max():.1f})")

    print("\nALL PASSED" if not FAILED else f"\nFAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
