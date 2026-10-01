"""
Tests of fv_turb.py that need no OpenFOAM run (seconds, CPU or GPU):
  1. constant D, nut = 0: TurbFV must reproduce TorchFV (the verified solver) to round-off;
  2. closed room, no flow, RANDOM variable D: total CO2 must equal the injected amount exactly
     (the flux form is conservative; no-flux walls);
  3. the seating source: total emission = N x PPM_M3S_PER_PERSON, only inside the seating box.
The physical cross-check against OpenFOAM's own turbulent CO2 transport is in compare_rans_pilot.py.
Usage (from experiments/gnot_openfoam_rans):  python3 test_fv_turb.py
"""
import sys

import numpy as np
import torch

import common
import check_co2_with_model_flow as L2
from fv_torch import TorchFV
from fv_turb import TurbFV, seat_source
from train_gnot import DIFFUSIVITY

FAILED = []


def check(name, ok, msg=""):
    print(f"{'PASS' if ok else 'FAIL'}: {name} {msg}")
    if not ok:
        FAILED.append(name)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float64
    rng = np.random.default_rng(0)
    V = [1.0, 0, 0, 0, 0, 0, 0, 0.5]
    g = L2.Grid(0.5, V)                      # coarse grid -> fast
    sh = g.X.shape
    # a smooth, arbitrary flow (not divergence-free -- only the discrete operators are compared)
    u = 0.3 * np.sin(g.Y / 3.0) * g.fluid
    v = -0.2 * np.cos(g.X / 4.0) * g.fluid
    w = 0.05 * np.sin(g.Z) * g.fluid
    snaps = [[u * 0, v * 0, w * 0], [u, v, w]]
    times = [0.0, 10.0]
    ref, _, _ = TorchFV(g, dev, dt).solve(times, snaps, [5.0, 20.0])
    D_const = DIFFUSIVITY
    tf = TurbFV(g, g.S, dev, dt, d_mol=D_const, sc_t=1.0)
    got, _, _ = tf.solve(times, [s + [np.zeros(sh)] for s in snaps], [5.0, 20.0])
    err = max(np.max(np.abs(got[t] - ref[t])) / np.max(np.abs(ref[t])) for t in ref)
    check("1 constant D reproduces fv_torch", err < 1e-12, f"(max rel. difference {err:.1e})")

    # 2. conservation with variable D, closed room, no flow
    gc = L2.Grid(0.5, [0.0] * 8)
    nut = rng.uniform(0.0, 0.05, gc.X.shape) * gc.fluid
    z = np.zeros(gc.X.shape)
    S = seat_source(gc, 20.0)
    tf = TurbFV(gc, S, dev, dt, d_mol=common.D_CO2, sc_t=common.SC_T)
    T_END = 60.0
    out, _, _ = tf.solve([0.0, 10.0], [[z, z, z, nut], [z, z, z, nut]], [T_END])
    cell_v = gc.h[0] * gc.h[1] * gc.h[2]
    m_got = out[T_END][gc.fluid].sum() * cell_v
    m_exp = 20.0 * common.PPM_M3S_PER_PERSON * T_END
    check("2 conservation with variable D (closed room)", abs(m_got / m_exp - 1) < 1e-10,
          f"(CO2 in room {m_got:.6g} vs injected {m_exp:.6g} ppm m^3)")
    check("2b variable D actually spreads CO2 differently", float(np.std(out[T_END][gc.fluid])) > 0)

    # 3. seating source
    m = S > 0
    (x0, x1), (y0, y1), (z0, z1) = common.SEAT_BOX
    inside = (gc.X[m] >= x0).all() and (gc.X[m] <= x1).all() and (gc.Z[m] >= z0).all() and (gc.Z[m] <= z1).all()
    tot = S.sum() * cell_v
    check("3 source total = N x emission, only in the seating box",
          inside and abs(tot / (20.0 * common.PPM_M3S_PER_PERSON) - 1) < 1e-12,
          f"({tot:.4g} ppm m^3/s for 20 people; {int(m.sum())} cells)")
    print("\nRESULT:", "ALL PASSED" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
