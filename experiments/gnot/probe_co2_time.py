"""
CO2 TIME-SERIES probe (v8 onward): is the network learning the right CO2
GROWTH but sitting on a wrong OFFSET?

Why: the interior CO2 equation only constrains how C CHANGES (dc/dt,
gradients, Laplacian) -- adding a constant offset to the whole field costs
nothing there. The absolute level is pinned only by c=0 at t=0 (initial
condition) and c=0 at the windows, both SOFT penalties. v8 at iter 5000 gave
C(source, t=60) ~ 0.02 vs the expected ~0.138, plus negative CO2 values --
consistent with "correct growth on top of a negative starting offset", but
also with "growth itself too slow". This script tells the two apart.

Scenario: all 8 windows closed (V=0), N_people=20, breathing height z=1.10 m.
Closed room, so no convection; diffusion only spreads CO2 by
sqrt(D*t) ~ 0.8 m over 120 s, small next to the source width sigma=2.5 m, so
the physically expected value is approximately
    C_expected(x, t) ~ N * E * t * exp(-dist^2 / sigma^2).

Usage:
    python3 probe_co2_time.py <checkpoint> [<checkpoint> ...]
"""
import sys
import numpy as np
import torch

from gnot_model import GNOTOperator, check_checkpoint_compat
from point_sampler import (NUM_WINDOWS, BREATHING_HEIGHT, ROOM_X, ROOM_Y, COLUMNS,
                           EMISSION_PER_PERSON, CO2_SOURCE_SIGMA)

N_PEOPLE = 20.0
TIMES = [0.0, 10.0, 30.0, 60.0, 90.0, 120.0]
SX, SY = (ROOM_X[0] + ROOM_X[1]) / 2, (ROOM_Y[0] + ROOM_Y[1]) / 2
PROBES = {  # name: (x, y) at breathing height
    "source (7.76,4.58)": (SX, SY),
    "2.5 m east of source": (SX + 2.5, SY),
    "far corner (1.0,1.0)": (1.0, 1.0),
}


def eval_C(model, x, y, t, device):
    """C at arrays x, y (breathing height), time t, closed windows, N=20."""
    n = len(x)
    xt = torch.tensor(np.asarray(x), dtype=torch.float32, device=device).view(-1, 1)
    yt = torch.tensor(np.asarray(y), dtype=torch.float32, device=device).view(-1, 1)
    zt = torch.full((n, 1), BREATHING_HEIGHT, device=device)
    tt = torch.full((n, 1), float(t), device=device)
    V = torch.zeros(n, NUM_WINDOWS, device=device)
    Np = torch.full((n, 1), N_PEOPLE, device=device)
    with torch.no_grad():
        _, _, _, C, _ = model(xt, yt, zt, tt, V, Np)
    return C.cpu().numpy().ravel()


def expected(x, y, t):
    d2 = (np.asarray(x) - SX) ** 2 + (np.asarray(y) - SY) ** 2
    return N_PEOPLE * EMISSION_PER_PERSON * t * np.exp(-d2 / CO2_SOURCE_SIGMA ** 2)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for path in sys.argv[1:]:
        ckpt = torch.load(path, map_location=device)
        check_checkpoint_compat(ckpt, path)
        model = GNOTOperator().to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"\n=== {path} (iter={ckpt.get('iter', '?')}) -- closed windows, N=20, z=1.10 m ===")

        # (1) time series at a few fixed points
        print(f"{'point':24s} " + " ".join(f"t={t:>4.0f}s" for t in TIMES))
        for name, (px, py) in PROBES.items():
            vals = [eval_C(model, [px], [py], t, device)[0] for t in TIMES]
            exps = [expected([px], [py], t)[0] for t in TIMES]
            print(f"{name:24s} " + " ".join(f"{v:+7.4f}" for v in vals) + "   <- predicted")
            print(f"{'':24s} " + " ".join(f"{e:+7.4f}" for e in exps) + "   <- expected")

        # (2) growth field C(60) - C(0) on a 40x40 grid: removes any constant
        # offset, so this shows whether the SHAPE and GROWTH are right
        xs = np.linspace(ROOM_X[0] + 0.1, ROOM_X[1] - 0.1, 40)
        ys = np.linspace(ROOM_Y[0] + 0.1, ROOM_Y[1] - 0.1, 40)
        Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
        xg, yg = Xg.ravel(), Yg.ravel()
        inside_col = np.zeros_like(xg, dtype=bool)
        for cx, cy, r, _, _ in COLUMNS:
            inside_col |= (xg - cx) ** 2 + (yg - cy) ** 2 <= r ** 2
        c0 = eval_C(model, xg, yg, 0.0, device)
        c60 = eval_C(model, xg, yg, 60.0, device)
        growth = c60 - c0
        growth[inside_col] = np.nan
        c0[inside_col] = np.nan
        G = growth.reshape(40, 40)
        pi, pj = np.unravel_index(np.nanargmax(G), G.shape)
        gmax, gmin = np.nanmax(G), np.nanmin(G)
        half = gmin + (gmax - gmin) / 2
        print(f"\nC at t=0 (initial condition says exactly 0 everywhere): "
              f"min {np.nanmin(c0):+.4f}  max {np.nanmax(c0):+.4f}  mean {np.nanmean(c0):+.4f}")
        print(f"growth C(60)-C(0): min {gmin:+.4f}  max {gmax:+.4f}  "
              f"(expected max ~{expected([SX], [SY], 60.0)[0]:.3f} at the source, ~0 far away)")
        print(f"growth peak at (x,y)=({xs[pi]:.2f}, {ys[pj]:.2f}); true source ({SX:.2f}, {SY:.2f})")
        # expected counts computed from the exact Gaussian on this same grid:
        # half-max full width = 2*sigma*sqrt(ln 2) = 4.16 m -> 18 cells along y
        # (0.23 m spacing), 10 cells along x (0.39 m spacing)
        print(f"growth above half-max: along y {np.sum(G[pi, :] > half)}/40, along x {np.sum(G[:, pj] > half)}/40 "
              f"(physically expected for sigma=2.5 m: 18/40 along y, 10/40 along x)")

    print("\nHOW TO READ THIS:\n"
          "  - C(t=0) clearly non-zero (e.g. -0.1) but growth ~ expected  -> OFFSET problem: the\n"
          "    initial condition is too weakly enforced. Fix: hard-enforce C(t=0)=0 in the model.\n"
          "  - C(t=0) ~ 0 but growth much smaller than expected            -> GROWTH problem: the\n"
          "    network isn't satisfying dc/dt = S near the source.\n"
          "  - growth peak near the source with sensible half-max counts   -> shape/localization OK.")


if __name__ == "__main__":
    main()
