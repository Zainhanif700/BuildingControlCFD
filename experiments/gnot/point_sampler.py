"""
Point sampler for GNOT training on the REAL room geometry.

All numbers below were measured directly from the STL files in
experiments/gnot/geometry/ (see geometry/inspect_geometry.py and
geometry/floor_plan.py for how they were extracted and verified):

  Room box:      x in [0, 15.53] m, y in [0, 9.16] m, z in [0, 3.15] m
  2 doors:       on the y=0 wall (outlet, p=0), floor to 2.11m high
  8 windows:     on the y=9.16 wall (inflow, one V_k each), floor to ceiling
  4 columns:     free-standing cylinders, floor to ceiling, radius ~0.285m

This mirrors exactly what Alexander's train_parametric_multi_window_tanh.py
does (same room, same 8 independent window velocities V1..V8, same 4
columns excluded from the interior), just reimplemented here in PyTorch
for GNOT instead of PhysicsNeMo.
"""
import torch
import numpy as np

# ---------------------------------------------------------------------------
# Real, fixed room facts (measured from STL geometry -- never change these)
# ---------------------------------------------------------------------------
ROOM_X = (0.0, 15.53)
ROOM_Y = (0.0, 9.16)
ROOM_Z = (0.0, 3.15)

# CO2 source spatial spread + height (matches Alexander's config exactly --
# see train_gnot.py's physics constants). Lives here, not duplicated in
# gnot_model.py or train_gnot.py, since both already import from this file --
# a second hardcoded copy would risk silently drifting out of sync.
CO2_SOURCE_SIGMA = 2.5
BREATHING_HEIGHT = 1.10

# (x_lo, x_hi, z_lo, z_hi) on the y = ROOM_Y[0] wall
DOORS = [
    (1.59, 2.65, 0.0, 2.11),
    (12.32, 13.38, 0.0, 2.11),
]

# (x_lo, x_hi, z_lo, z_hi) on the y = ROOM_Y[1] wall -- one per V_k, k=0..7
WINDOWS = [
    (2.09, 2.70, 0.0, 3.15),
    (4.11, 4.72, 0.0, 3.15),
    (4.79, 5.40, 0.0, 3.15),
    (7.49, 8.10, 0.0, 3.15),
    (9.50, 10.11, 0.0, 3.15),
    (10.19, 10.80, 0.0, 3.15),
    (12.89, 13.50, 0.0, 3.15),
    (14.91, 15.52, 0.0, 3.15),
]
NUM_WINDOWS = len(WINDOWS)

# (cx, cy, radius, z_lo, z_hi) -- free-standing floor-to-ceiling columns
COLUMNS = [
    (13.85, 8.44, 0.285, 0.0, 3.15),
    (5.74, 0.45, 0.285, 0.0, 3.15),
    (5.78, 8.50, 0.285, 0.0, 3.15),
    (13.83, 0.35, 0.285, 0.0, 3.15),
]

# Scenario ranges we sweep during physics-only training (made-up per iteration,
# not real sensor data -- see V range = 0-5 m/s confirmed by Alexander)
V_MIN, V_MAX = 0.0, 5.0
N_PEOPLE_MIN, N_PEOPLE_MAX = 0.0, 50.0
T_MIN, T_MAX = 0.0, 120.0


def _rand(n, lo, hi, device):
    return torch.rand(n, 1, device=device) * (hi - lo) + lo


def _in_any_column(x, y):
    """Boolean mask: True where (x,y) falls inside one of the 4 columns."""
    inside = torch.zeros_like(x, dtype=torch.bool)
    for cx, cy, r, _, _ in COLUMNS:
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        inside = inside | (d2 <= r ** 2)
    return inside


def sample_scenario(n, device):
    """Random (t, V1..V8, N_people) -- the made-up scenario values swept during
    physics-only training, matching Alexander's 0-5 m/s / 0-50 people ranges."""
    t = _rand(n, T_MIN, T_MAX, device)
    V = torch.rand(n, NUM_WINDOWS, device=device) * (V_MAX - V_MIN) + V_MIN
    N_people = _rand(n, N_PEOPLE_MIN, N_PEOPLE_MAX, device)
    return t, V, N_people


# Source location (room center at breathing height) -- used below for
# source-concentrated sampling. Same formula gnot_model.py's QueryEncoder
# computes independently for its source_proximity feature; not consolidated
# into one import since both are just deriving from ROOM_X/ROOM_Y/
# BREATHING_HEIGHT (already the single source of truth), so there's no
# separate magic number here that could drift out of sync.
SOURCE_X = (ROOM_X[0] + ROOM_X[1]) / 2
SOURCE_Y = (ROOM_Y[0] + ROOM_Y[1]) / 2

# FIX #2 (from the literature-grounded ranked plan -- Nabian et al. 2021,
# "Efficient Training of PINNs via Importance Sampling"): with 100% uniform
# interior sampling, very few of the POINTS_INTERIOR=1000 points per
# iteration land near the small CO2 source (sigma=2.5m in a
# ~15.5x9.16x3.15m room). Away from the source the source term S(x,y,z,t) is
# essentially 0, so a trivial C~0 field already satisfies the PDE residual
# there -- meaning most of the training signal each iteration pushes toward
# "CO2 stays near zero everywhere," and only a small minority of points ever
# see the region where the source term actually matters.
#
# DIAGNOSED DIRECTLY (not just theorized): after fix #1 (isotropic Fourier
# features + source_proximity feature) plus correctly-working adaptive CO2
# weighting, a partial training run (3000, then 10000 iterations) still
# showed a CO2 field that stayed near-zero and slightly negative everywhere
# -- no real bump structure at all. This is consistent with the network
# simply not having seen enough near-source points yet, not with fix #1
# being wrong.
#
# FIX: mix in a fraction of interior points drawn from a Gaussian centered
# on the known source location instead of sampling 100% uniformly. This does
# NOT change what's being trained against -- still the exact same PDE
# residual, at whatever points get sampled -- it only changes WHERE points
# are concentrated, giving the network many more per-iteration chances to
# see the region the source term actually depends on.
#
# TUNING HISTORY (diagnosed directly via the closed-window diagnostic, not
# just theorized): a first version used SOURCE_SAMPLE_FRAC=0.4 with the
# concentrated points spread at std=CO2_SOURCE_SIGMA (2.5m, the source's own
# physical width). After a 10,000-iteration partial run, CO2 was finally
# POSITIVE (an improvement over fix #1 alone, which stayed near-zero/
# negative), but still showed a room-wide band (elevated across 100% of the
# x-range at one y-slice) and a very narrow dynamic range (~0.005-0.007) --
# barely any real bump structure. Root cause: sampling with std=sigma still
# spreads points over a wide ~2.5m-radius region -- most of them land
# somewhere in "the source matters a bit here" territory, but few land close
# enough to the actual peak to force the network to resolve its SHARP
# curvature there. Fix: sample more points (higher frac) AND with a TIGHTER
# spread (std=sigma/2), so a much larger share of the concentrated subset
# clusters close to the true peak instead of merely somewhere within a few
# sigma of it.
SOURCE_SAMPLE_FRAC = 0.6  # was 0.4 -- raised since 0.4 wasn't enough signal
# concentrated near the source to overcome the general-domain "C~0 nearly
# everywhere" pull; the remaining 40% still stays uniform so general-domain
# NS structure and the rest of the room keep reasonable coverage.
SOURCE_SAMPLE_XY_STD = CO2_SOURCE_SIGMA / 2.0  # was CO2_SOURCE_SIGMA (2.5) --
# halved so points cluster closer to the actual peak, not just somewhere
# within the broader region where the source term is merely non-negligible.

# FIX (found by an earlier audit): using a z-spread as large as
# CO2_SOURCE_SIGMA (or even SOURCE_SAMPLE_XY_STD) is a large fraction of the
# room's entire height (3.15m) -- too much of it would fall outside [0, 3.15]
# and get clamped exactly onto the floor or ceiling (a hard pile-up
# artifact, not a smooth distribution near breathing height). X and Y don't
# have this problem (room is 15.53m / 9.16m, both much larger than
# SOURCE_SAMPLE_XY_STD, so clamping there stays negligible). Use a separate,
# smaller z-spread instead, scaled to the room's actual height.
SOURCE_SAMPLE_Z_STD = min(SOURCE_SAMPLE_XY_STD, (ROOM_Z[1] - ROOM_Z[0]) / 4.0)


def _sample_near_source(n, device):
    """Points drawn from an isotropic-in-(x,y) Gaussian centered on the CO2
    source (z uses its own smaller spread -- see SOURCE_SAMPLE_Z_STD comment
    above), clamped to stay inside the room bounds and rejecting any that
    land inside a column (same rejection rule as the uniform sampler
    below)."""
    pts = []
    remaining = n
    while remaining > 0:
        batch = max(remaining * 2, 256)  # oversample since some get rejected
        x = torch.normal(SOURCE_X, SOURCE_SAMPLE_XY_STD, size=(batch, 1), device=device).clamp(*ROOM_X)
        y = torch.normal(SOURCE_Y, SOURCE_SAMPLE_XY_STD, size=(batch, 1), device=device).clamp(*ROOM_Y)
        z = torch.normal(BREATHING_HEIGHT, SOURCE_SAMPLE_Z_STD, size=(batch, 1), device=device).clamp(*ROOM_Z)
        bad = _in_any_column(x, y)
        keep = ~bad.squeeze(-1)
        x, y, z = x[keep], y[keep], z[keep]
        pts.append(torch.cat([x, y, z], dim=1))
        remaining -= x.shape[0]
    xyz = torch.cat(pts, dim=0)[:n]
    return xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]


def sample_interior(n, device="cpu"):
    """Random points inside the room, excluding the 4 columns (rejection
    sampling). A fraction (SOURCE_SAMPLE_FRAC) is concentrated near the
    known CO2 source location instead of uniform -- see fix #2 comment
    above for why."""
    n_source = int(round(n * SOURCE_SAMPLE_FRAC))
    n_uniform = n - n_source

    pts = []
    remaining = n_uniform
    while remaining > 0:
        batch = max(remaining * 2, 256)  # oversample since some get rejected
        x = _rand(batch, *ROOM_X, device)
        y = _rand(batch, *ROOM_Y, device)
        z = _rand(batch, *ROOM_Z, device)
        bad = _in_any_column(x, y)
        keep = ~bad.squeeze(-1)
        x, y, z = x[keep], y[keep], z[keep]
        pts.append(torch.cat([x, y, z], dim=1))
        remaining -= x.shape[0]
    xyz_uniform = torch.cat(pts, dim=0)[:n_uniform] if n_uniform > 0 else torch.empty(0, 3, device=device)

    if n_source > 0:
        xs, ys, zs = _sample_near_source(n_source, device)
        xyz_source = torch.cat([xs, ys, zs], dim=1)
        xyz = torch.cat([xyz_uniform, xyz_source], dim=0)
    else:
        xyz = xyz_uniform

    t, V, N_people = sample_scenario(n, device)
    return xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3], t, V, N_people


def sample_walls(n, device="cpu"):
    """No-slip points on the 6 room faces, EXCLUDING door/window cutouts.
    Splits n roughly evenly across the 6 faces, rejecting any point that
    falls inside a door or window opening on the y=0 / y=ROOM_Y[1] faces."""
    n_each = n // 6
    all_x, all_y, all_z = [], [], []

    def reject_openings(x, y, z, openings):
        # openings: list of (x_lo,x_hi,z_lo,z_hi) cutouts to reject
        bad = torch.zeros_like(x, dtype=torch.bool)
        for xlo, xhi, zlo, zhi in openings:
            in_open = (x >= xlo) & (x <= xhi) & (z >= zlo) & (z <= zhi)
            bad = bad | in_open
        keep = ~bad.squeeze(-1)
        return x[keep], y[keep], z[keep]

    # y = 0 face (has 2 door cutouts)
    x = _rand(n_each * 2, *ROOM_X, device)
    z = _rand(n_each * 2, *ROOM_Z, device)
    y = torch.full_like(x, ROOM_Y[0])
    x, y, z = reject_openings(x, y, z, DOORS)
    all_x.append(x[:n_each]); all_y.append(y[:n_each]); all_z.append(z[:n_each])

    # y = ROOM_Y[1] face (has 8 window cutouts)
    x = _rand(n_each * 3, *ROOM_X, device)
    z = _rand(n_each * 3, *ROOM_Z, device)
    y = torch.full_like(x, ROOM_Y[1])
    x, y, z = reject_openings(x, y, z, WINDOWS)
    all_x.append(x[:n_each]); all_y.append(y[:n_each]); all_z.append(z[:n_each])

    # x = 0 and x = ROOM_X[1] faces (no cutouts)
    for fixed_val in [ROOM_X[0], ROOM_X[1]]:
        y = _rand(n_each, *ROOM_Y, device)
        z = _rand(n_each, *ROOM_Z, device)
        x = torch.full_like(y, fixed_val)
        all_x.append(x); all_y.append(y); all_z.append(z)

    # z = 0 (floor) and z = ROOM_Z[1] (ceiling) faces (no cutouts)
    for fixed_val in [ROOM_Z[0], ROOM_Z[1]]:
        x = _rand(n_each, *ROOM_X, device)
        y = _rand(n_each, *ROOM_Y, device)
        z = torch.full_like(x, fixed_val)
        all_x.append(x); all_y.append(y); all_z.append(z)

    x = torch.cat(all_x, dim=0)
    y = torch.cat(all_y, dim=0)
    z = torch.cat(all_z, dim=0)
    t, V, N_people = sample_scenario(x.shape[0], device)
    return x, y, z, t, V, N_people


def sample_columns_surface(n_per_column, device="cpu"):
    """No-slip points on the 4 columns' curved (cylindrical) side surfaces.

    FIX (found by audit): the 4 columns are solid floor-to-ceiling pillars,
    so air must not flow through them -- but sample_interior() only ever
    EXCLUDES points from inside the columns, it never adds points ON their
    surface for a no-slip loss. Without this, nothing in training actually
    stops the network from predicting flow straight through a column.
    Parametrizes each cylinder's side wall by angle theta in [0, 2*pi) and
    height z in [z_lo, z_hi]."""
    all_x, all_y, all_z = [], [], []
    for cx, cy, r, zlo, zhi in COLUMNS:
        theta = _rand(n_per_column, 0.0, 2 * torch.pi, device)
        z = _rand(n_per_column, zlo, zhi, device)
        x = cx + r * torch.cos(theta)
        y = cy + r * torch.sin(theta)
        all_x.append(x); all_y.append(y); all_z.append(z)
    x = torch.cat(all_x, dim=0); y = torch.cat(all_y, dim=0); z = torch.cat(all_z, dim=0)
    t, V, N_people = sample_scenario(x.shape[0], device)
    return x, y, z, t, V, N_people


def sample_doors(n, device="cpu"):
    """Outlet (p=0) points, split across the 2 real doors."""
    n_each = n // len(DOORS)
    all_x, all_y, all_z = [], [], []
    for xlo, xhi, zlo, zhi in DOORS:
        x = _rand(n_each, xlo, xhi, device)
        z = _rand(n_each, zlo, zhi, device)
        y = torch.full_like(x, ROOM_Y[0])
        all_x.append(x); all_y.append(y); all_z.append(z)
    x = torch.cat(all_x, dim=0); y = torch.cat(all_y, dim=0); z = torch.cat(all_z, dim=0)
    t, V, N_people = sample_scenario(x.shape[0], device)
    return x, y, z, t, V, N_people


def sample_windows(n_per_window, device="cpu"):
    """Inflow points, split evenly across the 8 real windows. Each returned
    point also carries WHICH window index it belongs to (window_idx), so the
    training loop knows which of the 8 V_k values is its own inflow speed."""
    all_x, all_y, all_z, all_idx = [], [], [], []
    for k, (xlo, xhi, zlo, zhi) in enumerate(WINDOWS):
        x = _rand(n_per_window, xlo, xhi, device)
        z = _rand(n_per_window, zlo, zhi, device)
        y = torch.full_like(x, ROOM_Y[1])
        idx = torch.full_like(x, k, dtype=torch.long)
        all_x.append(x); all_y.append(y); all_z.append(z); all_idx.append(idx)
    x = torch.cat(all_x, dim=0); y = torch.cat(all_y, dim=0); z = torch.cat(all_z, dim=0)
    window_idx = torch.cat(all_idx, dim=0)
    t, V, N_people = sample_scenario(x.shape[0], device)
    return x, y, z, t, V, N_people, window_idx


def sample_ic(n, device="cpu"):
    """t=0, room at rest, everywhere inside the room (excluding columns)."""
    x, y, z, _, V, N_people = sample_interior(n, device)
    t = torch.zeros_like(x)
    return x, y, z, t, V, N_people


if __name__ == "__main__":
    # quick self-test
    device = "cpu"
    for name, fn, extra in [
        ("interior", sample_interior, (2000,)),
        ("walls", sample_walls, (1200,)),
        ("doors", sample_doors, (400,)),
        ("ic", sample_ic, (500,)),
    ]:
        out = fn(*extra, device=device)
        x, y, z = out[0], out[1], out[2]
        print(f"{name}: {x.shape[0]} pts | x=[{x.min():.2f},{x.max():.2f}] "
              f"y=[{y.min():.2f},{y.max():.2f}] z=[{z.min():.2f},{z.max():.2f}]")
    out = sample_windows(150, device=device)
    x, y, z = out[0], out[1], out[2]
    print(f"windows: {x.shape[0]} pts | x=[{x.min():.2f},{x.max():.2f}] "
          f"y=[{y.min():.2f},{y.max():.2f}] z=[{z.min():.2f},{z.max():.2f}]")
    print("\nAll samplers ran without errors.")
