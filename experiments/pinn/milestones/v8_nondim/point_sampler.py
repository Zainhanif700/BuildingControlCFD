# ============================================================================
# MILESTONE SNAPSHOT: v8_nondim (2026-09-25) -- frozen copy, DO NOT EDIT.
# Taken from git commit 2aa439e (the exact code the v8 run trained with);
# probe_co2_time.py / co2_residual_breakdown.py are the diagnostics written
# afterwards to analyse it. See README.md. Run scripts from INSIDE this folder
# so they import this frozen model code, not the live experiments/gnot/ one.
# ============================================================================

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

# CO2 emission per person (matches Alexander's config). Moved here from
# train_gnot.py (v8_nondim) so gnot_model.py can use it for output scaling
# without a circular import -- same "single source of truth" reasoning as
# CO2_SOURCE_SIGMA above.
EMISSION_PER_PERSON = 1.15e-4

# Window inflow ramp time constant (s): target inflow = V * tanh(3t/TAU_RAMP),
# fully open within ~2 s. Moved here from train_gnot.py (v8_nondim) because
# gnot_model.py now also uses it as an input feature -- see QueryEncoder.
TAU_RAMP = 2.0

# --- v8_nondim: reference scales for non-dimensionalization ---
# ROOT CAUSE FOUND (v8): every run v1-v6 had the CO2 residual loss sitting
# at ~3e-6 from iteration ~10 onward -- which is EXACTLY the loss a
# constant (trivial, C=0) CO2 field gives: mean(S^2) over our own sampling
# distribution = 3.08e-6 (computed numerically). The network never left the
# trivial solution. Two scaling causes:
#   (1) raw t (0-120 s) and raw N_people (0-50) fed straight into Linear ->
#       Tanh layers saturate most units (measured: ~75% at t=60, ~88% at
#       t=120; ~82% at N=25), so the network can barely represent CO2
#       growing over time or scaling with occupancy. Velocity is unaffected
#       because its inflow target saturates within ~2 s (tanh(3t/2)).
#   (2) CO2 residuals are O(S_REF) ~ 6e-3, i.e. squared ~1e-5, while
#       velocity residuals are O(1e-2..1e-1) -- a ~1e4 imbalance, which is
#       also why the adaptive CO2 weight was always pinned at its ceiling.
# Fix: standard non-dimensionalization, step 1 of Wang, Sankaran, Wang &
# Perdikaris 2023, "An Expert's Guide to Training Physics-informed Neural
# Networks" (arXiv:2308.08468): inputs and outputs scaled to O(1).
S_REF = N_PEOPLE_MAX * EMISSION_PER_PERSON  # max source strength, 5.75e-3 per s
C_REF = S_REF * T_MAX                        # upper bound on accumulated CO2, 0.69
# (closed room, no diffusion: C <= S_max * t_max). Order-of-magnitude scale
# only -- not a hard bound the network is clamped to.


def _rand(n, lo, hi, device):
    return torch.rand(n, 1, device=device) * (hi - lo) + lo


def _in_any_column(x, y):
    """Boolean mask: True where (x,y) falls inside one of the 4 columns."""
    inside = torch.zeros_like(x, dtype=torch.bool)
    for cx, cy, r, _, _ in COLUMNS:
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        inside = inside | (d2 <= r ** 2)
    return inside


# FIX #3 (from the original 3-step ranked plan -- "hard-constrain or oversample
# the closed-window zero-velocity case"): the closed-window diagnostic
# (all 8 windows at V=0) keeps showing non-trivial residual velocity (up to
# ~0.3-0.6 m/s, the same order of magnitude as real forced airflow) even
# though the TRUE physical solution for zero forcing + zero IC + no body
# force term is exactly u=v=w=0 everywhere -- a case the network should be
# able to learn trivially, but apparently doesn't generalize to correctly.
#
# ROOT CAUSE (distinct from, and more severe than, the CO2 spatial-sampling
# problem): V has NUM_WINDOWS=8 independent dimensions, each sampled
# uniformly in [V_MIN, V_MAX]. Under independent per-axis uniform sampling,
# the probability that ALL 8 happen to be simultaneously near zero for the
# same training point is roughly (epsilon/V_MAX)^8 for a small tolerance
# epsilon -- astronomically small. Each individual window being near zero is
# common on its own; the JOINT event "every window near zero at once" (what
# the diagnostic actually tests) is a curse-of-dimensionality corner of an
# 8-dimensional hypercube that plain independent uniform sampling almost
# never reaches. So the network has essentially never been trained on
# anything resembling the exact scenario the diagnostic evaluates.
#
# FIX: explicitly inject correlated all-closed and partially-closed scenarios
# into training (mirroring the source-concentrated spatial sampling fix --
# same idea, applied in "scenario space" instead of physical space), instead
# of relying on independent per-axis randomness to occasionally produce one.
CLOSED_SCENARIO_FRAC = 0.3  # fraction of points whose V is EXACTLY all-zero
# (the exact scenario the closed-window diagnostic tests)
PARTIAL_CLOSED_SCENARIO_FRAC = 0.2  # fraction where a random SUBSET of the 8
# windows is zeroed (each window independently closed with 50% probability) --
# covers the "some but not all windows closed" regime too, not just the two
# extremes of "all open" (implicitly covered by uniform sampling) and
# "all closed" (covered by CLOSED_SCENARIO_FRAC above).
# Remaining 1 - 0.3 - 0.2 = 50% of points still use fully independent uniform
# sampling, so general open-window training coverage is not reduced to zero.


def sample_scenario(n, device):
    """Random (t, V1..V8, N_people) -- the made-up scenario values swept during
    physics-only training, matching Alexander's 0-5 m/s / 0-50 people ranges.
    A fraction of V configurations are deliberately all-closed or
    partially-closed (see fix #3 comment above) instead of 100% independent
    uniform, so the network actually sees the closed-window regime often
    enough to learn it correctly."""
    t = _rand(n, T_MIN, T_MAX, device)
    N_people = _rand(n, N_PEOPLE_MIN, N_PEOPLE_MAX, device)

    n_closed = int(round(n * CLOSED_SCENARIO_FRAC))
    n_partial = int(round(n * PARTIAL_CLOSED_SCENARIO_FRAC))
    n_uniform = n - n_closed - n_partial

    V_parts = []
    if n_uniform > 0:
        V_parts.append(torch.rand(n_uniform, NUM_WINDOWS, device=device) * (V_MAX - V_MIN) + V_MIN)
    if n_closed > 0:
        V_parts.append(torch.zeros(n_closed, NUM_WINDOWS, device=device))
    if n_partial > 0:
        V_partial = torch.rand(n_partial, NUM_WINDOWS, device=device) * (V_MAX - V_MIN) + V_MIN
        closed_mask = torch.rand(n_partial, NUM_WINDOWS, device=device) < 0.5
        V_partial = V_partial * (~closed_mask).float()
        V_parts.append(V_partial)
    V = torch.cat(V_parts, dim=0)

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


# ---------------------------------------------------------------------------
# STAGE 1 of the self-adaptive weighting + sampling upgrade (Chen, Howard &
# Stinis, "Self-adaptive weighting and sampling for physics-informed neural
# networks," arXiv:2511.05452, 2025). Context: v6_lr_decay and
# v7_higher_co2_weight both worked with one GLOBAL scalar CO2 loss weight,
# which either had to be capped at a value we invented ourselves (no
# literature backing -- see train_gnot.py's CO2_WEIGHT_MAX comment) or left
# CO2 undertrained. The cited paper instead uses a PER-POINT weight,
# renormalized to mean=1 every update, so there's no ceiling to guess. But
# per-point weights only make sense if a point is actually revisited across
# iterations -- our original design resampled 100% of points fresh every
# single iteration, so there was nothing for a per-point weight to track.
#
# STAGE 1 (this change): switch to a PERSISTENT POOL of n points per
# (n, device), refreshing only a fraction of them periodically -- the cited
# paper's own tested defaults for its adaptive-sampling component
# (POOL_REFRESH_FRAC=0.2 of points, every POOL_REFRESH_EVERY=100 iterations).
# This stage deliberately does NOT add per-point adaptive WEIGHTING yet
# (planned as Stage 2, in train_gnot.py) -- the goal here is to validate in
# isolation that switching from full per-iteration resampling to a mostly-
# persistent pool doesn't itself regress training. This is a genuine risk
# specific to this project: the cited paper's own benchmarks are all
# non-parametric (one fixed PDE, one fixed set of boundary conditions), while
# this project's network must generalize across many different window-
# velocity/occupancy scenarios (see sample_scenario above) -- a concern the
# paper's own experiments never tested. Validate this stage's health (no
# regression vs. v5/v7's already-confirmed closed/open-window behavior)
# before adding Stage 2 on top.
USE_PERSISTENT_POOL = False  # v8_nondim: DISABLED -- the non-dimensionalization
# fix (see S_REF/C_REF above) is being tested as a SINGLE-VARIABLE change
# against v5's already-documented sampling behavior (100% fresh points every
# iteration). The Stage 1 pool code is kept intact for later use (Stage 2
# adaptive weighting would need it), just switched off. With this False,
# sample_interior() behaves exactly as it did in v5.
POOL_REFRESH_FRAC = 0.2    # fraction of the pool replaced at each refresh
POOL_REFRESH_EVERY = 100   # refresh cadence, in calls to sample_interior()
# (one call == one training iteration in train_gnot.py's main loop)

_interior_pools = {}  # keyed by (n, device_str) -> dict of tensors + "calls"


def _generate_interior_batch(n, device):
    """The actual point-generation logic (uniform + source-concentrated
    spatial mixture, fix #2; plus scenario sampling, fix #3) -- factored out
    of sample_interior() so both the initial pool build and each periodic
    partial refresh below can reuse it identically."""
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


def interior_pool_composition(n, device="cpu"):
    """DIAGNOSTIC ONLY -- read-only snapshot of the CURRENT persistent pool's
    scenario/spatial composition for (n, device), without calling
    sample_interior() (which would advance its call counter / potentially
    trigger a refresh as a side effect of merely inspecting it).

    WHY THIS EXISTS: an independent review of Stage 1 (persistent-pool
    sampling, see module comment above) flagged a real, previously
    unconsidered risk -- since points now persist for up to
    POOL_REFRESH_EVERY-1 iterations instead of being re-randomized every
    single iteration, a skewed random draw of scenario mixture (e.g. too many
    closed-window points, or too few near-source points) could persist for a
    long stretch instead of being averaged away immediately. That could
    introduce a NEW low-frequency oscillation source layered on top of the
    CO2 magnitude oscillation this project is already trying to diagnose --
    confounding the investigation instead of isolating the resampling
    change's own effect. This function lets a training script log the pool's
    actual composition over time so that risk is directly OBSERVED, not just
    hoped against.

    Returns None if no pool exists yet for this (n, device) (i.e.
    sample_interior/sample_ic hasn't been called with these args yet).
    """
    key = (n, str(device))
    pool = _interior_pools.get(key)
    if pool is None:
        return None

    x, y, z, V = pool["x"], pool["y"], pool["z"], pool["V"]
    dist = torch.sqrt((x - SOURCE_X) ** 2 + (y - SOURCE_Y) ** 2 + (z - BREATHING_HEIGHT) ** 2)
    return {
        "calls": pool["calls"],
        "frac_all_closed": (V == 0).all(dim=1).float().mean().item(),
        "frac_any_closed": (V == 0).any(dim=1).float().mean().item(),
        "frac_near_source": (dist < CO2_SOURCE_SIGMA).float().mean().item(),
        "mean_dist_to_source": dist.mean().item(),
    }


def reset_interior_pools():
    """Clears all persistent interior pools. Call this between independent
    runs/tests (e.g. staged_smoke_test.py stages) so leftover pool state
    from one doesn't leak into another."""
    _interior_pools.clear()


def sample_interior(n, device="cpu"):
    """Random points inside the room, excluding the 4 columns (rejection
    sampling). A fraction (SOURCE_SAMPLE_FRAC) is concentrated near the
    known CO2 source location instead of uniform -- see fix #2 comment
    above for why.

    STAGE 1 persistent-pool version (see module comment above): maintains a
    pool of exactly n points per (n, device) combination, refreshing only
    POOL_REFRESH_FRAC of them every POOL_REFRESH_EVERY calls instead of
    regenerating all n points every single call. The first call for a given
    (n, device) still builds a full fresh pool via _generate_interior_batch,
    so one-shot callers (tests, or training's very first iteration) see the
    same distribution as the pre-Stage-1 code.

    NOTE: sample_ic() below calls this function directly (reusing the same
    spatial mixture, just overriding t=0), so it automatically gets its own
    independent persistent pool too, keyed separately since POINTS_IC !=
    POINTS_INTERIOR in train_gnot.py.
    """
    if not USE_PERSISTENT_POOL:
        return _generate_interior_batch(n, device)  # v5 behavior: 100% fresh every call

    key = (n, str(device))
    pool = _interior_pools.get(key)

    if pool is None:
        x, y, z, t, V, N_people = _generate_interior_batch(n, device)
        pool = {"x": x, "y": y, "z": z, "t": t, "V": V, "N_people": N_people, "calls": 0}
        _interior_pools[key] = pool
    else:
        pool["calls"] += 1
        if pool["calls"] % POOL_REFRESH_EVERY == 0:
            n_refresh = int(round(n * POOL_REFRESH_FRAC))
            if n_refresh > 0:
                new_x, new_y, new_z, new_t, new_V, new_N = _generate_interior_batch(n_refresh, device)
                idx = torch.randperm(n, device=device)[:n_refresh]
                pool["x"][idx] = new_x
                pool["y"][idx] = new_y
                pool["z"][idx] = new_z
                pool["t"][idx] = new_t
                pool["V"][idx] = new_V
                pool["N_people"][idx] = new_N

    # Return fresh, DETACHED leaf tensors each call. Callers (physics_loss,
    # ic_loss, etc. in train_gnot.py) call .requires_grad_(True) on the
    # returned x/y/z/t every iteration. Returning the pool's own stored
    # tensors directly instead of a clone would (a) leave requires_grad=True
    # permanently attached to the pool's storage, which then makes the
    # in-place refresh assignment above ILLEGAL on the next refresh (PyTorch
    # forbids in-place ops on a leaf tensor that requires grad), and (b) risk
    # reusing a tensor still referenced by a previous iteration's autograd
    # graph. clone().detach() avoids both.
    return (pool["x"].clone().detach(), pool["y"].clone().detach(),
            pool["z"].clone().detach(), pool["t"].clone().detach(),
            pool["V"].clone().detach(), pool["N_people"].clone().detach())


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
