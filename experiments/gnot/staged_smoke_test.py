"""
Staged smoke test for GNOT -- runs each layer of the pipeline in isolation,
from lowest-level to full training step, printing PASS/FAIL after each stage
and STOPPING at the first failure.

WHY THIS EXISTS: our last few bugs (backward-graph reuse, GPU OOM, two
separate CO2-weighting collapses) were only caught by running the FULL
30-iteration integrated smoke test and reading through the printed losses.
That works, but when something breaks you only know "somewhere in the whole
pipeline something is wrong" -- you still have to manually narrow it down.
This script narrows it down FOR you: each stage tests one specific piece
(the Fourier encoding, the query encoder, the full model forward, the
curl-trick divergence-free property, each individual loss term, then finally
one full combined training step), in the same order data actually flows
through the model. If stage 3 fails, you know immediately the bug is in the
model forward pass, not e.g. in a loss term you haven't reached yet.

Run on the SERVER (needs torch + CUDA):
    cd experiments/gnot
    python3 staged_smoke_test.py

Most stages use a TINY point count (16-64 points) purely for speed (stages 0b
and 5a call physics_loss at its full 1000 points) -- this is
about catching CRASHES / NaNs / shape bugs, not about training quality or
memory-ceiling behavior (the separate memory-sweep smoke test already covers
that, at the full POINTS_INTERIOR=1000 scale).
"""
import sys
import torch

# FIX (found by audit): PyTorch defaults to allow_tf32=True for float32 matmul
# on Ampere GPUs (e.g. the RTX A2000 this project trains on), which truncates
# matmul precision to ~10 mantissa bits. Chained through every Linear/attention
# matmul plus TWO rounds of second-order autograd (stage 4's divergence check),
# this can push a numerically-fine implementation's residual above a naive
# 1e-3 threshold -- a FALSE failure that looks like a broken curl trick but
# isn't. Disabling TF32 here trades a little speed for exact float32 semantics,
# which is what this script's tight numerical assertions actually assume.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

FAILED = False


def stage(name):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            global FAILED
            if FAILED:
                return None
            print(f"\n=== STAGE: {name} ===")
            try:
                result = fn(*args, **kwargs)
                print(f"PASS: {name}")
                return result
            except Exception as e:
                FAILED = True
                print(f"FAIL: {name}")
                print(f"  {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                return None
        return wrapper
    return decorator


def assert_finite(t, label):
    if torch.isnan(t).any():
        raise AssertionError(f"{label} contains NaN")
    if torch.isinf(t).any():
        raise AssertionError(f"{label} contains Inf")


@stage("0. sample_interior -- source-concentrated sampling: counts, bounds, no column overlap")
def test_sample_interior(device):
    from point_sampler import (
        sample_interior, SOURCE_X, SOURCE_Y, BREATHING_HEIGHT, CO2_SOURCE_SIGMA,
        SOURCE_SAMPLE_FRAC, ROOM_X, ROOM_Y, ROOM_Z, COLUMNS,
    )
    n = 1000
    x, y, z, t, V, N_people = sample_interior(n, device)
    for name, tensor in [("x", x), ("y", y), ("z", z), ("t", t), ("V", V), ("N_people", N_people)]:
        assert_finite(tensor, name)
    assert x.shape[0] == n, f"expected {n} points, got {x.shape[0]}"

    # every point must be within room bounds (the Gaussian branch rejects out-of-room
    # draws since v16 -- verify no bug slipped a point outside)
    assert (x >= ROOM_X[0]).all() and (x <= ROOM_X[1]).all(), "x out of room bounds"
    assert (y >= ROOM_Y[0]).all() and (y <= ROOM_Y[1]).all(), "y out of room bounds"
    assert (z >= ROOM_Z[0]).all() and (z <= ROOM_Z[1]).all(), "z out of room bounds"

    # no point should land inside a column (rejection sampling should have caught all of them)
    xn = x.detach().cpu().numpy().ravel()
    yn = y.detach().cpu().numpy().ravel()
    for cx, cy, r, _, _ in COLUMNS:
        inside = (xn - cx) ** 2 + (yn - cy) ** 2 <= r ** 2
        assert not inside.any(), f"{inside.sum()} points landed inside column at ({cx},{cy})"

    # sanity-check the mixture actually concentrates points near the source:
    # with SOURCE_SAMPLE_FRAC of points drawn from a Gaussian near the
    # source, the fraction of ALL n points within 1-sigma-ish of the source
    # should now be noticeably higher than pure uniform sampling would give
    # (a rough, not exact, check -- this isn't testing an exact probability,
    # just that the mixture is doing SOMETHING, not silently falling back to
    # pure uniform sampling due to a bug).
    dist = torch.sqrt((x - SOURCE_X) ** 2 + (y - SOURCE_Y) ** 2 + (z - BREATHING_HEIGHT) ** 2)
    frac_near = (dist < CO2_SOURCE_SIGMA).float().mean().item()
    print(f"  n_source={int(round(n * SOURCE_SAMPLE_FRAC))}, n_uniform={n - int(round(n * SOURCE_SAMPLE_FRAC))}, "
          f"fraction of all {n} points within ~1 sigma of source: {frac_near:.3f}")
    # rough uniform-sampling baseline: a sphere of radius sigma over the room's
    # volume (ignoring z-clipping/column effects, just an order-of-magnitude check)
    room_volume = (ROOM_X[1] - ROOM_X[0]) * (ROOM_Y[1] - ROOM_Y[0]) * (ROOM_Z[1] - ROOM_Z[0])
    sphere_volume = (4.0 / 3.0) * torch.pi * CO2_SOURCE_SIGMA ** 3
    uniform_baseline = min(1.0, sphere_volume / room_volume)
    if SOURCE_SAMPLE_FRAC > 0:
        assert frac_near > uniform_baseline, (
            f"fraction near source ({frac_near:.3f}) is not higher than the pure-uniform baseline "
            f"({uniform_baseline:.3f}) -- source-concentrated sampling may not be working"
        )
    else:
        # v14 uniform sampling: numpy estimate for truly uniform points is 0.116 (the
        # sphere/room ratio above ignores floor/ceiling clipping, so it is higher)
        assert 0.07 < frac_near < 0.16, (
            f"fraction near source {frac_near:.3f} is not the uniform-sampling value ~0.116 -- "
            f"sampling is not uniform")

    # sanity-check fix #3 (closed/partial-closed window-scenario oversampling):
    # verify a meaningful fraction of V rows are EXACTLY all-zero (the all-closed
    # scenario), not just occasionally-small from ordinary uniform sampling. The
    # old pure-uniform scheme would give ~1e-16 probability of landing on exact
    # all-zero, so seeing a near-CLOSED_SCENARIO_FRAC fraction here directly
    # confirms the fix is wired up, not silently bypassed.
    from point_sampler import CLOSED_SCENARIO_FRAC
    all_zero_frac = (V == 0).all(dim=1).float().mean().item()
    print(f"  fraction of {n} points with ALL windows exactly V=0: {all_zero_frac:.3f} "
          f"(expected close to CLOSED_SCENARIO_FRAC={CLOSED_SCENARIO_FRAC})")
    assert all_zero_frac > CLOSED_SCENARIO_FRAC * 0.5, (
        f"only {all_zero_frac:.3f} of points have all-zero V, expected close to "
        f"{CLOSED_SCENARIO_FRAC} -- closed-scenario oversampling may not be working"
    )

    # STAGE 1 persistent-pool check (see point_sampler.py's module comment on
    # POOL_REFRESH_FRAC/POOL_REFRESH_EVERY): verify the pool actually
    # PERSISTS most points between consecutive calls (not silently still
    # regenerating 100% fresh every call, which would defeat the whole point
    # of Stage 1 -- per-point weights, planned for Stage 2, need a point to
    # actually be revisited to track anything), AND verify it DOES refresh a
    # fraction of points once POOL_REFRESH_EVERY calls have passed (not
    # silently frozen forever, which would hurt the operator's generalization
    # across scenarios -- the concern flagged in point_sampler.py).
    from point_sampler import (reset_interior_pools, POOL_REFRESH_FRAC, POOL_REFRESH_EVERY,
                               USE_PERSISTENT_POOL)
    if not USE_PERSISTENT_POOL:
        # v8_nondim switches the pool off to keep v8 a single-variable test.
        # Instead, confirm sampling really is 100% fresh every call (v5 behavior).
        xa, _, _, _, _, _ = sample_interior(200, device)
        xb, _, _, _, _, _ = sample_interior(200, device)
        same = torch.isclose(xa, xb).float().mean().item()
        print(f"  persistent pool is OFF (USE_PERSISTENT_POOL=False): {same:.3f} of points "
              f"identical across two calls (expected ~0, i.e. fully fresh sampling)")
        assert same < 0.05, "pool is supposed to be off, but points are being reused across calls"
        return
    reset_interior_pools()
    n_pool_test = 200  # smaller n than 1000 purely for speed; POOL_REFRESH_EVERY
    # is a call-count, not point-count, so behavior is identical at any n
    x0, y0, z0, _, _, _ = sample_interior(n_pool_test, device)
    x1, y1, z1, _, _, _ = sample_interior(n_pool_test, device)
    unchanged = torch.isclose(x0, x1).float().mean().item()
    assert unchanged > 0.95, (
        f"only {unchanged:.3f} of points stayed identical between two consecutive "
        f"calls (expected ~1.0, no refresh due yet) -- persistent pool may not be "
        f"working, still resampling 100% fresh every call"
    )
    # Refresh happens when pool["calls"] (incremented on every call AFTER the
    # first, which only creates the pool) hits a multiple of
    # POOL_REFRESH_EVERY -- i.e. on the (POOL_REFRESH_EVERY+1)-th call overall
    # (call 1 creates with calls=0; call 2 -> calls=1; ...; call
    # POOL_REFRESH_EVERY+1 -> calls=POOL_REFRESH_EVERY, refresh fires).
    # x0/x1 above were calls #1 and #2, so POOL_REFRESH_EVERY-3 more calls
    # land us at call #(POOL_REFRESH_EVERY-1); capturing the NEXT call gives
    # call #POOL_REFRESH_EVERY (still no refresh), and the one after that is
    # call #(POOL_REFRESH_EVERY+1) (the refresh itself). Verified numerically
    # via a standalone script before writing this, not just reasoned about.
    for _ in range(POOL_REFRESH_EVERY - 3):
        sample_interior(n_pool_test, device)
    x_before, _, _, _, _, _ = sample_interior(n_pool_test, device)  # call #POOL_REFRESH_EVERY, no refresh yet
    x_after, _, _, _, _, _ = sample_interior(n_pool_test, device)   # call #(POOL_REFRESH_EVERY+1), refresh fires here
    frac_changed = (~torch.isclose(x_before, x_after)).float().mean().item()
    print(f"  pool persistence check: {unchanged:.3f} unchanged across 2 immediate "
          f"calls; {frac_changed:.3f} of points changed at the refresh boundary "
          f"(expected close to POOL_REFRESH_FRAC={POOL_REFRESH_FRAC})")
    assert frac_changed > POOL_REFRESH_FRAC * 0.5, (
        f"only {frac_changed:.3f} of points changed at the refresh boundary, expected "
        f"close to {POOL_REFRESH_FRAC} -- periodic refresh may not be working"
    )
    assert frac_changed < POOL_REFRESH_FRAC * 2.0, (
        f"{frac_changed:.3f} of points changed at the refresh boundary, way more than "
        f"the expected {POOL_REFRESH_FRAC} -- pool may be refreshing far too aggressively"
    )
    reset_interior_pools()  # leave a clean slate for the rest of this test run


@stage("0c. v16 scenario/location independence -- every window, wall face, column and interior region sees all scenario types")
def test_scenario_location_independence(device):
    """Guards the v16 bug fix (point_sampler.sample_scenario shuffles its rows). Before it,
    windows 5-6 only ever saw all-closed scenarios, the x-max wall / floor / column 3 never
    saw flow, and far-field interior points never saw closed windows."""
    import numpy as np
    from point_sampler import (sample_windows, sample_walls, sample_columns_surface, sample_doors,
                               sample_interior, CLOSED_SCENARIO_FRAC, SOURCE_X, SOURCE_Y,
                               BREATHING_HEIGHT, CO2_SOURCE_SIGMA, ROOM_X, ROOM_Z, COLUMNS, NUM_WINDOWS)
    torch.manual_seed(0)
    lo, hi = CLOSED_SCENARIO_FRAC - 0.12, CLOSED_SCENARIO_FRAC + 0.12
    groups = []
    x, y, z, t, V, N, idx = sample_windows(400, device)
    closed_all = (V == 0).all(dim=1).cpu().numpy()
    own_open = (V.gather(1, idx) > 0).squeeze(1).cpu().numpy()
    idx = idx.squeeze(1).cpu().numpy()
    for k in range(NUM_WINDOWS):
        groups.append((f"window {k + 1}", closed_all[idx == k]))
        assert own_open[idx == k].mean() > 0.4, (
            f"window {k + 1}: only {own_open[idx == k].mean():.2f} of its points have it OPEN -- "
            f"it is barely trained as an inflow")
    x, y, z, t, V, N = sample_walls(3000, device)
    ca = (V == 0).all(dim=1).cpu().numpy()
    xn, zn = x.squeeze(1).cpu().numpy(), z.squeeze(1).cpu().numpy()
    groups += [("wall x=max", ca[xn == ROOM_X[1]]), ("wall x=0", ca[xn == ROOM_X[0]]),
               ("floor", ca[zn == ROOM_Z[0]]), ("ceiling", ca[zn == ROOM_Z[1]])]
    x, y, z, t, V, N = sample_columns_surface(400, device)
    ca = (V == 0).all(dim=1).cpu().numpy()
    xn, yn = x.squeeze(1).cpu().numpy(), y.squeeze(1).cpu().numpy()
    for k, (cx, cy, r, _, _) in enumerate(COLUMNS):
        groups.append((f"column {k + 1}", ca[np.abs(np.hypot(xn - cx, yn - cy) - r) < 1e-3]))
    x, y, z, t, V, N = sample_doors(800, device)
    ca = (V == 0).all(dim=1).cpu().numpy()
    xn = x.squeeze(1).cpu().numpy()
    groups += [("door 1", ca[xn < ROOM_X[1] / 2]), ("door 2", ca[xn >= ROOM_X[1] / 2])]
    x, y, z, t, V, N = sample_interior(4000, device)
    ca = (V == 0).all(dim=1).cpu().numpy()
    d = torch.sqrt((x - SOURCE_X) ** 2 + (y - SOURCE_Y) ** 2 + (z - BREATHING_HEIGHT) ** 2).squeeze(1).cpu().numpy()
    groups += [("interior near source", ca[d < CO2_SOURCE_SIGMA]), ("interior far field", ca[d >= CO2_SOURCE_SIGMA])]
    bad = []
    for name, g in groups:
        f = g.mean() if len(g) else float("nan")
        if not (len(g) >= 30 and lo <= f <= hi):
            bad.append(f"{name}: {f:.2f} (n={len(g)})")
    print("  all-closed share per location: " + ", ".join(f"{n} {g.mean():.2f}" for n, g in groups))
    assert not bad, (f"all-closed share should be ~{CLOSED_SCENARIO_FRAC} everywhere, but: " + "; ".join(bad))


@stage("0d. v16 audit fixes -- no floor pile-up, area-proportional walls, uniform IC, closed-window CO2, no pressure IC")
def test_audit_fixes(device):
    import math
    import numpy as np
    from point_sampler import (sample_interior, sample_walls, sample_ic, sample_windows, ROOM_X, ROOM_Y,
                               ROOM_Z, COLUMNS, DOORS, WINDOWS, SOURCE_X, SOURCE_Y, BREATHING_HEIGHT,
                               CO2_SOURCE_SIGMA, C_REF, TAU_RAMP)
    from train_gnot import windows_loss, ic_loss, POINTS_WINDOWS_PER
    torch.manual_seed(0)
    # (c) no source-sampling pile-up on floor/ceiling (was ~5% of interior points at z = 0)
    x, y, z, t, V, N = sample_interior(4000, device)
    on_bound = ((z == ROOM_Z[0]) | (z == ROOM_Z[1])).float().mean().item()
    assert on_bound < 1e-3, f"{on_bound:.3f} of interior points sit exactly on floor/ceiling (clamping?)"
    # (d) walls: counts proportional to net face area; nothing inside column footprints
    n = 6000
    x, y, z, t, V, N = sample_walls(n, device)
    x, y, z = (a.squeeze(1).cpu().numpy() for a in (x, y, z))
    Lx, Ly, Lz = ROOM_X[1] - ROOM_X[0], ROOM_Y[1] - ROOM_Y[0], ROOM_Z[1] - ROOM_Z[0]
    col = sum(math.pi * r ** 2 for _, _, r, _, _ in COLUMNS)
    faces = {"y=0": ((y == ROOM_Y[0]), Lx * Lz - sum((a1 - a0) * (b1 - b0) for a0, a1, b0, b1 in DOORS)),
             "y=max": ((y == ROOM_Y[1]), Lx * Lz - sum((a1 - a0) * (b1 - b0) for a0, a1, b0, b1 in WINDOWS)),
             "x=0": ((x == ROOM_X[0]), Ly * Lz), "x=max": ((x == ROOM_X[1]), Ly * Lz),
             "floor": ((z == ROOM_Z[0]), Lx * Ly - col), "ceiling": ((z == ROOM_Z[1]), Lx * Ly - col)}
    total = sum(a for _, a in faces.values())
    for name, (mask, area) in faces.items():
        assert abs(mask.sum() - n * area / total) <= 2, (
            f"wall face {name}: {mask.sum()} points, expected {n * area / total:.0f} (area-proportional)")
    fc = (z == ROOM_Z[0]) | (z == ROOM_Z[1])
    for cx, cy, r, _, _ in COLUMNS:
        assert not (fc & ((x - cx) ** 2 + (y - cy) ** 2 <= r ** 2)).any(), "floor/ceiling point inside a column"
    # (d) IC points uniform (near-source fraction ~0.116 for uniform points, ~0.45 if source-concentrated)
    x, y, z, t, V, N = sample_ic(4000, device)
    near = (torch.sqrt((x - SOURCE_X) ** 2 + (y - SOURCE_Y) ** 2 + (z - BREATHING_HEIGHT) ** 2)
            < CO2_SOURCE_SIGMA).float().mean().item()
    assert 0.07 < near < 0.16 and (t == 0).all(), f"IC points not uniform at t=0 (near-source share {near:.3f})"

    # (a)/(b) loss formulas, with a stub model: zero velocity, C = C_REF everywhere, p = 1
    class Stub(torch.nn.Module):
        def forward(self, x, y, z, t, V, N):
            zero = 0.0 * (x + y + z)
            return zero, zero, zero, C_REF + zero, 1.0 + zero

        def velocity_from_potential(self, A1, A2, A3, x, y, z):
            return 0.0 * x, 0.0 * y, 0.0 * z
    stub = Stub()
    torch.manual_seed(123)
    L = windows_loss(stub, "cpu", 1.0).item()
    torch.manual_seed(123)                                   # same draw as inside windows_loss
    xw, yw, zw, tw, Vw, Nw, idx = sample_windows(POINTS_WINDOWS_PER, "cpu")
    Vk = Vw.gather(1, idx)
    target = Vk * torch.tanh(3.0 * tw / TAU_RAMP)
    from train_gnot import V_REL_FLOOR, USE_WINDOW_NORMAL_TARGET
    if USE_WINDOW_NORMAL_TARGET:
        vel = (target ** 2 / target.abs().clamp_min(V_REL_FLOOR) ** 2).mean().item()   # v17: relative error
    else:
        vel = 0.0   # v19: no normal-velocity target (flux exact); stub has u = w = 0
    frac_open = (Vk > 0).float().mean().item()
    assert abs(L - (vel + frac_open)) < 1e-4, (
        f"windows_loss {L:.5f} != relative velocity term {vel:.5f} + open share {frac_open:.3f}: CO2 c=0 must "
        f"apply only at OPEN windows (old behaviour would give {vel + 1:.5f})")
    Lic = ic_loss(stub, "cpu", 1.0).item()
    assert abs(Lic - 1.0) < 1e-5, f"ic_loss {Lic:.5f} != 1 (CO2 term only): a pressure term is still in the IC"
    print(f"  floor/ceiling pile-up {on_bound:.4f}; wall counts area-proportional; IC near-source share {near:.3f}; "
          f"windows_loss CO2 term on {frac_open:.2f} of window points (open only); ic_loss has no pressure term")


@stage("0e. v17 flux-scaled wall BC -- leak term = (A_SOLID * u_n / Q_scale)^2 on walls and columns")
def test_flux_scaled_walls(device):
    """Stub model with a CONSTANT velocity (a, b, c): walls_loss must equal the no-slip part
    a^2+b^2+c^2 (planar) + a^2+b^2+c^2 (columns) plus mean((A_SOLID*u_n/Q)^2) with u_n = a / b / c on
    x / y / z faces and a*nx + b*ny on columns. Also checks A_SOLID and throughflow_scale."""
    from point_sampler import sample_walls, sample_columns_surface, ROOM_X, ROOM_Y, COLUMNS, WINDOWS
    from train_gnot import (walls_loss, throughflow_scale, A_SOLID, Q_FLOOR, POINTS_WALLS, POINTS_COLUMNS_PER)
    assert 430 < A_SOLID < 450, f"A_SOLID = {A_SOLID:.1f} m^2, expected ~441"
    V = torch.tensor([[3.0] + [0.0] * 7, [0.0] * 8, [5.0] * 8])
    t = torch.tensor([[60.0], [60.0], [0.0]])
    q = throughflow_scale(V, t).squeeze(1).tolist()
    areas = [(w[1] - w[0]) * (w[3] - w[2]) for w in WINDOWS]
    expect_q = [3.0 * areas[0], Q_FLOOR, 5.0 * sum(areas)]   # time-independent (no ramp), floor only if ~closed
    assert all(abs(a - b) < 1e-3 for a, b in zip(q, expect_q)), f"throughflow_scale {q} != {expect_q}"
    a, b, c = 0.03, -0.02, 0.01

    class Stub(torch.nn.Module):
        def forward(self, x, y, z, t, V, N):
            zero = 0.0 * (x + y + z)
            return zero, zero, zero, zero, zero

        def velocity_from_potential(self, A1, A2, A3, x, y, z):
            return a + 0.0 * x, b + 0.0 * y, c + 0.0 * z
    torch.manual_seed(7)
    L = walls_loss(Stub(), "cpu").item()
    torch.manual_seed(7)
    x, y, z, tw, Vw, _ = sample_walls(POINTS_WALLS, "cpu")
    xc, yc, zc, tc, Vc, _ = sample_columns_surface(POINTS_COLUMNS_PER, "cpu")
    on_x = (x == ROOM_X[0]) | (x == ROOM_X[1])
    on_y = (y == ROOM_Y[0]) | (y == ROOM_Y[1])
    un = torch.where(on_x, torch.full_like(x, a), torch.where(on_y, torch.full_like(x, b), torch.full_like(x, c)))
    cols = []
    for xi, yi in zip(xc.squeeze(1).tolist(), yc.squeeze(1).tolist()):
        cx, cy, r, _, _ = min(COLUMNS, key=lambda C: (xi - C[0]) ** 2 + (yi - C[1]) ** 2)
        cols.append(a * (xi - cx) / r + b * (yi - cy) / r)
    unc = torch.tensor(cols).view(-1, 1)
    flux = torch.cat([A_SOLID * un / throughflow_scale(Vw, tw), A_SOLID * unc / throughflow_scale(Vc, tc)])
    from train_gnot import velocity_scale, USE_WINDOW_NORMAL_TARGET
    s2 = a * a + b * b + c * c
    slip = (s2 / velocity_scale(Vw) ** 2).mean().item() + (s2 / velocity_scale(Vc) ** 2).mean().item()  # v19: relative
    expected = slip + (flux ** 2).mean().item()
    assert abs(L - expected) < 1e-4 * max(1.0, expected), f"walls_loss {L:.6f} != expected {expected:.6f}"
    # warm-up factor 0 must remove the leak term entirely (only the no-slip part remains)
    torch.manual_seed(7)
    L0 = walls_loss(Stub(), "cpu", flux_weight=0.0).item()
    assert abs(L0 - slip) < 1e-5 * max(1.0, slip), f"walls_loss with flux_weight=0 is {L0}, not the no-slip part {slip}"
    # windows_loss velocity part (co2_weight 0)
    from train_gnot import windows_loss, POINTS_WINDOWS_PER
    from point_sampler import sample_windows, TAU_RAMP
    torch.manual_seed(11)
    Lw = windows_loss(Stub(), "cpu", 0.0, rel_weight=0.0).item()
    torch.manual_seed(11)
    _, _, _, tw_, Vw_, _, idx_ = sample_windows(POINTS_WINDOWS_PER, "cpu")
    if USE_WINDOW_NORMAL_TARGET:   # rel_weight = 0 -> the old absolute error
        tgt = -Vw_.gather(1, idx_) * torch.tanh(3.0 * tw_ / TAU_RAMP)
        expect_w = (a * a) + ((b - tgt) ** 2).mean().item() + (c * c)
    else:                          # v19: tangential components only, relative to U_ref
        expect_w = ((a * a + c * c) / velocity_scale(Vw_) ** 2).mean().item()
    assert abs(Lw - expect_w) < 1e-4 * max(1.0, expect_w), f"windows_loss {Lw:.6f} != expected {expect_w:.6f}"
    print(f"  A_SOLID={A_SOLID:.1f} m^2; walls_loss for a constant 0.01-0.03 m/s velocity = {L:.3f} "
          f"(no-slip part only would be {slip:.5f}) -- leak is expensive; window/slip terms relative to U_ref")


@stage("0f. v19 exact through-flow -- no normal velocity on ANY solid surface, exact window/door fluxes, alpha split")
def test_throughflow_exact(device):
    """Random-init REAL model: velocity = curl(B_p + s*phi*A). Normal velocity must vanish on walls,
    floor, ceiling, columns, closed windows and the wall strips above the doors (up to float32
    round-off), each open window must deliver exactly V_k*A_k (quadrature), and the doors must
    release exactly the total inflow, split alpha : 1-alpha."""
    import math
    from gnot_model import GNOTOperator
    from train_gnot import get_velocity_and_derivs
    from point_sampler import (sample_walls, sample_columns_surface, WINDOWS, DOORS, COLUMNS,
                               ROOM_X, ROOM_Y, ROOM_Z, NUM_WINDOWS)
    torch.manual_seed(3)
    model = GNOTOperator().to(device)
    with torch.no_grad():      # review: at init alpha = 0.5 exactly, which would hide a door-order swap
        model.alpha_head[2].bias.fill_(1.5)                                          # -> alpha ~ 0.82
    Vrow = torch.tensor([[2.5, 0.0, 4.0, 0.0, 1.0, 0.0, 0.0, 5.0]], device=device)   # mixed open/closed
    t60 = 60.0

    def vel(x, y, z, V=Vrow, t=t60):
        n = x.shape[0]
        x, y, z = (a.detach().clone().requires_grad_(True) for a in (x, y, z))
        u, v, w, _, _ = get_velocity_and_derivs(model, x, y, z, torch.full((n, 1), t, device=device),
                                                V.expand(n, -1), torch.full((n, 1), 20.0, device=device))
        return u.detach(), v.detach(), w.detach()

    # interior speed scale
    g = torch.Generator().manual_seed(0)
    xi = (torch.rand(2000, 1, generator=g) * (ROOM_X[1] - 2) + 1).to(device)
    yi = (torch.rand(2000, 1, generator=g) * (ROOM_Y[1] - 2) + 1).to(device)
    zi = (torch.rand(2000, 1, generator=g) * (ROOM_Z[1] - 0.6) + 0.3).to(device)
    u, v, w = vel(xi, yi, zi)
    scale = torch.sqrt(u ** 2 + v ** 2 + w ** 2).mean().item()
    assert scale > 0.05, f"interior speed {scale:.3e} -- the through-flow is missing"

    worst = {}
    x, y, z, _, _, _ = sample_walls(3000, device)                   # planar walls without openings
    u, v, w = vel(x, y, z)
    on_x = (x == ROOM_X[0]) | (x == ROOM_X[1])
    on_y = (y == ROOM_Y[0]) | (y == ROOM_Y[1])
    un = torch.where(on_x, u, torch.where(on_y, v, w))
    worst["walls/floor/ceiling"] = un.abs().max().item()
    xc, yc, zc, _, _, _ = sample_columns_surface(300, device)       # columns
    u, v, w = vel(xc, yc, zc)
    k = torch.stack([(xc - cx) ** 2 + (yc - cy) ** 2 for cx, cy, _, _, _ in COLUMNS], 0).argmin(0)
    ctr = torch.tensor([[cx, cy, r] for cx, cy, r, _, _ in COLUMNS], device=device)[k.squeeze(-1)]
    nx, ny = (xc.squeeze(-1) - ctr[:, 0]) / ctr[:, 2], (yc.squeeze(-1) - ctr[:, 1]) / ctr[:, 2]
    worst["columns"] = (u.squeeze(-1) * nx + v.squeeze(-1) * ny).abs().max().item()
    pts = []                                                         # closed windows + strips above doors
    for kk, (a, b, c, d) in enumerate(WINDOWS):
        if Vrow[0, kk] == 0:
            pts.append((torch.rand(300, 1, device=device) * (b - a) + a, torch.full((300, 1), ROOM_Y[1], device=device),
                        torch.rand(300, 1, device=device) * (d - c) + c))
    for a, b, c, d in DOORS:
        pts.append((torch.rand(300, 1, device=device) * (b - a) + a, torch.full((300, 1), ROOM_Y[0], device=device),
                    torch.rand(300, 1, device=device) * (ROOM_Z[1] - d) + d))
    x, y, z = (torch.cat([p[i] for p in pts]) for i in range(3))
    _, v, _ = vel(x, y, z)
    worst["closed windows + above doors"] = v.abs().max().item()
    for name, val in worst.items():
        assert val < 1e-3 * scale, f"normal velocity {val:.2e} m/s on {name} (interior speed {scale:.2f}) -- LEAK"

    def flux(a, b, c, d, yval, m1=60, m2=30):                        # outward flux, midpoint rule
        xs = a + (torch.arange(m1, device=device) + 0.5) * (b - a) / m1
        zs = c + (torch.arange(m2, device=device) + 0.5) * (d - c) / m2
        X, Z = torch.meshgrid(xs, zs, indexing="ij")
        X, Z = X.reshape(-1, 1), Z.reshape(-1, 1)
        _, v, _ = vel(X, torch.full_like(X, yval), Z)
        sign = 1.0 if yval == ROOM_Y[1] else -1.0
        return sign * v.mean().item() * (b - a) * (d - c)
    ramp = math.tanh(3.0 * t60 / 2.0)
    for kk, (a, b, c, d) in enumerate(WINDOWS):
        target = -Vrow[0, kk].item() * ramp * (b - a) * (d - c)
        got = flux(a, b, c, d, ROOM_Y[1])
        assert abs(got - target) < 0.01 * max(1.0, abs(target)), f"window {kk + 1}: flux {got:.4f} != {target:.4f}"
    q_in = sum(Vrow[0, kk].item() * ramp * (b - a) * (d - c) for kk, (a, b, c, d) in enumerate(WINDOWS))
    fd = [flux(a, b, c, d, ROOM_Y[0], 60, 60) for a, b, c, d in DOORS]
    assert abs(sum(fd) - q_in) < 0.01 * q_in, f"door outflow {sum(fd):.4f} != inflow {q_in:.4f}"
    alpha = model.door_split(torch.full((1, 1), t60, device=device), Vrow).item()
    assert abs(fd[0] / sum(fd) - alpha) < 0.01, f"door split {fd[0] / sum(fd):.3f} != alpha {alpha:.3f}"
    print(f"  interior speed {scale:.2f} m/s; max normal velocity: " +
          ", ".join(f"{k_} {v_:.1e}" for k_, v_ in worst.items()) +
          f"; inflow {q_in:.3f} = door outflow {sum(fd):.3f} m^3/s, split {fd[0] / sum(fd):.3f} (alpha {alpha:.3f})")


@stage("0b. v8 non-dimensionalization -- no input saturation, training CO2 loss actually scaled, checkpoint guard")
def test_nondim(device):
    from gnot_model import GNOTOperator, check_checkpoint_compat
    from point_sampler import NUM_WINDOWS, T_MAX, N_PEOPLE_MAX, V_MAX, ROOM_X, ROOM_Y, ROOM_Z
    from train_gnot import trivial_co2_floor
    torch.manual_seed(0)
    model = GNOTOperator().to(device)
    n = 64

    # (a) SATURATION: the root cause found for v1-v6 was raw t (up to 120 s) and
    # raw N_people (up to 50) saturating ~75-90% of the first tanh layer's
    # units. At the EXTREME ends of the input ranges, far fewer units should
    # be saturated now. Measured on the actual first Linear layers via hooks.
    pre = {}
    h1 = model.query_encoder.proj[0].register_forward_hook(lambda m, i, o: pre.__setitem__("query", o.detach()))
    h2 = model.token_encoder.proj[0].register_forward_hook(lambda m, i, o: pre.__setitem__("token", o.detach()))
    x = (torch.rand(n, 1, device=device) * (ROOM_X[1] - ROOM_X[0])).requires_grad_(True)
    y = (torch.rand(n, 1, device=device) * (ROOM_Y[1] - ROOM_Y[0])).requires_grad_(True)
    z = (torch.rand(n, 1, device=device) * (ROOM_Z[1] - ROOM_Z[0])).requires_grad_(True)
    t = torch.full((n, 1), T_MAX, device=device)                  # worst case: t = 120 s
    V = torch.full((n, NUM_WINDOWS), V_MAX, device=device)        # worst case: all windows at 5 m/s
    N_people = torch.full((n, 1), N_PEOPLE_MAX, device=device)    # worst case: 50 people
    model(x, y, z, t, V, N_people)
    h1.remove(); h2.remove()
    # "saturated" = tanh'(pre) = 1 - tanh^2 < 0.05, the same criterion used to
    # measure the 75-88% (t) / 82-91% (N_people) saturation in the old model.
    # This is what actually verifies that the input SCALING is applied: with
    # raw t=120 / N=50 these fractions were ~88% / ~91%; scaled, ~0%.
    def sat(p):
        return ((1 - torch.tanh(p) ** 2) < 0.05).float().mean().item()
    sat_q = sat(pre["query"])
    tok = pre["token"]                 # (B, 11, D): 8 windows, 2 doors, 1 occupancy
    sat_win = sat(tok[:, :NUM_WINDOWS])
    sat_occ = sat(tok[:, -1])          # occupancy token -- since v12 it always carries the
    # CONSTANT value N_MAX/N_MAX = 1 (N enters only via the linear C factor), so this
    # just checks that constant token isn't saturated; N-independence is tested in (g).
    print(f"  saturated first-layer units at t={T_MAX:.0f}s: {sat_q * 100:.1f}% (was ~88% before v8); "
          f"window tokens at V={V_MAX}: {sat_win * 100:.1f}%; constant occupancy token: "
          f"{sat_occ * 100:.1f}% (raw N=50 was ~91% before v8)")
    assert sat_q < 0.10, f"query encoder still {sat_q:.2%} saturated at t=T_MAX -- t scaling not applied?"
    assert sat_win < 0.10, f"window tokens {sat_win:.2%} saturated -- V/position scaling not applied?"
    assert sat_occ < 0.10, f"occupancy token {sat_occ:.2%} saturated"
    in_dim = model.query_encoder.proj[0].in_features
    expected_in = 2 * model.query_encoder.fourier.n_freq + 3  # fourier + t_hat + ramp + proximity
    assert in_dim == expected_in, f"query encoder input dim {in_dim}, expected {expected_in} (ramp feature missing?)"

    # (b) THE ACTUAL TRAINING LOSS is scaled. FIX (found by audit): an earlier
    # version of this stage compared autograd dC/dt against a finite
    # difference through the SAME model -- which always agrees whatever scaling
    # is inside, so it proved nothing. Instead: force the model's CO2 output to
    # be exactly zero everywhere (zero the C row of the final layer), which
    # makes dc/dt = grad c = lap c = 0, so physics_loss's CO2 term must equal
    # the trivial floor mean((S/S_REF)^2) ~ 0.09. If the /S_REF were missing in
    # physics_loss this would read ~3e-6; if applied twice, thousands.
    from train_gnot import physics_loss
    zero_c = GNOTOperator().to(device)
    with torch.no_grad():
        zero_c.out_head[-1].weight[3].zero_()
        zero_c.out_head[-1].bias[3].zero_()
    _, co2_zero = physics_loss(zero_c, device)
    # expected trivial floor (numpy, same sampling): 0.093 with N ~ U[0,50];
    # 0.279 (= 3x) since v13 evaluates CO2 losses at N = N_MAX. Using the wrong one
    # would mean CO2_LOSS_AT_FULL_OCCUPANCY isn't wired into physics_loss.
    from train_gnot import CO2_LOSS_AT_FULL_OCCUPANCY
    # the trivial floor depends on WHERE points are sampled too (numpy estimates):
    #   source-concentrated (SOURCE_SAMPLE_FRAC=0.6): 0.279 at N_MAX, 0.093 with N~U[0,50]
    #   uniform (SOURCE_SAMPLE_FRAC=0, v14):           0.0527 at N_MAX
    from point_sampler import SOURCE_SAMPLE_FRAC as _ssf
    if _ssf == 0:
        floor_expected = 0.0527 if CO2_LOSS_AT_FULL_OCCUPANCY else 0.0527 / 3
    else:
        floor_expected = 0.279 if CO2_LOSS_AT_FULL_OCCUPANCY else 0.093
    floor = trivial_co2_floor(device, n=50000)
    print(f"  physics_loss CO2 term with C forced to 0: {co2_zero.item():.4f}; "
          f"trivial floor estimate: {floor:.4f} (expected ~{floor_expected:.3f}; was 3.1e-6 before v8)")
    assert 0.65 * floor_expected < floor < 1.5 * floor_expected, (
        f"scaled trivial CO2 floor {floor:.4f} is outside the expected ~{floor_expected:.3f} range")
    assert 0.65 * floor_expected < co2_zero.item() < 1.5 * floor_expected, (
        f"physics_loss CO2 term is {co2_zero.item():.3e} for a zero CO2 field -- expected ~{floor_expected:.3f}. "
        f"The /S_REF scaling in physics_loss is missing or wrong."
    )
    # (b2) OUTPUT SCALING (found by audit: nothing else checks it). Force the
    # raw CO2 output C_hat to exactly 1 everywhere (zero weights, bias 1); the
    # model must then return the physical value C = C_REF * 1.
    from point_sampler import C_REF
    with torch.no_grad():
        zero_c.out_head[-1].bias[3].fill_(1.0)
        _, _, _, C_one, _ = zero_c(x.detach(), y.detach(), z.detach(),
                                   torch.full((n, 1), 60.0, device=device), V, N_people)
    # v10: C = C_REF * (t/T_MAX) * C_hat, so at t=60 s and C_hat=1 expect C_REF*60/T_MAX
    # v12: ... * (N/N_MAX) as well; this call uses N = N_PEOPLE_MAX, so that factor is 1
    c_expected = C_REF * 60.0 / T_MAX * (N_people[0, 0].item() / N_PEOPLE_MAX)  # plain float (printed with :.4f below)
    max_dev = (C_one - c_expected).abs().max().item()
    print(f"  output scaling: C_hat=1, t=60s -> C={C_one.mean().item():.4f} (expected C_REF*60/T_MAX={c_expected:.4f})")
    assert max_dev < 1e-5, f"C deviates from C_REF*t/T_MAX by {max_dev:.2e} -- output scaling wrong"

    # (b3) v9 CO2 BOUNDARY CONDITIONS. A spatially CONSTANT CO2 field (still
    # zero_c, now C = C_REF everywhere) has dc/dn = 0 on every boundary, so the
    # no-flux/outflow loss must be exactly 0 -- a nonzero value would mean the
    # term is picking up something other than the normal gradient.
    from train_gnot import co2_boundary_loss, _planar_wall_normal_derivative
    from point_sampler import sample_walls
    bc_const = co2_boundary_loss(zero_c, device, 1.0).item()
    print(f"  CO2 boundary loss for a constant CO2 field: {bc_const:.2e} (expected exactly 0)")
    assert bc_const < 1e-12, f"CO2 boundary loss is {bc_const:.2e} for a constant field -- should be 0"
    # every planar wall point must be recognised as lying on one of the 6 faces
    # (the normal axis is inferred by exact coordinate equality)
    xw, yw, zw, _, _, _ = sample_walls(600, device)
    _, on_face = _planar_wall_normal_derivative(xw, yw, zw, xw, yw, zw)
    frac_on = on_face.float().mean().item()
    print(f"  wall points assigned to a face: {frac_on * 100:.1f}% (expected 100%)")
    assert frac_on == 1.0, f"only {frac_on:.3%} of wall points matched a face -- normal selection broken"
    # and for the untrained random model the term must be finite and non-zero
    bc_rand = co2_boundary_loss(model, device, 1.0).item()
    assert bc_rand == bc_rand and 0 < bc_rand < float("inf"), f"CO2 boundary loss not finite/positive: {bc_rand}"
    print(f"  CO2 boundary loss for the random-init model: {bc_rand:.4f} (finite, > 0)")
    del zero_c

    # (d) CHECKPOINT GUARD: v5 (unscaled) and v8 (scaled, but no zero-flow
    # constraint) checkpoints must both be refused; the live format accepted.
    from gnot_model import MODEL_FORMAT_KEY, MODEL_FORMAT
    for old in ({"version": "v5_closed_window_fix"}, {"version": "v8_nondim", "nondim": True},
                {"version": "v9_zeroflow_bc", "nondim": True, "model_format": "v9_zeroflow"},
                {"version": "v10_hardic", "nondim": True, "model_format": "v10_hardic"},
                {"version": "v13_fullocc", "nondim": True, "model_format": "v12_linear_n"}):
        try:
            check_checkpoint_compat(old, "fake_old.pth")
            raise AssertionError(f"check_checkpoint_compat accepted an old checkpoint: {old}")
        except RuntimeError:
            pass
    check_checkpoint_compat({"version": "v9", "nondim": True, MODEL_FORMAT_KEY: MODEL_FORMAT}, "fake_new.pth")
    print(f"  checkpoint guard: rejects v5, v8, v9, v10 and v12-v18, accepts {MODEL_FORMAT} -- OK")

    # (e) v9 HARD ZERO-FLOW: with all windows closed the velocity must be
    # EXACTLY zero by construction (the loophole v8 exploited: a spurious slow
    # flow balancing the CO2 source by convection), and clearly non-zero with
    # windows open. Checked through the real curl path used in training.
    xe = (torch.rand(n, 1, device=device) * (ROOM_X[1] - ROOM_X[0])).requires_grad_(True)
    ye = (torch.rand(n, 1, device=device) * (ROOM_Y[1] - ROOM_Y[0])).requires_grad_(True)
    ze = (torch.rand(n, 1, device=device) * (ROOM_Z[1] - ROOM_Z[0])).requires_grad_(True)
    te = torch.rand(n, 1, device=device) * T_MAX
    Ne = torch.rand(n, 1, device=device) * N_PEOPLE_MAX
    speeds = {}
    for label, Ve in (("closed", torch.zeros(n, NUM_WINDOWS, device=device)),
                      ("open", torch.full((n, NUM_WINDOWS), V_MAX, device=device))):
        A1, A2, A3, _, _ = model(xe, ye, ze, te, Ve, Ne)
        u, v, w = model.velocity_from_potential(A1, A2, A3, xe, ye, ze)
        speeds[label] = torch.sqrt(u ** 2 + v ** 2 + w ** 2).max().item()
    print(f"  zero-flow constraint: max speed closed={speeds['closed']:.2e} (must be 0), "
          f"open={speeds['open']:.2e} (must be > 0)")
    assert speeds["closed"] == 0.0, f"closed-window speed {speeds['closed']:.2e} is not exactly 0"
    assert speeds["open"] > 1e-6, "open-window speed is ~0 -- the s(V) factor is suppressing all flow"

    # (f) v10 HARD INITIAL CONDITION: C(t=0) must be exactly 0 for ANY weights
    # (v9 had a -0.039 offset there), and non-zero for t > 0.
    V_mix = torch.rand(n, NUM_WINDOWS, device=device) * V_MAX
    with torch.no_grad():
        _, _, _, C_t0, _ = model(xe, ye, ze, torch.zeros(n, 1, device=device), V_mix, Ne)
        _, _, _, C_t60, _ = model(xe, ye, ze, torch.full((n, 1), 60.0, device=device), V_mix, Ne)
    print(f"  hard IC: max|C(t=0)|={C_t0.abs().max().item():.2e} (must be 0), "
          f"max|C(t=60)|={C_t60.abs().max().item():.2e} (must be > 0)")
    assert C_t0.abs().max().item() == 0.0, "C(t=0) is not exactly 0 -- hard IC not applied"
    assert C_t60.abs().max().item() > 0.0, "C(t=60) is exactly 0 -- CO2 output is dead"

    # (g) v12 EXACT LINEARITY IN OCCUPANCY: C must be exactly 0 for an empty room
    # and exactly double when N doubles (same x, t, V), for ANY weights.
    t60 = torch.full((n, 1), 60.0, device=device)
    with torch.no_grad():
        _, _, _, C_n0, _ = model(xe, ye, ze, t60, V_mix, torch.zeros(n, 1, device=device))
        _, _, _, C_n10, _ = model(xe, ye, ze, t60, V_mix, torch.full((n, 1), 10.0, device=device))
        _, _, _, C_n20, _ = model(xe, ye, ze, t60, V_mix, torch.full((n, 1), 20.0, device=device))
    lin_dev = (C_n20 - 2 * C_n10).abs().max().item() / max(C_n20.abs().max().item(), 1e-30)
    print(f"  occupancy linearity: max|C(N=0)|={C_n0.abs().max().item():.2e} (must be 0), "
          f"relative |C(20) - 2*C(10)| = {lin_dev:.2e} (must be ~0)")
    assert C_n0.abs().max().item() == 0.0, "C is not exactly 0 for an empty room"
    assert lin_dev < 1e-5, f"C does not scale linearly with N (rel. deviation {lin_dev:.2e})"
    # ...and the FLOW must not depend on N at all (no buoyancy term in this model)
    A_10 = model(xe, ye, ze, t60, V_mix, torch.full((n, 1), 10.0, device=device))
    A_20 = model(xe, ye, ze, t60, V_mix, torch.full((n, 1), 20.0, device=device))
    u10 = model.velocity_from_potential(A_10[0], A_10[1], A_10[2], xe, ye, ze)
    u20 = model.velocity_from_potential(A_20[0], A_20[1], A_20[2], xe, ye, ze)
    flow_dev = max((a - b).abs().max().item() for a, b in zip(u10 + (A_10[4],), u20 + (A_20[4],)))
    flow_scale = max(q.abs().max().item() for q in u10 + (A_10[4],))
    print(f"  flow independent of N: max |(u,v,w,p)(N=10) - (u,v,w,p)(N=20)| = {flow_dev:.2e} "
          f"(flow scale {flow_scale:.2e}; must be ~0)")
    # tolerance, not exact equality: identical inputs can still differ in the last bits
    # between two GPU autograd passes; a real N leak shows up at ~1e-3 relative
    assert flow_dev <= 1e-6 * flow_scale + 1e-12, f"velocity/pressure change with N by {flow_dev:.2e} -- N leaks into the network"


@stage("1. FourierFeatures -- shape + finiteness + 1st/2nd derivative w.r.t. raw coords")
def test_fourier_features(device):
    from gnot_model import FourierFeatures
    ff = FourierFeatures(room_length=15.53, sigma=2.5, n_octaves=7).to(device)
    n = 16
    coords = torch.randn(n, 3, device=device, requires_grad=True)
    out = ff(coords)
    expected_dim = 2 * ff.n_freq
    assert out.shape == (n, expected_dim), f"expected shape ({n},{expected_dim}), got {tuple(out.shape)}"
    assert_finite(out, "FourierFeatures output")

    # first + second derivative w.r.t. coords must exist and be finite
    loss = out.sum()
    grad1 = torch.autograd.grad(loss, coords, create_graph=True)[0]
    assert_finite(grad1, "1st derivative of FourierFeatures output")
    grad2 = torch.autograd.grad(grad1.sum(), coords, retain_graph=True)[0]
    assert_finite(grad2, "2nd derivative of FourierFeatures output")
    print(f"  n_freq={ff.n_freq}, output dim={out.shape[1]}, "
          f"grad1 max abs={grad1.abs().max().item():.4f}, grad2 max abs={grad2.abs().max().item():.4f}")
    return ff


@stage("2. QueryEncoder -- shape + source_proximity range + 1st/2nd derivatives")
def test_query_encoder(device):
    from gnot_model import QueryEncoder, D_MODEL
    qe = QueryEncoder().to(device)
    n = 16
    x = torch.randn(n, 1, device=device, requires_grad=True)
    y = torch.randn(n, 1, device=device, requires_grad=True)
    z = torch.randn(n, 1, device=device, requires_grad=True)
    t = torch.rand(n, 1, device=device, requires_grad=True) * 120.0
    out = qe(x, y, z, t)
    # FIX (found by audit): import D_MODEL instead of hardcoding 128, so this
    # doesn't false-fail if D_MODEL is ever changed in gnot_model.py.
    assert out.shape == (n, D_MODEL), f"expected ({n},{D_MODEL}), got {tuple(out.shape)}"
    assert_finite(out, "QueryEncoder output")

    # sanity-check source_proximity directly (recompute the same way QueryEncoder does)
    dist_sq = (x - qe._SOURCE_X) ** 2 + (y - qe._SOURCE_Y) ** 2 + (z - qe._SOURCE_Z) ** 2
    prox = torch.exp(-dist_sq / qe._sigma2)
    assert (prox > 0).all() and (prox <= 1.0).all(), "source_proximity out of expected (0,1] range"

    loss = out.sum()
    grad1 = torch.autograd.grad(loss, [x, y, z], create_graph=True)
    for g, name in zip(grad1, ["x", "y", "z"]):
        assert_finite(g, f"1st derivative w.r.t. {name}")
    grad2x = torch.autograd.grad(grad1[0].sum(), x, retain_graph=True)[0]
    assert_finite(grad2x, "2nd derivative w.r.t. x (this is exactly what physics_loss needs for the Laplacian)")
    print(f"  source_proximity range: [{prox.min().item():.4f}, {prox.max().item():.4f}]")
    return qe


@stage("3. GNOTOperator forward -- full model, tiny batch, all 5 outputs finite")
def test_model_forward(device):
    from gnot_model import GNOTOperator
    from point_sampler import NUM_WINDOWS
    model = GNOTOperator().to(device)
    n = 16
    x = torch.randn(n, 1, device=device, requires_grad=True)
    y = torch.randn(n, 1, device=device, requires_grad=True)
    z = torch.randn(n, 1, device=device, requires_grad=True)
    t = torch.rand(n, 1, device=device, requires_grad=True) * 120.0
    V = torch.rand(n, NUM_WINDOWS, device=device) * 5.0
    N_people = torch.rand(n, 1, device=device) * 50.0

    A1, A2, A3, C, p = model(x, y, z, t, V, N_people)
    for name, tensor in [("A1", A1), ("A2", A2), ("A3", A3), ("C", C), ("p", p)]:
        assert tensor.shape == (n, 1), f"{name} expected shape ({n},1), got {tuple(tensor.shape)}"
        assert_finite(tensor, name)
    print(f"  n_params={sum(p_.numel() for p_ in model.parameters()):,}")
    return model, (x, y, z, t, V, N_people, A1, A2, A3, C, p)


@stage("4. Curl trick -- velocity_from_potential produces a divergence-free field")
def test_curl_trick(device, model_and_inputs):
    model, (x, y, z, t, V, N_people, A1, A2, A3, C, p) = model_and_inputs
    u, v, w = model.velocity_from_potential(A1, A2, A3, x, y, z)
    for name, tensor in [("u", u), ("v", v), ("w", w)]:
        assert tensor.shape == (x.shape[0], 1), f"{name} wrong shape: {tuple(tensor.shape)}"
        assert_finite(tensor, name)

    du_dx = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
    dv_dy = torch.autograd.grad(v.sum(), y, create_graph=True, retain_graph=True)[0]
    dw_dz = torch.autograd.grad(w.sum(), z, create_graph=True, retain_graph=True)[0]
    divergence = du_dx + dv_dy + dw_dz
    max_div = divergence.abs().max().item()
    # should be ~0 up to float32 numerical error (curl of any vector potential
    # is exactly divergence-free analytically -- this checks the IMPLEMENTATION
    # matches the math, not just that it runs)
    assert max_div < 1e-3, f"divergence not near-zero: max|div|={max_div:.6f} (curl trick may be broken)"
    print(f"  max|div(u,v,w)| = {max_div:.2e} (should be ~1e-6 to 1e-4, float32 noise)")


@stage("5a. physics_loss (interior NS + CO2 residuals) -- finite, grads exist")
def test_physics_loss(device):
    from train_gnot import physics_loss
    from gnot_model import GNOTOperator
    model = GNOTOperator().to(device)
    ns_loss, co2_loss = physics_loss(model, device)
    assert_finite(ns_loss, "ns_loss")
    assert_finite(co2_loss, "co2_loss")
    params = list(model.parameters())
    grad_ns = torch.autograd.grad(ns_loss, params, retain_graph=True, allow_unused=True)
    grad_co2 = torch.autograd.grad(co2_loss, params, allow_unused=True)
    n_ns = sum(1 for g in grad_ns if g is not None)
    n_co2 = sum(1 for g in grad_co2 if g is not None)
    assert n_ns > 0, "ns_loss produced no gradients w.r.t. any parameter"
    assert n_co2 > 0, "co2_loss produced no gradients w.r.t. any parameter"
    print(f"  ns_loss={ns_loss.item():.6f}, co2_loss={co2_loss.item():.6f}, "
          f"params with grad: ns={n_ns}/{len(params)}, co2={n_co2}/{len(params)}")


@stage("5b. walls_loss / windows_loss / doors_loss / ic_loss -- each finite, no crash")
def test_boundary_losses(device):
    from train_gnot import walls_loss, windows_loss, doors_loss, ic_loss
    from gnot_model import GNOTOperator
    model = GNOTOperator().to(device)
    co2_weight = 1.0
    L_walls = walls_loss(model, device)
    assert_finite(L_walls, "walls_loss")
    L_windows = windows_loss(model, device, co2_weight)
    assert_finite(L_windows, "windows_loss")
    L_doors = doors_loss(model, device)
    assert_finite(L_doors, "doors_loss")
    L_ic = ic_loss(model, device, co2_weight)
    assert_finite(L_ic, "ic_loss")
    from train_gnot import co2_boundary_loss
    L_co2bc = co2_boundary_loss(model, device, co2_weight)
    assert_finite(L_co2bc, "co2_boundary_loss")
    print(f"  walls={L_walls.item():.5f} windows={L_windows.item():.5f} "
          f"doors={L_doors.item():.5f} ic={L_ic.item():.5f} co2_bc={L_co2bc.item():.5f}")


def _training_step(model, params, optimizer, device):
    """One iteration exactly as train_gnot.main() does it."""
    from train_gnot import (
        physics_loss, walls_loss, windows_loss, doors_loss, ic_loss,
        compute_param_grads, GRAD_CLIP_MAX_NORM,
    )
    co2_weight = 1.0
    optimizer.zero_grad()
    L_ns, L_co2 = physics_loss(model, device)
    grad_ns = compute_param_grads(L_ns, params, retain_graph=True)
    grad_co2 = compute_param_grads(L_co2, params, retain_graph=False)
    for p, g_ns, g_co2 in zip(params, grad_ns, grad_co2):
        total_grad = None
        if g_ns is not None:
            total_grad = g_ns
        if g_co2 is not None:
            weighted = co2_weight * g_co2
            total_grad = weighted if total_grad is None else total_grad + weighted
        if total_grad is None:
            continue
        p.grad = total_grad.clone() if p.grad is None else p.grad + total_grad

    walls_loss(model, device).backward()
    windows_loss(model, device, co2_weight).backward()
    doors_loss(model, device).backward()
    ic_loss(model, device, co2_weight).backward()
    from train_gnot import co2_boundary_loss
    co2_boundary_loss(model, device, co2_weight).backward()  # v9: mirrors train_gnot.main()

    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_MAX_NORM)
    optimizer.step()


@stage("6. Full combined training steps (configured optimizer) -- weights change, state finite, resumable")
def test_one_training_step(device):
    """v15: uses the CONFIGURED optimizer (train_gnot.make_optimizer). SOAP's first
    step only initialises its preconditioner and leaves the weights unchanged
    (by design, soap.py), so 3 steps are run and the weights checked after them."""
    import io
    from train_gnot import make_optimizer, OPTIMIZER
    from gnot_model import GNOTOperator
    model = GNOTOperator().to(device)
    params = list(model.parameters())
    optimizer = make_optimizer(params)
    before = [p.detach().clone() for p in params]

    for _ in range(3):
        _training_step(model, params, optimizer, device)

    n_changed = sum(1 for b, p in zip(before, params) if not torch.equal(b, p.detach()))
    assert n_changed > 0, "optimizer.step() ran but NO parameters changed -- gradients may all be zero/None"
    for p in params:
        assert_finite(p.detach(), "a model parameter after optimizer.step()")
    n_state = 0
    for st in optimizer.state.values():   # moments (+ SOAP's preconditioner matrices) finite
        for v in st.values():
            for t in (v if isinstance(v, list) else [v]):
                if torch.is_tensor(t):
                    assert_finite(t, "an optimizer state tensor")
                    n_state += 1
    print(f"  optimizer={OPTIMIZER}: {n_changed}/{len(params)} parameter tensors changed after 3 steps "
          f"(expected: all of them); {n_state} state tensors finite")

    # checkpoint round trip: save the optimizer state, load into a fresh optimizer, keep training
    buf = io.BytesIO()
    torch.save(optimizer.state_dict(), buf)
    buf.seek(0)
    opt2 = make_optimizer(params)
    opt2.load_state_dict(torch.load(buf, map_location=device))
    _training_step(model, params, opt2, device)
    for p in params:
        assert_finite(p.detach(), "a model parameter after a step with the reloaded optimizer")
    print("  optimizer state saves, reloads and keeps training (resume path OK)")


def _check_decay_schedule(lr_at, LR, LR_MIN, LR_DECAY_START, MAX_ITERS):
    """Constant LR up to the decay start, monotone cosine, exact endpoints/midpoint."""
    assert lr_at(0) == LR and lr_at(LR_DECAY_START) == LR, "LR must be constant before LR_DECAY_START"
    assert abs(lr_at(MAX_ITERS) - LR_MIN) < 1e-15, f"final LR {lr_at(MAX_ITERS)} != LR_MIN {LR_MIN}"
    vals = [lr_at(i) for i in range(LR_DECAY_START, MAX_ITERS + 1, 50)]
    assert all(a >= b for a, b in zip(vals, vals[1:])), "LR schedule is not monotonically decreasing"
    mid = lr_at((LR_DECAY_START + MAX_ITERS) // 2)
    assert abs(mid - (LR_MIN + 0.5 * (LR - LR_MIN))) < 1e-9, f"cosine midpoint {mid} is wrong"
    print(f"  LR: {lr_at(0):.1e} until iter {LR_DECAY_START}, {mid:.2e} at midpoint, {lr_at(MAX_ITERS):.1e} at {MAX_ITERS}")


@stage("7. LR schedule + resume -- schedule as configured; resume checkpoint (if any) loads cleanly")
def test_lr_schedule_and_resume(device):
    import os
    from train_gnot import lr_at, LR, LR_MIN, LR_DECAY_START, MAX_ITERS, RESUME_FROM, HERE
    from gnot_model import GNOTOperator, check_checkpoint_compat
    if LR_DECAY_START is None:  # constant-LR configuration (the default)
        assert all(lr_at(i) == LR for i in range(0, MAX_ITERS + 1, 500)), "LR should be constant"
        print(f"  LR: constant {LR:g} for all {MAX_ITERS} iterations (LR_DECAY_START=None)")
    else:
        _check_decay_schedule(lr_at, LR, LR_MIN, LR_DECAY_START, MAX_ITERS)

    # resume path: the exact load sequence train_gnot.main() will perform
    if RESUME_FROM is None:
        print("  RESUME_FROM is None -- fresh run, nothing to resume")
        return
    path = os.path.join(HERE, RESUME_FROM)
    assert os.path.isfile(path), f"RESUME_FROM checkpoint not found: {path}"
    ckpt = torch.load(path, map_location=device)
    check_checkpoint_compat(ckpt, path)
    assert "optimizer_state" in ckpt, "resume checkpoint has no optimizer_state"
    from train_gnot import make_optimizer, OPTIMIZER
    assert ckpt.get("optimizer", "adam") == OPTIMIZER, \
        f"resume checkpoint used {ckpt.get('optimizer', 'adam')!r}, OPTIMIZER is {OPTIMIZER!r}"
    model = GNOTOperator().to(device)
    opt = make_optimizer(model.parameters())
    model.load_state_dict(ckpt["model_state"])
    opt.load_state_dict(ckpt["optimizer_state"])
    assert ckpt["iter"] + 1 <= MAX_ITERS, f"resume iter {ckpt['iter']} is already past MAX_ITERS {MAX_ITERS}"
    print(f"  resume checkpoint OK: {RESUME_FROM} (iter={ckpt['iter']}, version={ckpt.get('version')}), "
          f"model + {OPTIMIZER} state load; will train iters {ckpt['iter'] + 1}-{MAX_ITERS}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: running on CPU -- fine for catching shape/NaN bugs, "
              "but won't tell you anything about GPU memory behavior.")

    test_sample_interior(device)
    test_scenario_location_independence(device)
    test_audit_fixes(device)
    test_flux_scaled_walls(device)
    test_throughflow_exact(device)
    test_nondim(device)
    test_fourier_features(device)
    test_query_encoder(device)
    model_and_inputs = test_model_forward(device)
    if model_and_inputs is not None:
        test_curl_trick(device, model_and_inputs)
    test_physics_loss(device)
    test_boundary_losses(device)
    test_one_training_step(device)
    test_lr_schedule_and_resume(device)

    print("\n" + "=" * 60)
    if FAILED:
        print("RESULT: at least one stage FAILED -- see above for exactly which one.")
        sys.exit(1)
    else:
        print("RESULT: ALL STAGES PASSED. Safe to proceed to the multi-iteration smoke test.")


if __name__ == "__main__":
    main()
