"""
Step-by-step test of the physics-only pipeline, from the sampler to a full training step; stops at the first failure.
Usage: python3 staged_smoke_test.py
"""
import sys
import torch

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

    assert (x >= ROOM_X[0]).all() and (x <= ROOM_X[1]).all(), "x out of room bounds"
    assert (y >= ROOM_Y[0]).all() and (y <= ROOM_Y[1]).all(), "y out of room bounds"
    assert (z >= ROOM_Z[0]).all() and (z <= ROOM_Z[1]).all(), "z out of room bounds"

    xn = x.detach().cpu().numpy().ravel()
    yn = y.detach().cpu().numpy().ravel()
    for cx, cy, r, _, _ in COLUMNS:
        inside = (xn - cx) ** 2 + (yn - cy) ** 2 <= r ** 2
        assert not inside.any(), f"{inside.sum()} points landed inside column at ({cx},{cy})"

    dist = torch.sqrt((x - SOURCE_X) ** 2 + (y - SOURCE_Y) ** 2 + (z - BREATHING_HEIGHT) ** 2)
    frac_near = (dist < CO2_SOURCE_SIGMA).float().mean().item()
    print(f"  n_source={int(round(n * SOURCE_SAMPLE_FRAC))}, n_uniform={n - int(round(n * SOURCE_SAMPLE_FRAC))}, "
          f"fraction of all {n} points within ~1 sigma of source: {frac_near:.3f}")
    room_volume = (ROOM_X[1] - ROOM_X[0]) * (ROOM_Y[1] - ROOM_Y[0]) * (ROOM_Z[1] - ROOM_Z[0])
    sphere_volume = (4.0 / 3.0) * torch.pi * CO2_SOURCE_SIGMA ** 3
    uniform_baseline = min(1.0, sphere_volume / room_volume)
    if SOURCE_SAMPLE_FRAC > 0:
        assert frac_near > uniform_baseline, (
            f"fraction near source ({frac_near:.3f}) is not higher than the pure-uniform baseline "
            f"({uniform_baseline:.3f}) -- source-concentrated sampling may not be working"
        )
    else:
        assert 0.07 < frac_near < 0.16, (
            f"fraction near source {frac_near:.3f} is not the uniform-sampling value ~0.116 -- "
            f"sampling is not uniform")

    from point_sampler import CLOSED_SCENARIO_FRAC
    all_zero_frac = (V == 0).all(dim=1).float().mean().item()
    print(f"  fraction of {n} points with ALL windows exactly V=0: {all_zero_frac:.3f} "
          f"(expected close to CLOSED_SCENARIO_FRAC={CLOSED_SCENARIO_FRAC})")
    assert all_zero_frac > CLOSED_SCENARIO_FRAC * 0.5, (
        f"only {all_zero_frac:.3f} of points have all-zero V, expected close to "
        f"{CLOSED_SCENARIO_FRAC} -- closed-scenario oversampling may not be working"
    )

    from point_sampler import (reset_interior_pools, POOL_REFRESH_FRAC, POOL_REFRESH_EVERY,
                               USE_PERSISTENT_POOL)
    if not USE_PERSISTENT_POOL:
        xa, _, _, _, _, _ = sample_interior(200, device)
        xb, _, _, _, _, _ = sample_interior(200, device)
        same = torch.isclose(xa, xb).float().mean().item()
        print(f"  persistent pool is OFF (USE_PERSISTENT_POOL=False): {same:.3f} of points "
              f"identical across two calls (expected ~0, i.e. fully fresh sampling)")
        assert same < 0.05, "pool is supposed to be off, but points are being reused across calls"
        return
    reset_interior_pools()
    n_pool_test = 200
    x0, y0, z0, _, _, _ = sample_interior(n_pool_test, device)
    x1, y1, z1, _, _, _ = sample_interior(n_pool_test, device)
    unchanged = torch.isclose(x0, x1).float().mean().item()
    assert unchanged > 0.95, (
        f"only {unchanged:.3f} of points stayed identical between two consecutive "
        f"calls (expected ~1.0, no refresh due yet) -- persistent pool may not be "
        f"working, still resampling 100% fresh every call"
    )
    for _ in range(POOL_REFRESH_EVERY - 3):
        sample_interior(n_pool_test, device)
    x_before, _, _, _, _, _ = sample_interior(n_pool_test, device)
    x_after, _, _, _, _, _ = sample_interior(n_pool_test, device)
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
    reset_interior_pools()


@stage("0c. v16 scenario/location independence -- every window, wall face, column and interior region sees all scenario types")
def test_scenario_location_independence(device):
    """Test that the scenario values do not depend on the point position."""
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
    x, y, z, t, V, N = sample_interior(4000, device)
    on_bound = ((z == ROOM_Z[0]) | (z == ROOM_Z[1])).float().mean().item()
    assert on_bound < 1e-3, f"{on_bound:.3f} of interior points sit exactly on floor/ceiling (clamping?)"
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
    x, y, z, t, V, N = sample_ic(4000, device)
    near = (torch.sqrt((x - SOURCE_X) ** 2 + (y - SOURCE_Y) ** 2 + (z - BREATHING_HEIGHT) ** 2)
            < CO2_SOURCE_SIGMA).float().mean().item()
    assert 0.07 < near < 0.16 and (t == 0).all(), f"IC points not uniform at t=0 (near-source share {near:.3f})"

    class Stub(torch.nn.Module):
        def forward(self, x, y, z, t, V, N):
            zero = 0.0 * (x + y + z)
            return zero, zero, zero, C_REF + zero, 1.0 + zero

        def velocity_from_potential(self, A1, A2, A3, x, y, z):
            return 0.0 * x, 0.0 * y, 0.0 * z
    stub = Stub()
    torch.manual_seed(123)
    L = windows_loss(stub, "cpu", 1.0).item()
    torch.manual_seed(123)
    xw, yw, zw, tw, Vw, Nw, idx = sample_windows(POINTS_WINDOWS_PER, "cpu")
    Vk = Vw.gather(1, idx)
    target = Vk * torch.tanh(3.0 * tw / TAU_RAMP)
    from train_gnot import V_REL_FLOOR, USE_WINDOW_NORMAL_TARGET
    if USE_WINDOW_NORMAL_TARGET:
        vel = (target ** 2 / target.abs().clamp_min(V_REL_FLOOR) ** 2).mean().item()
    else:
        vel = 0.0
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
    """Test of the flux-scaled wall loss with a constant-velocity stub model."""
    from point_sampler import sample_walls, sample_columns_surface, ROOM_X, ROOM_Y, COLUMNS, WINDOWS
    from train_gnot import (walls_loss, throughflow_scale, A_SOLID, Q_FLOOR, POINTS_WALLS, POINTS_COLUMNS_PER)
    assert 430 < A_SOLID < 450, f"A_SOLID = {A_SOLID:.1f} m^2, expected ~441"
    V = torch.tensor([[3.0] + [0.0] * 7, [0.0] * 8, [5.0] * 8])
    t = torch.tensor([[60.0], [60.0], [0.0]])
    q = throughflow_scale(V, t).squeeze(1).tolist()
    areas = [(w[1] - w[0]) * (w[3] - w[2]) for w in WINDOWS]
    expect_q = [3.0 * areas[0], Q_FLOOR, 5.0 * sum(areas)]
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
    from train_gnot import velocity_scale, slip_scale, USE_WINDOW_NORMAL_TARGET
    s2 = a * a + b * b + c * c
    slip = (s2 / slip_scale(Vw) ** 2).mean().item() + (s2 / slip_scale(Vc) ** 2).mean().item()
    expected = slip + (flux ** 2).mean().item()
    assert abs(L - expected) < 1e-4 * max(1.0, expected), f"walls_loss {L:.6f} != expected {expected:.6f}"
    torch.manual_seed(7)
    L0 = walls_loss(Stub(), "cpu", flux_weight=0.0).item()
    assert abs(L0 - slip) < 1e-5 * max(1.0, slip), f"walls_loss with flux_weight=0 is {L0}, not the no-slip part {slip}"
    from train_gnot import windows_loss, POINTS_WINDOWS_PER
    from point_sampler import sample_windows, TAU_RAMP
    torch.manual_seed(11)
    Lw = windows_loss(Stub(), "cpu", 0.0, rel_weight=0.0).item()
    torch.manual_seed(11)
    _, _, _, tw_, Vw_, _, idx_ = sample_windows(POINTS_WINDOWS_PER, "cpu")
    if USE_WINDOW_NORMAL_TARGET:
        tgt = -Vw_.gather(1, idx_) * torch.tanh(3.0 * tw_ / TAU_RAMP)
        expect_w = (a * a) + ((b - tgt) ** 2).mean().item() + (c * c)
    else:
        expect_w = ((a * a + c * c) / slip_scale(Vw_) ** 2).mean().item()
    assert abs(Lw - expect_w) < 1e-4 * max(1.0, expect_w), f"windows_loss {Lw:.6f} != expected {expect_w:.6f}"
    print(f"  A_SOLID={A_SOLID:.1f} m^2; walls_loss for a constant 0.01-0.03 m/s velocity = {L:.3f} "
          f"(no-slip part only would be {slip:.5f}) -- leak is expensive; window/slip terms relative to slip_scale")


@stage("0f. v19 exact through-flow -- no normal velocity on ANY solid surface, exact window/door fluxes, alpha split")
def test_throughflow_exact(device):
    """Test that the built-in base flow meets the window and door fluxes exactly."""
    import math
    from gnot_model import GNOTOperator
    from train_gnot import get_velocity_and_derivs
    from point_sampler import (sample_walls, sample_columns_surface, WINDOWS, DOORS, COLUMNS,
                               ROOM_X, ROOM_Y, ROOM_Z, NUM_WINDOWS)
    torch.manual_seed(3)
    model = GNOTOperator().to(device)
    with torch.no_grad():
        model.alpha_head[2].bias.fill_(1.5)
    Vrow = torch.tensor([[2.5, 0.0, 4.0, 0.0, 1.0, 0.0, 0.0, 5.0]], device=device)
    t60 = 60.0

    def vel(x, y, z, V=Vrow, t=t60):
        n = x.shape[0]
        x, y, z = (a.detach().clone().requires_grad_(True) for a in (x, y, z))
        u, v, w, _, _ = get_velocity_and_derivs(model, x, y, z, torch.full((n, 1), t, device=device),
                                                V.expand(n, -1), torch.full((n, 1), 20.0, device=device))
        return u.detach(), v.detach(), w.detach()

    g = torch.Generator().manual_seed(0)
    xi = (torch.rand(2000, 1, generator=g) * (ROOM_X[1] - 2) + 1).to(device)
    yi = (torch.rand(2000, 1, generator=g) * (ROOM_Y[1] - 2) + 1).to(device)
    zi = (torch.rand(2000, 1, generator=g) * (ROOM_Z[1] - 0.6) + 0.3).to(device)
    u, v, w = vel(xi, yi, zi)
    scale = torch.sqrt(u ** 2 + v ** 2 + w ** 2).mean().item()
    assert scale > 0.05, f"interior speed {scale:.3e} -- the through-flow is missing"

    worst = {}
    x, y, z, _, _, _ = sample_walls(3000, device)
    u, v, w = vel(x, y, z)
    on_x = (x == ROOM_X[0]) | (x == ROOM_X[1])
    on_y = (y == ROOM_Y[0]) | (y == ROOM_Y[1])
    un = torch.where(on_x, u, torch.where(on_y, v, w))
    worst["walls/floor/ceiling"] = un.abs().max().item()
    xc, yc, zc, _, _, _ = sample_columns_surface(300, device)
    u, v, w = vel(xc, yc, zc)
    k = torch.stack([(xc - cx) ** 2 + (yc - cy) ** 2 for cx, cy, _, _, _ in COLUMNS], 0).argmin(0)
    ctr = torch.tensor([[cx, cy, r] for cx, cy, r, _, _ in COLUMNS], device=device)[k.squeeze(-1)]
    nx, ny = (xc.squeeze(-1) - ctr[:, 0]) / ctr[:, 2], (yc.squeeze(-1) - ctr[:, 1]) / ctr[:, 2]
    worst["columns"] = (u.squeeze(-1) * nx + v.squeeze(-1) * ny).abs().max().item()
    pts = []
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

    def flux(a, b, c, d, yval, m1=60, m2=30):
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


@stage("0g. v20/v21 CO2 window factor (ON: c = 0 on open windows / OFF in v21), potential-flow door split, slip scale")
def test_v20(device):
    from gnot_model import GNOTOperator
    from throughflow import (co2_window_factor, alpha_potential, DOOR1_SHARE_PER_WINDOW, EPS_WINDOW)
    from train_gnot import slip_scale, velocity_scale, V_REL_FLOOR
    from point_sampler import WINDOWS, ROOM_Y, ROOM_Z, NUM_WINDOWS
    torch.manual_seed(5)
    model = GNOTOperator().to(device)
    V = torch.tensor([[3.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device)
    n = 400
    from gnot_model import USE_CO2_WINDOW_FACTOR
    def pts(k, core):
        a, b = WINDOWS[k][0], WINDOWS[k][1]
        lo, hi = (a + EPS_WINDOW, b - EPS_WINDOW) if core else (a, b)
        x = torch.rand(n, 1, device=device) * (hi - lo) + lo
        return x, torch.full_like(x, ROOM_Y[1]), torch.rand(n, 1, device=device) * ROOM_Z[1]

    def C_at(x, y, z):
        with torch.no_grad():
            _, _, _, C, _ = model(x, y, z, torch.full_like(x, 60.0), V.expand(x.shape[0], -1),
                                  torch.full_like(x, 20.0))
        return C
    c_ref = C_at(torch.rand(n, 1, device=device) * 10 + 2, torch.rand(n, 1, device=device) * 5 + 2,
                 torch.rand(n, 1, device=device) * 2 + 0.5).abs().mean().item()
    c_open = max(C_at(*pts(0, True)).abs().max().item(), C_at(*pts(2, True)).abs().max().item())
    if USE_CO2_WINDOW_FACTOR:
        assert c_ref > 0 and c_open < 1e-5 * max(c_ref, 1e-12) + 1e-9, (
            f"|C| on open-window cores {c_open:.2e} (interior mean {c_ref:.2e}): c = 0 not exact")
    else:
        assert c_ref > 0 and c_open > 1e-3 * c_ref, (
            f"|C| on open-window cores {c_open:.2e} ~ 0 although USE_CO2_WINDOW_FACTOR is False")
    xc, yc, zc = pts(1, False)
    om_closed = co2_window_factor(xc, yc, V.expand(n, -1))
    xw = torch.full((n, 1), 3.4, device=device)
    om_wall = co2_window_factor(xw, torch.full_like(xw, ROOM_Y[1]), V.expand(n, -1))
    assert (om_closed - 1).abs().max().item() < 1e-6 and (om_wall - 1).abs().max().item() < 1e-6, \
        "co2 window factor must be 1 on closed windows and walls"
    probe = torch.zeros(NUM_WINDOWS + 1, NUM_WINDOWS, device=device)
    for k in range(NUM_WINDOWS):
        probe[k, k] = 3.0
    a = model.door_split(torch.full((NUM_WINDOWS + 1, 1), 60.0, device=device), probe).squeeze(1).tolist()
    for k in range(NUM_WINDOWS):
        assert abs(a[k] - DOOR1_SHARE_PER_WINDOW[k]) < 1e-4, f"alpha(W{k + 1} only) {a[k]:.4f} != {DOOR1_SHARE_PER_WINDOW[k]}"
    assert abs(a[-1] - 0.5) < 1e-6, f"alpha(all closed) {a[-1]} != 0.5"
    from train_gnot import SLIP_MAX_RATIO
    Vs = torch.tensor([[3.0] * 8, [3.0] + [0.0] * 7, [0.0] * 8], device=device)
    s = slip_scale(Vs).squeeze(1).tolist()
    u = velocity_scale(Vs).squeeze(1).tolist()
    assert abs(s[0] - max(3.0, u[0] / SLIP_MAX_RATIO)) < 1e-4 and abs(s[1] - u[1]) < 1e-4 \
        and abs(s[2] - max(V_REL_FLOOR, u[2] / SLIP_MAX_RATIO)) < 1e-6, f"slip_scale {s} (U_ref {u})"
    print(f"  CO2 window factor {'ON' if USE_CO2_WINDOW_FACTOR else 'OFF (v21)'}: |C| on open-window cores {c_open:.1e} "
          f"(interior {c_ref:.1e}); omega = 1 on closed windows/walls; "
          f"alpha(W1..W8) = {', '.join(f'{v:.3f}' for v in a[:-1])}; slip scale {s[0]:.2f}/{s[1]:.2f}/{s[2]:.2f} m/s")


@stage("0h. v21 single scenario + viscosity curriculum -- every sampler uses FIXED_V, nu schedule, nu enters linearly")
def test_v21(device):
    import point_sampler as ps
    from train_gnot import (SINGLE_SCENARIO_V, NU_SCHEDULE, nu_at, NU, MAX_ITERS, physics_loss)
    from gnot_model import GNOTOperator
    for last, nu in NU_SCHEDULE:
        if last is not None:
            assert nu_at(last) == nu, f"nu_at({last}) = {nu_at(last)} != {nu} (stage end must be inclusive)"
    assert nu_at(MAX_ITERS) == NU and NU_SCHEDULE[-1][0] is None, "curriculum must end at the physical NU"
    nus = [nu_at(i) for i in range(0, MAX_ITERS + 1, 100)]
    assert all(a >= b for a, b in zip(nus, nus[1:])), "viscosity must not increase during training"
    assert ps.FIXED_V is None, "point_sampler.FIXED_V must default to None (only main() sets it)"
    ps.FIXED_V = SINGLE_SCENARIO_V
    try:
        target = torch.tensor(SINGLE_SCENARIO_V, device=device) if SINGLE_SCENARIO_V else None
        outs = {"interior": ps.sample_interior(500, device), "walls": ps.sample_walls(300, device),
                "columns": ps.sample_columns_surface(20, device), "doors": ps.sample_doors(100, device),
                "windows": ps.sample_windows(20, device), "ic": ps.sample_ic(200, device)}
        for name, o in outs.items():
            V_ = o[4]
            if target is not None:
                assert (V_ == target).all(), f"{name}: V differs from FIXED_V"
            assert o[3].std().item() > 0 or name == "ic", f"{name}: t is not random any more"
        print(f"  FIXED_V = {SINGLE_SCENARIO_V}: all 6 samplers return it for every point; t random")
    finally:
        ps.FIXED_V = None
    V_mix = ps.sample_interior(500, device)[4]
    assert (V_mix == 0).all(dim=1).any() and (V_mix > 0).all(dim=1).any(), "mix not restored after FIXED_V = None"
    import train_gnot as tg
    torch.manual_seed(11)
    model = GNOTOperator().to(device)
    L = []
    w_flag, h_flag = tg.USE_BP_RESIDUAL_WEIGHT, tg.NS_PSEUDO_HUBER
    tg.USE_BP_RESIDUAL_WEIGHT = tg.NS_PSEUDO_HUBER = False
    try:
        for nu in (0.0, 0.05, 0.1, 0.01, None):
            torch.manual_seed(3)
            L.append(physics_loss(model, device, nu=nu)[0].item() if nu is not None
                     else physics_loss(model, device)[0].item())
    finally:
        tg.USE_BP_RESIDUAL_WEIGHT, tg.NS_PSEUDO_HUBER = w_flag, h_flag
    L_default = L.pop()
    h = 0.05
    a, b = L[0], (4 * L[1] - L[2] - 3 * L[0]) / (2 * h)
    c = (L[2] - 2 * L[1] + L[0]) / (2 * h * h)
    pred = a + b * 0.01 + c * 0.01 ** 2
    assert abs(pred - L[3]) < 1e-4 * max(abs(L[3]), 1e-12), f"L(nu) not quadratic: predicted {pred}, got {L[3]}"
    assert L[2] != L[0], "physics_loss ignores its nu argument"
    assert abs(L_default - L[3]) < 1e-5 * max(abs(L[3]), 1e-12), "physics_loss default nu is not NU = 0.01"
    print(f"  nu schedule {NU_SCHEDULE} OK; NS loss at nu 0 / 0.05 / 0.1 / 0.01 = "
          f"{L[0]:.4g} / {L[1]:.4g} / {L[2]:.4g} / {L[3]:.4g} (exact quadratic in nu, default = NU)")


@stage("0i. v22 smooth jet -- psi unchanged on window + door wall, exact end values, much smaller B_p curvature inside")
def test_v22(device):
    import throughflow as tfl
    from point_sampler import ROOM_X, ROOM_Y, ROOM_Z
    assert tfl.JET_SMOOTH, "v22 expects throughflow.JET_SMOOTH = True"
    n = 2000
    dt = torch.float64
    V = torch.tensor([[1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0]], device=device, dtype=dt).expand(n, -1)
    t = torch.full((n, 1), 60.0, device=device, dtype=dt)
    al = torch.full((n, 1), 0.55, device=device, dtype=dt)
    x = torch.rand(n, 1, device=device, dtype=dt) * (ROOM_X[1] - ROOM_X[0]) + ROOM_X[0]
    z = torch.rand(n, 1, device=device, dtype=dt) * ROOM_Z[1]
    d0 = d1 = 0.0
    for yv in (ROOM_Y[1], ROOM_Y[0]):
        xw = x.clone().requires_grad_(True)
        yw = torch.full((n, 1), yv, device=device, dtype=dt)
        _, ps_ = tfl.through_flow_potential(xw, yw, z, t, V, al, smooth=True)
        _, pl_ = tfl.through_flow_potential(xw, yw, z, t, V, al, smooth=False)
        d0 = max(d0, (ps_ - pl_).abs().max().item())
        d1 = max(d1, (torch.autograd.grad(ps_.sum(), xw)[0] - torch.autograd.grad(pl_.sum(), xw)[0]).abs().max().item())
    assert d0 < 1e-12 and d1 < 1e-9, f"psi / normal velocity on the window or door wall changed: {d0:.1e} / {d1:.1e}"
    rt = tfl.ramp(t[:2])
    xe = torch.tensor([[ROOM_X[0]], [ROOM_X[1]]], device=device, dtype=dt)
    Fs = tfl._F_top_smooth(xe, V[:2], rt, tfl.JET_EDGE_W).squeeze(1).tolist()
    Fl = tfl._F_top(xe, V[:2], rt).squeeze(1).tolist()
    assert abs(Fs[0]) < 1e-12 and abs(Fs[1] - Fl[1]) < 1e-12, f"smooth F_top ends {Fs} vs sharp {Fl}"
    xm = x.clone().requires_grad_(True)
    ym = torch.full((n, 1), ROOM_Y[1] / 2, device=device, dtype=dt)
    lap = {}
    for sm in (False, True):
        _, p_ = tfl.through_flow_potential(xm, ym, z, t, V, al, smooth=sm)
        v_ = -torch.autograd.grad(p_.sum(), xm, create_graph=True)[0]
        d2 = torch.autograd.grad(torch.autograd.grad(v_.sum(), xm, create_graph=True)[0].sum(), xm)[0]
        lap[sm] = d2.abs().max().item()
    assert lap[True] < 0.05 * lap[False], f"smooth jet curvature {lap[True]:.3g} not << sharp {lap[False]:.3g}"
    print(f"  window + door wall: psi / normal velocity unchanged ({d0:.1e} / {d1:.1e}); smooth profile ends exact; "
          f"max |d2v/dx2| mid-room: sharp {lap[False]:.3g} -> smooth {lap[True]:.3g} 1/(m s)")


@stage("0j. v22/v23 lap(curl B_p) vs finite differences; v23 Pseudo-Huber NS loss consistent with the plain loss")
def test_v22_weight(device):
    import throughflow as tfl
    import train_gnot as tg
    import point_sampler as ps
    from gnot_model import GNOTOperator
    assert tg.NS_PSEUDO_HUBER and not tg.USE_BP_RESIDUAL_WEIGHT, "v23 expects Pseudo-Huber ON, B_p weighting OFF"
    dt = torch.float64
    torch.manual_seed(4)
    n = 16
    x = torch.rand(n, 1, dtype=dt, device=device) * 12 + 2
    y = torch.rand(n, 1, dtype=dt, device=device) * 7 + 1
    z = torch.rand(n, 1, dtype=dt, device=device) * 2.5 + 0.3
    t = torch.full((n, 1), 60.0, dtype=dt, device=device)
    V = torch.tensor([[1.0, 0, 0, 0, 0, 0, 2.0, 0]], dtype=dt, device=device).expand(n, -1)
    al = torch.full((n, 1), 0.55, dtype=dt, device=device)
    lap = tfl.bp_velocity_laplacian(x, y, z, t, V, al)

    def vel(xx, yy, zz):
        xx, yy, zz = (q.clone().requires_grad_(True) for q in (xx, yy, zz))
        chi, psi = tfl.through_flow_potential(xx, yy, zz, t, V, al)
        gp = torch.autograd.grad(psi.sum(), (xx, yy))
        gc = torch.autograd.grad(chi.sum(), (yy, zz))
        return torch.cat([gp[1], gc[1] - gp[0], -gc[0]], 1).detach()
    h = 1e-3
    u0 = vel(x, y, z)
    fd = sum(vel(*(q + h * (i == k) for i, q in enumerate((x, y, z)))) + vel(*(q - h * (i == k) for i, q in enumerate((x, y, z))))
             - 2 * u0 for k in range(3)) / h ** 2
    err = ((fd - lap).abs().max() / lap.abs().max().clamp_min(1e-12)).item()
    assert err < 1e-3, f"lap(curl B_p) differs from finite differences by {err:.1e} (relative)"
    assert not lap.requires_grad, "bp_velocity_laplacian must be detached"
    xs = torch.linspace(0.0, 15.53, 400, device=device).view(-1, 1)
    o = torch.ones_like(xs)
    lap32 = tfl.bp_velocity_laplacian(xs, 4.0 * o, 1.0 * o, 60.0 * o,
                                      torch.tensor([[1.0] + [0.0] * 7], device=device).expand(400, -1), 0.55 * o)
    assert torch.isfinite(lap32).all(), "lap(curl B_p) not finite in float32 somewhere along x"
    torch.manual_seed(12)
    model = GNOTOperator().to(device)
    ps.FIXED_V = [1.0] + [0.0] * 7
    try:
        torch.manual_seed(5)
        L_h = tg.physics_loss(model, device)[0].item()
        raw = tg._NS_DIAG["raw"]
        tg.NS_PSEUDO_HUBER = False
        torch.manual_seed(5)
        L_plain = tg.physics_loss(model, device)[0].item()
    finally:
        tg.NS_PSEUDO_HUBER = True
        ps.FIXED_V = None
    assert abs(raw - L_plain) <= 1e-4 * abs(L_plain), f"logged NSraw {raw} != plain loss {L_plain}"
    assert 0 < L_h <= L_plain + 1e-9, f"Pseudo-Huber {L_h} not in (0, plain {L_plain}]"
    print(f"  lap(curl B_p) vs finite differences: rel. error {err:.1e}; W1 1 m/s random model: NS Pseudo-Huber "
          f"{L_h:.4g} vs plain {L_plain:.4g} (logged raw {raw:.4g})")


@stage("0k. v23/v24 output scales -- s = door jet speed (0 closed), p = U_ref^2 p_raw without ramp, doors loss scaled, alpha fixed")
def test_v23(device):
    import gnot_model as gm
    import train_gnot as tg
    assert gm.P_SCALE_FLOOR == tg.U_NS_FLOOR, "gnot_model.P_SCALE_FLOOR must equal train_gnot.U_NS_FLOOR"
    Vs = torch.tensor([[1.0] + [0.0] * 7, [0.0] * 8, [3.0] * 8], device=device)
    s = gm.door_jet_speed(Vs).squeeze(1).tolist()
    u = tg.velocity_scale(Vs).squeeze(1).tolist()
    assert s[1] == 0.0, "correction scale must be exactly 0 with all windows closed"
    assert abs(max(s[0], 0.5) - u[0]) < 1e-6 and abs(s[2] - u[2]) < 1e-5, f"s {s} vs velocity_scale {u}"
    torch.manual_seed(0)
    model = gm.GNOTOperator().to(device)
    with torch.no_grad():
        model.out_head[-1].weight[4].zero_()
        model.out_head[-1].bias[4].fill_(1.0)
    n = 32
    x = torch.rand(n, 1, device=device) * 10 + 2
    y = torch.rand(n, 1, device=device) * 6 + 1
    z = torch.rand(n, 1, device=device) * 2 + 0.5
    for Vrow, ur in ((Vs[0:1], u[0]), (Vs[2:3], u[2])):
        for tv in (0.0, 60.0):
            with torch.no_grad():
                _, _, _, _, p = model(x, y, z, torch.full((n, 1), tv, device=device), Vrow.expand(n, -1),
                                      torch.full((n, 1), 20.0, device=device))
            assert (p - ur ** 2).abs().max().item() < 1e-5 * ur ** 2, f"p {p[0].item()} != U_ref^2 {ur ** 2} at t={tv}"
    import point_sampler as ps
    ps.FIXED_V = [3.0] * 8
    try:
        Ld = tg.doors_loss(model, device).item()
    finally:
        ps.FIXED_V = None
    assert abs(Ld - 1.0) < 1e-5, f"doors loss {Ld} != 1 for p_raw = 1 (p not measured on the U_ref^2 scale)"
    from throughflow import alpha_potential
    assert not gm.LEARN_ALPHA, "v24 expects LEARN_ALPHA = False"
    with torch.no_grad():
        model.alpha_head[2].bias.fill_(1.5)
        a_m = model.door_split(torch.full((3, 1), 60.0, device=device), Vs)
    a_p = alpha_potential(Vs).clamp(1e-4, 1 - 1e-4)
    assert (a_m - a_p).abs().max().item() < 1e-7, f"alpha {a_m.squeeze(1).tolist()} != potential {a_p.squeeze(1).tolist()}"
    print(f"  s(W1 1 m/s) = {s[0]:.3f} m/s (v9-v22: 0.071), s(closed) = 0, s(all 3 m/s) = {s[2]:.2f}; "
          f"p = U_ref^2 p_raw at t = 0 and 60 s; doors loss scaled; alpha fixed = potential ({a_m[0].item():.3f} for W1)")


@stage("0l. v25 NS on uniform points only -- first rows of an interior batch are uniform, NS uses exactly those")
def test_v25(device):
    import point_sampler as ps
    import train_gnot as tg
    assert tg.NS_UNIFORM_POINTS_ONLY and not ps.USE_PERSISTENT_POOL
    n = 4000
    nu_ = ps.interior_uniform_count(n)
    assert nu_ == n - int(round(n * ps.SOURCE_SAMPLE_FRAC)) and 0 < nu_ < n
    torch.manual_seed(1)
    x, y, z, t, V, N = ps.sample_interior(n, device)
    sx_u, sx_s = x[:nu_].std().item(), x[nu_:].std().item()
    assert 3.8 < sx_u < 5.2 and sx_s < 2.5, f"row order broken: x std uniform part {sx_u:.2f}, source part {sx_s:.2f}"
    from gnot_model import GNOTOperator
    torch.manual_seed(2)
    model = GNOTOperator().to(device)
    torch.manual_seed(3)
    tg.physics_loss(model, device)
    assert tg._NS_DIAG["n"] == ps.interior_uniform_count(tg.POINTS_INTERIOR), f"NS used {tg._NS_DIAG['n']} points"
    print(f"  interior batch: first {nu_}/{n} rows uniform (x std {sx_u:.2f} m vs source part {sx_s:.2f} m); "
          f"NS residual on {tg._NS_DIAG['n']} of {tg.POINTS_INTERIOR} points, CO2 on all")


@stage("0b. v8 non-dimensionalization -- no input saturation, training CO2 loss actually scaled, checkpoint guard")
def test_nondim(device):
    from gnot_model import GNOTOperator, check_checkpoint_compat
    from point_sampler import NUM_WINDOWS, T_MAX, N_PEOPLE_MAX, V_MAX, ROOM_X, ROOM_Y, ROOM_Z
    from train_gnot import trivial_co2_floor
    torch.manual_seed(0)
    model = GNOTOperator().to(device)
    n = 64

    pre = {}
    h1 = model.query_encoder.proj[0].register_forward_hook(lambda m, i, o: pre.__setitem__("query", o.detach()))
    h2 = model.token_encoder.proj[0].register_forward_hook(lambda m, i, o: pre.__setitem__("token", o.detach()))
    x = (torch.rand(n, 1, device=device) * (ROOM_X[1] - ROOM_X[0])).requires_grad_(True)
    y = (torch.rand(n, 1, device=device) * (ROOM_Y[1] - ROOM_Y[0])).requires_grad_(True)
    z = (torch.rand(n, 1, device=device) * (ROOM_Z[1] - ROOM_Z[0])).requires_grad_(True)
    t = torch.full((n, 1), T_MAX, device=device)
    V = torch.full((n, NUM_WINDOWS), V_MAX, device=device)
    N_people = torch.full((n, 1), N_PEOPLE_MAX, device=device)
    model(x, y, z, t, V, N_people)
    h1.remove(); h2.remove()
    def sat(p):
        return ((1 - torch.tanh(p) ** 2) < 0.05).float().mean().item()
    sat_q = sat(pre["query"])
    tok = pre["token"]
    sat_win = sat(tok[:, :NUM_WINDOWS])
    sat_occ = sat(tok[:, -1])
    print(f"  saturated first-layer units at t={T_MAX:.0f}s: {sat_q * 100:.1f}% (was ~88% before v8); "
          f"window tokens at V={V_MAX}: {sat_win * 100:.1f}%; constant occupancy token: "
          f"{sat_occ * 100:.1f}% (raw N=50 was ~91% before v8)")
    assert sat_q < 0.10, f"query encoder still {sat_q:.2%} saturated at t=T_MAX -- t scaling not applied?"
    assert sat_win < 0.10, f"window tokens {sat_win:.2%} saturated -- V/position scaling not applied?"
    assert sat_occ < 0.10, f"occupancy token {sat_occ:.2%} saturated"
    in_dim = model.query_encoder.proj[0].in_features
    expected_in = 2 * model.query_encoder.fourier.n_freq + 3
    assert in_dim == expected_in, f"query encoder input dim {in_dim}, expected {expected_in} (ramp feature missing?)"

    from train_gnot import physics_loss
    zero_c = GNOTOperator().to(device)
    with torch.no_grad():
        zero_c.out_head[-1].weight[3].zero_()
        zero_c.out_head[-1].bias[3].zero_()
    _, co2_zero = physics_loss(zero_c, device)
    from train_gnot import CO2_LOSS_AT_FULL_OCCUPANCY
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
    from point_sampler import C_REF
    with torch.no_grad():
        zero_c.out_head[-1].bias[3].fill_(1.0)
        _, _, _, C_one, _ = zero_c(x.detach(), y.detach(), z.detach(),
                                   torch.full((n, 1), 60.0, device=device), V, N_people)
    c_expected = C_REF * 60.0 / T_MAX * (N_people[0, 0].item() / N_PEOPLE_MAX)
    from throughflow import co2_window_factor
    from gnot_model import USE_CO2_WINDOW_FACTOR
    omega = co2_window_factor(x.detach(), y.detach(), V) if USE_CO2_WINDOW_FACTOR else 1.0
    max_dev = (C_one - c_expected * omega).abs().max().item()
    print(f"  output scaling: C_hat=1, t=60s -> C={C_one.mean().item():.4f} (expected C_REF*60/T_MAX={c_expected:.4f})")
    assert max_dev < 1e-5, f"C deviates from C_REF*t/T_MAX*omega by {max_dev:.2e} -- output scaling wrong"

    from train_gnot import co2_boundary_loss, _planar_wall_normal_derivative
    from point_sampler import sample_walls
    bc_const = co2_boundary_loss(zero_c, device, 1.0).item()
    print(f"  CO2 boundary loss for a constant CO2 field: {bc_const:.2e} (expected exactly 0)")
    assert bc_const < 1e-12, f"CO2 boundary loss is {bc_const:.2e} for a constant field -- should be 0"
    xw, yw, zw, _, _, _ = sample_walls(600, device)
    _, on_face = _planar_wall_normal_derivative(xw, yw, zw, xw, yw, zw)
    frac_on = on_face.float().mean().item()
    print(f"  wall points assigned to a face: {frac_on * 100:.1f}% (expected 100%)")
    assert frac_on == 1.0, f"only {frac_on:.3%} of wall points matched a face -- normal selection broken"
    bc_rand = co2_boundary_loss(model, device, 1.0).item()
    assert bc_rand == bc_rand and 0 < bc_rand < float("inf"), f"CO2 boundary loss not finite/positive: {bc_rand}"
    print(f"  CO2 boundary loss for the random-init model: {bc_rand:.4f} (finite, > 0)")
    del zero_c

    from gnot_model import MODEL_FORMAT_KEY, MODEL_FORMAT
    for old in ({"version": "v5_closed_window_fix"}, {"version": "v8_nondim", "nondim": True},
                {"version": "v9_zeroflow_bc", "nondim": True, "model_format": "v9_zeroflow"},
                {"version": "v10_hardic", "nondim": True, "model_format": "v10_hardic"},
                {"version": "v13_fullocc", "nondim": True, "model_format": "v12_linear_n"},
                {"version": "v19_throughflow", "nondim": True, "model_format": "v19_throughflow"},
                {"version": "v20_co2window", "nondim": True, "model_format": "v20_co2window"},
                {"version": "v21_single", "nondim": True, "model_format": "v21_single"},
                {"version": "v22_smoothjet", "nondim": True, "model_format": "v22_smoothjet"},
                {"version": "v23_scale_huber", "nondim": True, "model_format": "v23_scale_huber"}):
        try:
            check_checkpoint_compat(old, "fake_old.pth")
            raise AssertionError(f"check_checkpoint_compat accepted an old checkpoint: {old}")
        except RuntimeError:
            pass
    check_checkpoint_compat({"version": "v9", "nondim": True, MODEL_FORMAT_KEY: MODEL_FORMAT}, "fake_new.pth")
    print(f"  checkpoint guard: rejects v5, v8, v9, v10, v12-v18 and v19-v23, accepts {MODEL_FORMAT} -- OK")

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

    V_mix = torch.rand(n, NUM_WINDOWS, device=device) * V_MAX
    with torch.no_grad():
        _, _, _, C_t0, _ = model(xe, ye, ze, torch.zeros(n, 1, device=device), V_mix, Ne)
        _, _, _, C_t60, _ = model(xe, ye, ze, torch.full((n, 1), 60.0, device=device), V_mix, Ne)
    print(f"  hard IC: max|C(t=0)|={C_t0.abs().max().item():.2e} (must be 0), "
          f"max|C(t=60)|={C_t60.abs().max().item():.2e} (must be > 0)")
    assert C_t0.abs().max().item() == 0.0, "C(t=0) is not exactly 0 -- hard IC not applied"
    assert C_t60.abs().max().item() > 0.0, "C(t=60) is exactly 0 -- CO2 output is dead"

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
    A_10 = model(xe, ye, ze, t60, V_mix, torch.full((n, 1), 10.0, device=device))
    A_20 = model(xe, ye, ze, t60, V_mix, torch.full((n, 1), 20.0, device=device))
    u10 = model.velocity_from_potential(A_10[0], A_10[1], A_10[2], xe, ye, ze)
    u20 = model.velocity_from_potential(A_20[0], A_20[1], A_20[2], xe, ye, ze)
    flow_dev = max((a - b).abs().max().item() for a, b in zip(u10 + (A_10[4],), u20 + (A_20[4],)))
    flow_scale = max(q.abs().max().item() for q in u10 + (A_10[4],))
    print(f"  flow independent of N: max |(u,v,w,p)(N=10) - (u,v,w,p)(N=20)| = {flow_dev:.2e} "
          f"(flow scale {flow_scale:.2e}; must be ~0)")
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
    assert out.shape == (n, D_MODEL), f"expected ({n},{D_MODEL}), got {tuple(out.shape)}"
    assert_finite(out, "QueryEncoder output")

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
    """One training iteration, exactly as in train_gnot.main()."""
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

    from train_gnot import WALLS_WEIGHT, SKIP_IC_LOSS
    (WALLS_WEIGHT * walls_loss(model, device)).backward()
    windows_loss(model, device, co2_weight).backward()
    doors_loss(model, device).backward()
    if not SKIP_IC_LOSS:
        ic_loss(model, device, co2_weight).backward()
    from train_gnot import co2_boundary_loss
    co2_boundary_loss(model, device, co2_weight).backward()

    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_MAX_NORM)
    optimizer.step()


@stage("6. Full combined training steps (configured optimizer) -- weights change, state finite, resumable")
def test_one_training_step(device):
    """One full training step with the configured optimiser."""
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
    for st in optimizer.state.values():
        for v in st.values():
            for t in (v if isinstance(v, list) else [v]):
                if torch.is_tensor(t):
                    assert_finite(t, "an optimizer state tensor")
                    n_state += 1
    print(f"  optimizer={OPTIMIZER}: {n_changed}/{len(params)} parameter tensors changed after 3 steps "
          f"(expected: all of them); {n_state} state tensors finite")

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
    """Checks the learning-rate schedule (constant, then cosine decay)."""
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
    from train_gnot import lr_at, LR, LR_MIN, LR_DECAY_START, MAX_ITERS, RESUME_FROM, HERE, LR_EXP_DECAY
    from gnot_model import GNOTOperator, check_checkpoint_compat
    if LR_EXP_DECAY is not None:
        rate, steps = LR_EXP_DECAY
        assert lr_at(0) == LR, f"lr_at(0) = {lr_at(0)} != LR"
        assert abs(lr_at(steps) - LR * rate) < 1e-15 and abs(lr_at(10 * steps) - LR * rate ** 10) < 1e-15, \
            "exponential decay: lr_at(k*steps) must equal LR*rate**k"
        vals = [lr_at(i) for i in range(0, MAX_ITERS + 1, 100)]
        assert all(a > b for a, b in zip(vals, vals[1:])), "LR must decrease strictly"
        assert lr_at(MAX_ITERS) > 1e-6, f"final LR {lr_at(MAX_ITERS):.1e} is below 1e-6 (training would freeze)"
        print(f"  LR: {LR:g} * {rate}**(it/{steps}): {lr_at(10000):.2e} at 10k, {lr_at(20000):.2e} at 20k, "
              f"{lr_at(MAX_ITERS):.2e} at {MAX_ITERS}")
    elif LR_DECAY_START is None:
        assert all(lr_at(i) == LR for i in range(0, MAX_ITERS + 1, 500)), "LR should be constant"
        print(f"  LR: constant {LR:g} for all {MAX_ITERS} iterations (LR_DECAY_START=None)")
    else:
        _check_decay_schedule(lr_at, LR, LR_MIN, LR_DECAY_START, MAX_ITERS)

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
    test_v20(device)
    test_v21(device)
    test_v22(device)
    test_v22_weight(device)
    test_v23(device)
    test_v25(device)
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
