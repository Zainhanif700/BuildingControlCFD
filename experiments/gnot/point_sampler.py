"""
Room geometry (measured from the STL files) and the random training points for the physics-only model.
"""
import math

import torch
import numpy as np

ROOM_X = (0.0, 15.53)
ROOM_Y = (0.0, 9.16)
ROOM_Z = (0.0, 3.15)

CO2_SOURCE_SIGMA = 2.5
BREATHING_HEIGHT = 1.10

DOORS = [
    (1.59, 2.65, 0.0, 2.11),
    (12.32, 13.38, 0.0, 2.11),
]

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

COLUMNS = [
    (13.85, 8.44, 0.285, 0.0, 3.15),
    (5.74, 0.45, 0.285, 0.0, 3.15),
    (5.78, 8.50, 0.285, 0.0, 3.15),
    (13.83, 0.35, 0.285, 0.0, 3.15),
]

V_MIN, V_MAX = 0.0, 5.0
N_PEOPLE_MIN, N_PEOPLE_MAX = 0.0, 50.0
T_MIN, T_MAX = 0.0, 120.0

EMISSION_PER_PERSON = 1.15e-4

TAU_RAMP = 2.0

S_REF = N_PEOPLE_MAX * EMISSION_PER_PERSON
C_REF = S_REF * T_MAX


def _rand(n, lo, hi, device):
    return torch.rand(n, 1, device=device) * (hi - lo) + lo


def _in_any_column(x, y):
    """Boolean mask: True where (x,y) falls inside one of the 4 columns."""
    inside = torch.zeros_like(x, dtype=torch.bool)
    for cx, cy, r, _, _ in COLUMNS:
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        inside = inside | (d2 <= r ** 2)
    return inside


CLOSED_SCENARIO_FRAC = 0.3
PARTIAL_CLOSED_SCENARIO_FRAC = 0.2


def sample_scenario(n, device):
    """Random scenario values (time, window speeds, number of people) for training."""
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
    V = V[torch.randperm(n, device=device)]
    if FIXED_V is not None:
        V = torch.tensor(FIXED_V, device=device, dtype=V.dtype).view(1, -1).expand(n, -1).clone()

    return t, V, N_people


FIXED_V = None


SOURCE_X = (ROOM_X[0] + ROOM_X[1]) / 2
SOURCE_Y = (ROOM_Y[0] + ROOM_Y[1]) / 2

SOURCE_SAMPLE_FRAC = 0.6
SOURCE_SAMPLE_XY_STD = CO2_SOURCE_SIGMA / 2.0

SOURCE_SAMPLE_Z_STD = min(SOURCE_SAMPLE_XY_STD, (ROOM_Z[1] - ROOM_Z[0]) / 4.0)


def _sample_near_source(n, device):
    """Random points concentrated around the CO2 source."""
    pts = []
    remaining = n
    while remaining > 0:
        batch = max(remaining * 2, 256)
        x = torch.normal(SOURCE_X, SOURCE_SAMPLE_XY_STD, size=(batch, 1), device=device)
        y = torch.normal(SOURCE_Y, SOURCE_SAMPLE_XY_STD, size=(batch, 1), device=device)
        z = torch.normal(BREATHING_HEIGHT, SOURCE_SAMPLE_Z_STD, size=(batch, 1), device=device)
        outside = ((x < ROOM_X[0]) | (x > ROOM_X[1]) | (y < ROOM_Y[0]) | (y > ROOM_Y[1])
                   | (z < ROOM_Z[0]) | (z > ROOM_Z[1]))
        bad = _in_any_column(x, y) | outside
        keep = ~bad.squeeze(-1)
        x, y, z = x[keep], y[keep], z[keep]
        pts.append(torch.cat([x, y, z], dim=1))
        remaining -= x.shape[0]
    xyz = torch.cat(pts, dim=0)[:n]
    return xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]


USE_PERSISTENT_POOL = False
POOL_REFRESH_FRAC = 0.2
POOL_REFRESH_EVERY = 100

_interior_pools = {}


def _sample_uniform_xyz(n, device):
    """(n, 3) points uniform in the room, excluding the columns (rejection sampling)."""
    if n <= 0:
        return torch.empty(0, 3, device=device)
    pts = []
    remaining = n
    while remaining > 0:
        batch = max(remaining * 2, 256)
        x = _rand(batch, *ROOM_X, device)
        y = _rand(batch, *ROOM_Y, device)
        z = _rand(batch, *ROOM_Z, device)
        keep = ~_in_any_column(x, y).squeeze(-1)
        x, y, z = x[keep], y[keep], z[keep]
        pts.append(torch.cat([x, y, z], dim=1))
        remaining -= x.shape[0]
    return torch.cat(pts, dim=0)[:n]


def interior_uniform_count(n):
    """Number of uniform points in an interior batch of n."""
    return n - int(round(n * SOURCE_SAMPLE_FRAC))


def _generate_interior_batch(n, device):
    """Random interior points: a mix of uniform points and points near the source."""
    n_uniform = interior_uniform_count(n)
    n_source = n - n_uniform
    xyz_uniform = _sample_uniform_xyz(n_uniform, device)

    if n_source > 0:
        xs, ys, zs = _sample_near_source(n_source, device)
        xyz_source = torch.cat([xs, ys, zs], dim=1)
        xyz = torch.cat([xyz_uniform, xyz_source], dim=0)
    else:
        xyz = xyz_uniform

    t, V, N_people = sample_scenario(n, device)
    return xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3], t, V, N_people


def interior_pool_composition(n, device="cpu"):
    """Diagnostic: what the current point pool contains."""
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
    """Clears all persistent interior pools."""
    _interior_pools.clear()


def sample_interior(n, device="cpu"):
    """Random points inside the room, excluding the 4 columns (rejection sampling)."""
    if not USE_PERSISTENT_POOL:
        return _generate_interior_batch(n, device)

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

    return (pool["x"].clone().detach(), pool["y"].clone().detach(),
            pool["z"].clone().detach(), pool["t"].clone().detach(),
            pool["V"].clone().detach(), pool["N_people"].clone().detach())


def sample_walls(n, device="cpu"):
    """No-slip points on the walls, floor and ceiling, outside the openings and columns."""
    Lx, Ly, Lz = ROOM_X[1] - ROOM_X[0], ROOM_Y[1] - ROOM_Y[0], ROOM_Z[1] - ROOM_Z[0]
    col_area = sum(math.pi * r ** 2 for _, _, r, _, _ in COLUMNS)

    def in_openings(x, z, openings):
        bad = torch.zeros_like(x, dtype=torch.bool)
        for xlo, xhi, zlo, zhi in openings:
            bad = bad | ((x >= xlo) & (x <= xhi) & (z >= zlo) & (z <= zhi))
        return bad

    faces = [
        (Lx * Lz - sum((a1 - a0) * (b1 - b0) for a0, a1, b0, b1 in DOORS),
         lambda m: (_rand(m, *ROOM_X, device), None, _rand(m, *ROOM_Z, device), ("y", ROOM_Y[0])),
         lambda x, y, z: in_openings(x, z, DOORS)),
        (Lx * Lz - sum((a1 - a0) * (b1 - b0) for a0, a1, b0, b1 in WINDOWS),
         lambda m: (_rand(m, *ROOM_X, device), None, _rand(m, *ROOM_Z, device), ("y", ROOM_Y[1])),
         lambda x, y, z: in_openings(x, z, WINDOWS)),
        (Ly * Lz, lambda m: (None, _rand(m, *ROOM_Y, device), _rand(m, *ROOM_Z, device), ("x", ROOM_X[0])),
         lambda x, y, z: torch.zeros_like(x, dtype=torch.bool)),
        (Ly * Lz, lambda m: (None, _rand(m, *ROOM_Y, device), _rand(m, *ROOM_Z, device), ("x", ROOM_X[1])),
         lambda x, y, z: torch.zeros_like(x, dtype=torch.bool)),
        (Lx * Ly - col_area, lambda m: (_rand(m, *ROOM_X, device), _rand(m, *ROOM_Y, device), None, ("z", ROOM_Z[0])),
         lambda x, y, z: _in_any_column(x, y)),
        (Lx * Ly - col_area, lambda m: (_rand(m, *ROOM_X, device), _rand(m, *ROOM_Y, device), None, ("z", ROOM_Z[1])),
         lambda x, y, z: _in_any_column(x, y)),
    ]
    areas = [f[0] for f in faces]
    total = sum(areas)
    counts = [int(n * a / total) for a in areas]
    order = sorted(range(len(faces)), key=lambda i: n * areas[i] / total - counts[i], reverse=True)
    for i in order[: n - sum(counts)]:
        counts[i] += 1

    all_x, all_y, all_z = [], [], []
    for (area, gen, reject), cnt in zip(faces, counts):
        got = []
        remaining = cnt
        while remaining > 0:
            m = max(remaining * 2, 64)
            x, y, z, (axis, val) = gen(m)
            ref = x if x is not None else y
            if axis == "x":
                x = torch.full_like(ref, val)
            elif axis == "y":
                y = torch.full_like(ref, val)
            else:
                z = torch.full_like(ref, val)
            keep = ~reject(x, y, z).squeeze(-1)
            got.append(torch.cat([x[keep], y[keep], z[keep]], dim=1))
            remaining -= int(keep.sum())
        xyz = torch.cat(got, dim=0)[:cnt]
        all_x.append(xyz[:, 0:1]); all_y.append(xyz[:, 1:2]); all_z.append(xyz[:, 2:3])

    x = torch.cat(all_x, dim=0)
    y = torch.cat(all_y, dim=0)
    z = torch.cat(all_z, dim=0)
    t, V, N_people = sample_scenario(x.shape[0], device)
    return x, y, z, t, V, N_people


def sample_columns_surface(n_per_column, device="cpu"):
    """No-slip points on the 4 columns' curved (cylindrical) side surfaces."""
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
    """Inflow points, split evenly across the 8 real windows."""
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
    """Initial-condition points (t = 0, room at rest), excluding the columns."""
    xyz = _sample_uniform_xyz(n, device)
    _, V, N_people = sample_scenario(n, device)
    x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
    t = torch.zeros_like(x)
    return x, y, z, t, V, N_people


if __name__ == "__main__":
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
