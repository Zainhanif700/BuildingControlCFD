"""
Derives EVERY geometry constant used in point_sampler.py directly from the
STL files, in one place, so none of it is untraceable hardcoding.

This closes a gap found in an audit: the earlier floor_plan.py only computed
X-ranges for doors/windows (used to confirm "2 doors, 8 windows" visually),
and the 4 columns' centers/radii were found via one-off commands typed
directly in a terminal session, never saved to a committed script. Anyone
re-deriving these numbers from scratch (e.g. a thesis committee, or you in
six months) should be able to run this ONE script and get everything.

Run: python3 derive_all_constants.py
"""
import os
import trimesh
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))


def get_full_bbox_clusters(fname, gap_tol=0.03):
    """Full (x,z) bounding box per physical opening (door or window),
    not just the x-range floor_plan.py originally computed."""
    m = trimesh.load(os.path.join(HERE, fname), process=False)
    m.merge_vertices()
    parts = m.split(only_watertight=False)
    boxes = sorted([(p.bounds[0][0], p.bounds[1][0], p.bounds[0][2], p.bounds[1][2]) for p in parts])
    merged = [list(boxes[0])]
    for xlo, xhi, zlo, zhi in boxes[1:]:
        if xlo <= merged[-1][1] + gap_tol:
            merged[-1][1] = max(merged[-1][1], xhi)
            merged[-1][2] = min(merged[-1][2], zlo)
            merged[-1][3] = max(merged[-1][3], zhi)
        else:
            merged.append([xlo, xhi, zlo, zhi])
    return merged


print("=" * 70)
print("ROOM BOUNDS")
print("=" * 70)
room = trimesh.load(os.path.join(HERE, "RoomVolume.stl"), process=False)
b = room.bounds
print(f"ROOM_X = ({max(0.0, b[0][0]):.2f}, {b[1][0]:.2f})")
print(f"ROOM_Y = ({max(0.0, b[0][1]):.2f}, {b[1][1]:.2f})")
print(f"ROOM_Z = ({max(0.0, b[0][2]):.2f}, {b[1][2]:.2f})")

print("\n" + "=" * 70)
print("DOORS (x_lo, x_hi, z_lo, z_hi)")
print("=" * 70)
doors = get_full_bbox_clusters("Doors.stl")
print(f"Found {len(doors)} doors:")
for d in doors:
    print(f"  ({d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f}, {d[3]:.2f})")

print("\n" + "=" * 70)
print("WINDOWS (x_lo, x_hi, z_lo, z_hi)")
print("=" * 70)
windows = get_full_bbox_clusters("Windows.stl")
print(f"Found {len(windows)} windows:")
for w in windows:
    print(f"  ({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}, {w[3]:.2f})")

print("\n" + "=" * 70)
print("COLUMNS (cx, cy, radius, z_lo, z_hi) -- via PyVista split_bodies(),")
print("same method Alexander's own geometry_utils.split_walls_and_obstacles() uses")
print("=" * 70)
walls_pv = pv.read(os.path.join(HERE, "RoomVolume_Walls.stl"))
bodies = walls_pv.split_bodies()

# Identify the outer wall shell by VOLUME (largest by far), not by assuming
# split_bodies() always returns it first -- that ordering isn't guaranteed
# to be stable across pyvista versions or re-exports of the STL.
volumes = [body.extract_surface().volume for body in bodies]
shell_idx = int(max(range(len(bodies)), key=lambda i: volumes[i]))
print(f"Found {len(bodies)} bodies total (body {shell_idx} = outer wall shell, by volume, "
      f"rest = columns/obstacles):")

n_columns = 0
for i, body in enumerate(bodies):
    bb = body.bounds
    bx, by, bz = bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4]
    c = body.center
    if i == shell_idx:
        print(f"  Body {i}: OUTER WALL SHELL (size {bx:.2f} x {by:.2f} x {bz:.2f}, "
              f"volume {volumes[i]:.1f}) -- not a column")
        continue
    is_column = bx < 1.0 and by < 1.0 and bz > 2.0
    # radius computed independently every time -- no special-casing toward
    # an expected answer, so this is a genuine check against point_sampler.py
    radius = max(bx, by) / 2.0 + 0.002  # small fudge for mesh wall thickness
    kind = "COLUMN" if is_column else "OTHER OBSTACLE (not a slender column -- check manually)"
    if is_column:
        n_columns += 1
    print(f"  Body {i}: ({c[0]:.2f}, {c[1]:.2f}, {radius:.3f}, {bb[4]:.2f}, {bb[5]:.2f}) -- {kind}")

assert n_columns == 4, (
    f"Expected exactly 4 columns, found {n_columns}. Geometry may have changed -- "
    f"do NOT silently trust point_sampler.py's hardcoded COLUMNS list until this is resolved."
)
print(f"\nConfirmed: exactly {n_columns} columns found (matches point_sampler.py's COLUMNS list length).")

print("\nDone. Compare this output against the hardcoded constants at the top")
print("of point_sampler.py -- they should match exactly.")
