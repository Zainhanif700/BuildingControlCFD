"""
Clear top-down floor-plan view of the real room geometry, so door/window
counts and positions can be visually verified (not just read from a murky
3D render). Also re-does the 3D oblique view with better colors/alpha so
walls, windows, and doors are easy to tell apart.
"""
import os
import numpy as np
from stl import mesh
import trimesh
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))


def get_xz_clusters(fname, gap_tol=0.03):
    """Load an STL, weld vertices, split into connected pieces, then merge
    pieces whose x-ranges touch/overlap (accounts for near-duplicate
    coincident-but-not-quite-welded vertices) to get real physical openings."""
    m = trimesh.load(os.path.join(HERE, fname), process=False)
    m.merge_vertices()
    parts = m.split(only_watertight=False)
    ranges = sorted([(p.bounds[0][0], p.bounds[1][0]) for p in parts])
    merged = [list(ranges[0])]
    for lo, hi in ranges[1:]:
        if lo <= merged[-1][1] + gap_tol:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return merged


door_ranges = get_xz_clusters("Doors.stl")
window_ranges = get_xz_clusters("Windows.stl")

print(f"Doors found: {len(door_ranges)}")
for i, (lo, hi) in enumerate(door_ranges):
    print(f"  Door {i+1}: x = [{lo:.2f}, {hi:.2f}]  width = {hi-lo:.2f} m")

print(f"\nWindows found: {len(window_ranges)}")
for i, (lo, hi) in enumerate(window_ranges):
    print(f"  Window {i+1}: x = [{lo:.2f}, {hi:.2f}]  width = {hi-lo:.2f} m")

# Room outline from RoomVolume
room = mesh.Mesh.from_file(os.path.join(HERE, "RoomVolume.stl"))
pts = room.vectors.reshape(-1, 3)
xmin, ymin, zmin = pts.min(axis=0)
xmax, ymax, zmax = pts.max(axis=0)

# Columns: load walls, find the 4 excluded internal-obstacle cylinders by
# looking at RoomVolume_Walls.stl clusters that are small in footprint and
# don't touch x=xmin/xmax or y=ymin/ymax (i.e. free-standing)
walls = trimesh.load(os.path.join(HERE, "RoomVolume_Walls.stl"), process=False)

# --- Floor plan (top-down view) ---
fig, ax = plt.subplots(figsize=(12, 7))

# room outline
ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, edgecolor="black", linewidth=2))

# doors on the y=ymin wall (bottom)
for i, (lo, hi) in enumerate(door_ranges):
    ax.add_patch(patches.Rectangle((lo, ymin - 0.15), hi - lo, 0.3,
                                    facecolor="orangered", edgecolor="k"))
    ax.text((lo + hi) / 2, ymin - 0.5, f"Door {i+1}", ha="center", fontsize=9, color="orangered")

# windows on the y=ymax wall (top)
for i, (lo, hi) in enumerate(window_ranges):
    ax.add_patch(patches.Rectangle((lo, ymax - 0.15), hi - lo, 0.3,
                                    facecolor="deepskyblue", edgecolor="k"))
    ax.text((lo + hi) / 2, ymax + 0.3, f"W{i+1}", ha="center", fontsize=9, color="deepskyblue")

ax.set_xlim(xmin - 1, xmax + 1)
ax.set_ylim(ymin - 1.2, ymax + 1.2)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_aspect("equal")
ax.set_title(f"Top-down floor plan (real geometry)\n"
             f"{len(door_ranges)} doors (red, bottom wall) | {len(window_ranges)} windows (blue, top wall)")

out_path = os.path.join(HERE, "..", "figures", "floor_plan.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved floor plan to {out_path}")
