"""
Load the real room geometry (STL files from Alexander's dt_pinn_training repo)
and check that it makes sense before we build GNOT's point sampler around it.

Prints bounding box of each part, and saves a 3D figure showing all four
parts together (room volume, walls, windows, doors) with different colors.
"""
import os
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))

PARTS = {
    "RoomVolume": ("RoomVolume.stl", "lightgray", 0.15),
    "RoomVolume_Walls": ("RoomVolume_Walls.stl", "saddlebrown", 0.5),
    "Windows": ("Windows.stl", "deepskyblue", 0.9),
    "Doors": ("Doors.stl", "orangered", 0.9),
}

def load_and_report(fname):
    path = os.path.join(HERE, fname)
    m = mesh.Mesh.from_file(path)
    pts = m.vectors.reshape(-1, 3)
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    print(f"{fname}:")
    print(f"  triangles = {m.vectors.shape[0]}")
    print(f"  bbox min (x,y,z) = {bbox_min}")
    print(f"  bbox max (x,y,z) = {bbox_max}")
    print(f"  size (dx,dy,dz)  = {bbox_max - bbox_min}")
    return m, bbox_min, bbox_max

meshes = {}
all_min = np.array([np.inf, np.inf, np.inf])
all_max = np.array([-np.inf, -np.inf, -np.inf])

for name, (fname, color, alpha) in PARTS.items():
    m, bmin, bmax = load_and_report(fname)
    meshes[name] = (m, color, alpha)
    all_min = np.minimum(all_min, bmin)
    all_max = np.maximum(all_max, bmax)
    print()

print(f"OVERALL bounding box: min={all_min}, max={all_max}, size={all_max - all_min}")

# --- 3D visualization ---
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection="3d")

for name, (m, color, alpha) in meshes.items():
    coll = Poly3DCollection(m.vectors, alpha=alpha, facecolor=color, edgecolor="k", linewidths=0.05)
    ax.add_collection3d(coll)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Real room geometry (from Alexander's repo)\ngray=room volume, brown=walls, blue=windows, red=doors")

# equal aspect ratio
max_range = (all_max - all_min).max() / 2.0
mid = (all_max + all_min) / 2.0
ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

out_path = os.path.join(HERE, "..", "figures", "room_geometry_check.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved figure to {out_path}")
