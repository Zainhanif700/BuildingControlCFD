"""
Mesh study: the same window setting on several meshes (e.g. 15, 10 and 8 cm), compared with the finest one.
Compares the mean flow (150-600 s), OpenFOAM's own CO2 at 600 s, the door split and y+.
Usage: python3 compare_meshes.py cases/S02_mesh_dx0.15 cases/S02_mesh_dx0.1 cases/S02_mesh_dx0.08
"""
import argparse
import os
import re

import numpy as np

import common
from compare_with_openfoam import read_internal, read_patch_sum, time_dirs

T_AVG, T_END = 150.0, 600.0


def load(case):
    """Grid, mean velocity (150-600 s), CO2 at 600 s and log numbers of one finished case."""
    meta = dict(l.split(None, 1) for l in open(os.path.join(case, "scenario.txt")).read().splitlines() if l.strip())
    n = [int(v) for v in meta["n"].split()]
    bm = open(os.path.join(case, "system", "blockMeshDict")).read()
    verts = np.array(re.findall(r"\(([-0-9.eE+]+) ([-0-9.eE+]+) ([-0-9.eE+]+)\)", bm.split("blocks")[0]), float)
    lo, hi = verts.min(0), verts.max(0)
    h = (hi - lo) / n
    c1d = [lo[a] + (np.arange(n[a]) + 0.5) * h[a] for a in range(3)]
    Cc = read_internal(os.path.join(case, "0", "C"), 3)
    idx = tuple(np.clip(np.floor((Cc[:, a] - lo[a]) / h[a]).astype(int), 0, n[a] - 1) for a in range(3))
    td = dict(time_dirs(case))
    avg_t = [t for t in td if T_AVG <= t <= T_END + 1e-6]
    assert T_END in td, f"{case}: no results at {T_END:g} s (run not finished?)"
    U = np.mean([read_internal(os.path.join(case, td[t], "U"), 3, len(Cc)) for t in avg_t], axis=0)
    s = read_internal(os.path.join(case, td[T_END], "s"), 1, len(Cc))[:, 0]
    fluid = np.zeros(n, bool)
    fluid[idx] = True

    def grid(v):                       # cell values -> full grid, columns filled from their fluid neighbours
        out = np.full(tuple(n) + v.shape[1:], np.nan)
        out[idx] = v
        return fill(out, fluid)
    phi_t = max((t for t in td if os.path.exists(os.path.join(case, td[t], "phi")) or
                 os.path.exists(os.path.join(case, td[t], "phi.gz"))), default=None)
    split = None
    if phi_t is not None:
        d1, d2 = (read_patch_sum(os.path.join(case, td[phi_t], "phi"), f"door{j}") for j in (1, 2))
        split = (phi_t, d1 / (d1 + d2))
    yplus = {}
    logp = os.path.join(case, "log.pimpleFoam")
    if os.path.isfile(logp):
        for p, mn, mx, av in re.findall(r"patch (\w+) y\+ : min = ([0-9.eE+-]+), max = ([0-9.eE+-]+), average = ([0-9.eE+-]+)",
                                        open(logp).read()):
            yplus[p] = (float(mn), float(mx), float(av))
    vol = float(np.prod(h))
    return dict(case=case, dx=float(meta["dx"]), n=n, cells=len(Cc), c1d=c1d, U=grid(U), s=grid(s[:, None])[..., 0],
                speed=float(np.linalg.norm(U, axis=1).mean()), co2_total=float(s.sum() * vol), split=split, yplus=yplus)


def fill(a, fluid):
    """Fill the solid (column) cells with the mean of their already known neighbours, step by step inwards."""
    a = a.copy()
    known = fluid.copy()
    while not known.all():
        acc = np.zeros(a.shape)
        cnt = np.zeros(fluid.shape)
        for ax in range(3):
            for s in (-1, 1):
                k = np.roll(known, s, ax)
                v = np.roll(np.where(known[(...,) + (None,) * (a.ndim - 3)], a, 0.0), s, ax)
                acc += v
                cnt += k
        new = ~known & (cnt > 0)
        if not new.any():
            break
        a[new] = (acc[new].T / cnt[new]).T if a.ndim > 3 else acc[new] / cnt[new]
        known |= new
    return a


def interp(c1d, f, P):
    """Trilinear interpolation of a cell-centred grid field f at points P (values outside: nearest cell)."""
    w, i0 = [], []
    for a in range(3):
        h = c1d[a][1] - c1d[a][0]
        x = np.clip((P[:, a] - c1d[a][0]) / h, 0, len(c1d[a]) - 1)
        i = np.minimum(np.floor(x).astype(int), len(c1d[a]) - 2)
        i0.append(i)
        w.append(x - i)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                wt = (w[0] if dx else 1 - w[0]) * (w[1] if dy else 1 - w[1]) * (w[2] if dz else 1 - w[2])
                v = f[i0[0] + dx, i0[1] + dy, i0[2] + dz]
                out = out + (wt[:, None] * v if v.ndim > 1 else wt * v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+", help="the same window setting on different meshes; the finest is the reference")
    args = ap.parse_args()
    runs = sorted((load(c) for c in args.cases), key=lambda r: -r["dx"])
    ref = runs[-1]
    coarse = runs[0]
    # comparison points: the cell centres of the coarsest mesh (volume) and its planes nearest 1.1 m and 1.6 m
    X, Y, Z = np.meshgrid(*coarse["c1d"], indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    cols = [(13.85, 8.44), (5.74, 0.45), (5.78, 8.50), (13.83, 0.35)]
    away = np.all([(P[:, 0] - cx) ** 2 + (P[:, 1] - cy) ** 2 > 0.45 ** 2 for cx, cy in cols], axis=0)   # not next to a column
    planes = {z: away & (np.abs(P[:, 2] - coarse["c1d"][2][np.argmin(np.abs(coarse["c1d"][2] - z))]) < 1e-6) for z in (1.1, 1.6)}
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    Ur, sr = interp(ref["c1d"], ref["U"], P), interp(ref["c1d"], ref["s"], P)
    print(f"reference: {ref['case']} (dx {ref['dx']:g} m)\n")
    print(f"{'mesh':>6s} {'cells':>9s} | {'mean speed':>10s} | {'velocity vs finest':>21s} | {'CO2 at 600 s vs finest':>34s} | {'CO2 total':>9s} | door 1 share")
    print(f"{'dx [m]':>6s} {'':>9s} | {'[m/s]':>10s} | {'volume':>9s} {'1.6 m':>11s} | {'excess 1.6 m':>14s} {'paper 1.6 m':>11s} {'volume':>7s} | {'vs finest':>9s} |")
    for r in runs:
        U, s = interp(r["c1d"], r["U"], P), interp(r["c1d"], r["s"], P)
        pl = planes[1.6]
        ev, ep = rel(U[away], Ur[away]), rel(U[pl], Ur[pl])
        es, ea = rel(s[pl], sr[pl]), rel(s[pl] + common.PPM_OUTDOOR, sr[pl] + common.PPM_OUTDOOR)
        esv = rel(s[away], sr[away])
        sp = f"{r['split'][1]:.3f} ({r['split'][0]:g} s)" if r["split"] else "-"
        print(f"{r['dx']:6g} {r['cells']:9,d} | {r['speed']:10.4f} | {100 * ev:8.1f}% {100 * ep:10.1f}% | "
              f"{100 * es:13.1f}% {100 * ea:10.2f}% {100 * esv:6.1f}% | {r['co2_total'] / ref['co2_total']:9.3f} | {sp}")
    print("\ny+ at the last write (patch: min / max / average):")
    for r in runs:
        print(f"  dx {r['dx']:g}: " + "; ".join(f"{p} {v[0]:.0f}/{v[1]:.0f}/{v[2]:.0f}" for p, v in r["yplus"].items()
                                              if p in ("walls", "columns")))
    print("\nReading: small differences between the 10 cm mesh and the finest one (a few %) mean the 10 cm results are"
          "\nmesh-independent enough; the coarse mesh shows how fast the results converge.")


if __name__ == "__main__":
    main()
