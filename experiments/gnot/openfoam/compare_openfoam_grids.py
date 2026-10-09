"""
Grid check of the OpenFOAM reference: the same case on two grids.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)


def load(case):
    from compare_with_openfoam import read_internal, time_dirs
    meta = dict(line.split(None, 1) for line in open(os.path.join(case, "scenario.txt")).read().splitlines())
    n = [int(v) for v in meta["n"].split()]
    C = read_internal(os.path.join(case, "0", "C"), 3)
    return meta, n, C, dict(time_dirs(case))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fine", required=True)
    ap.add_argument("--coarse", required=True)
    ap.add_argument("--times", type=float, nargs="*", default=[30.0, 60.0, 120.0])
    args = ap.parse_args()
    try:
        from point_sampler import ROOM_X, ROOM_Y, ROOM_Z, BREATHING_HEIGHT
    except ImportError:
        ROOM_X, ROOM_Y, ROOM_Z, BREATHING_HEIGHT = (0.0, 15.53), (0.0, 9.16), (0.0, 3.15), 1.10
    from compare_with_openfoam import read_internal, read_patch_sum
    fine, coarse = (os.path.abspath(os.path.expanduser(p)) for p in (args.fine, args.coarse))
    mf, nf, Cf, tf = load(fine)
    mc, nc, Cc, tc = load(coarse)
    L = [ROOM_X[1] - ROOM_X[0], ROOM_Y[1] - ROOM_Y[0], ROOM_Z[1] - ROOM_Z[0]]
    hc = [l / m for l, m in zip(L, nc)]
    lo = [ROOM_X[0], ROOM_Y[0], ROOM_Z[0]]
    ic = [np.clip(np.floor((Cc[:, a] - lo[a]) / hc[a]).astype(int), 0, nc[a] - 1) for a in range(3)]
    kz_f = np.abs(Cf[:, 2] - BREATHING_HEIGHT) < 0.5 * L[2] / nf[2] + 1e-9
    print(f"fine   {fine} ({len(Cf)} cells)\ncoarse {coarse} ({len(Cc)} cells)")
    print(f"{'t [s]':>6s} | {'volume':>7s} {'plane':>7s} | split fine  split coarse")
    for t in args.times:
        if t not in tf or t not in tc:
            print(f"{t:6.0f} | not saved in both cases -- skipped")
            continue
        Uf = read_internal(os.path.join(fine, tf[t], "U"), 3, len(Cf))
        Uc = read_internal(os.path.join(coarse, tc[t], "U"), 3, len(Cc))
        G = np.full(tuple(nc) + (3,), np.nan)
        G[ic[0], ic[1], ic[2]] = Uc
        for _ in range(4):
            nan = np.isnan(G[..., 0])
            if not nan.any():
                break
            acc = np.zeros_like(G)
            cnt = np.zeros(G.shape[:3])
            for ax in range(3):
                for s in (1, -1):
                    R = np.roll(G, s, axis=ax)
                    ok = ~np.isnan(R[..., 0])
                    acc[ok] += R[ok]
                    cnt[ok] += 1
            fill = nan & (cnt > 0)
            G[fill] = acc[fill] / cnt[fill][:, None]
        f = [np.clip((Cf[:, a] - lo[a]) / hc[a] - 0.5, 0, nc[a] - 1) for a in range(3)]
        i0 = [np.minimum(np.floor(fa).astype(int), nc[a] - 2) for a, fa in enumerate(f)]
        wgt = [fa - ia for fa, ia in zip(f, i0)]
        Ui = np.zeros_like(Uf)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    w = ((wgt[0] if di else 1 - wgt[0]) * (wgt[1] if dj else 1 - wgt[1]) * (wgt[2] if dk else 1 - wgt[2]))
                    Ui += w[:, None] * np.nan_to_num(G[i0[0] + di, i0[1] + dj, i0[2] + dk])
        e_v = np.linalg.norm(Ui - Uf) / np.linalg.norm(Uf)
        e_p = np.linalg.norm(Ui[kz_f] - Uf[kz_f]) / np.linalg.norm(Uf[kz_f])
        sp = []
        for case, td in ((fine, tf), (coarse, tc)):
            d1 = read_patch_sum(os.path.join(case, td[t], "phi"), "door1")
            d2 = read_patch_sum(os.path.join(case, td[t], "phi"), "door2")
            sp.append(d1 / (d1 + d2))
        print(f"{t:6.0f} | {100 * e_v:6.1f}% {100 * e_p:6.1f}% | {sp[0]:10.3f} {sp[1]:12.3f}")
    print("Reading: a few % = the fine OpenFOAM solution is grid-converged enough to serve as reference;"
          "\n         tens of % = the reference itself is still grid-dependent (refine, or compare trends only).")


if __name__ == "__main__":
    main()
