"""
Extract one OpenFOAM case into a compact training dataset (npz):
  * velocity U from OpenFOAM at the dataset times T_DATA (fluid cells of the shared FV grid);
  * CO2 C at the same times, transported through the OpenFOAM flow by the verified FV solver of
    experiments/gnot/check_co2_with_model_flow.py (the SAME call as compare_with_openfoam.py, so the
    data equal the CO2 reference used in all Track-1 checks), at N_REF people. C is exactly linear in N.
Grid / cell mapping / readers are those of compare_with_openfoam.py (imported, not copied).

Usage (training env, from experiments/gnot_openfoam):
  python3 extract_case.py --case ../gnot/openfoam/cases/W1_1ms_dx0.1
Output: data/<case name>.npz  (~50-100 MB for dx = 0.1)
"""
import argparse
import os
import time

import numpy as np

import common
from common import DATA_DIR, N_REF, T_DATA


def extract(case, out_dir=DATA_DIR, times=T_DATA):
    import check_co2_with_model_flow as L2
    from compare_with_openfoam import read_internal, time_dirs
    from point_sampler import ROOM_X, ROOM_Y, BREATHING_HEIGHT
    assert abs(L2.N_PEOPLE - N_REF) < 1e-12, "FV solver occupancy must equal common.N_REF"

    case = os.path.abspath(os.path.expanduser(case))
    meta = dict(line.split(None, 1) for line in open(os.path.join(case, "scenario.txt")).read().splitlines())
    V = [float(v) for v in meta["V"].split()]
    dx = float(meta["dx"])
    nu = float(meta.get("nu", "0.01"))
    g = L2.Grid(dx, V)
    Cc = read_internal(os.path.join(case, "0", "C"), 3)
    idx = [np.clip(np.floor((Cc[:, a] - (ROOM_X[0], ROOM_Y[0], 0.0)[a]) / g.h[a]).astype(int), 0, g.n[a] - 1)
           for a in range(3)]
    occupied = np.zeros(g.X.shape, bool)
    occupied[idx[0], idx[1], idx[2]] = True
    mism = int(np.sum(occupied != g.fluid))
    assert mism < 0.01 * g.fluid.sum(), f"cell layouts do not match ({mism} cells) -- case made with another dx?"

    def to_grid(vals):
        out = np.full(g.X.shape + vals.shape[1:], np.nan)
        out[idx[0], idx[1], idx[2]] = vals
        return out

    tdirs = time_dirs(case)
    t_of = [t for t, _ in tdirs]
    missing = [t for t in times if t > 0 and not any(abs(t - s) < 1e-6 for s in t_of)]
    assert not missing, f"OpenFOAM did not save t = {missing}"
    # all saved OpenFOAM velocities (the FV transport interpolates between them)
    snaps, stimes, U_at = [], [], {}
    for t, d in tdirs:
        Ug = to_grid(read_internal(os.path.join(case, d, "U"), 3, len(Cc)))
        Ug = np.nan_to_num(Ug) * g.fluid[..., None]
        snaps.append([Ug[..., a] for a in range(3)])
        stimes.append(t)
        if any(abs(t - s) < 1e-6 for s in times):
            U_at[round(t, 6)] = Ug[g.fluid].astype(np.float32)
    if stimes[0] > 0:                      # at rest before the first save (as in compare_with_openfoam)
        snaps.insert(0, [np.zeros(g.X.shape)] * 3)
        stimes.insert(0, 0.0)
    U_at.setdefault(0.0, np.zeros((int(g.fluid.sum()), 3), np.float32))
    L2.T_SNAP = stimes
    L2.T_OUT = tuple(t for t in times if t > 0)
    t0 = time.time()
    ref, dt_used, nsteps = L2.solve_co2(g, snaps)
    print(f"FV CO2: {nsteps} steps, dt = {dt_used:.4f} s, {time.time() - t0:.0f} s wall")
    C_list = [np.zeros(int(g.fluid.sum()), np.float32) if t == 0 else ref[t][g.fluid].astype(np.float32)
              for t in times]
    U_list = [U_at[round(t, 6)] for t in times]
    P = np.stack([g.X[g.fluid], g.Y[g.fluid], g.Z[g.fluid]], 1).astype(np.float32)
    kz = int(np.argmin(np.abs(g.c1d[2] - BREATHING_HEIGHT)))
    plane = np.abs(P[:, 2] - g.c1d[2][kz]) < 1e-5
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.basename(case.rstrip("/")) + ".npz")
    np.savez_compressed(out, P=P, t=np.array(times, np.float32), U=np.stack(U_list), C=np.stack(C_list),
                        V=np.array(V, np.float32), N_ref=np.float32(N_REF), nu=np.float32(nu), dx=np.float32(dx),
                        plane=plane, z_plane=np.float32(g.c1d[2][kz]))
    print(f"saved {out}: {P.shape[0]} points x {len(times)} times, V = {V}, nu = {nu:g}, "
          f"breathing plane z = {g.c1d[2][kz]:.3f} m ({int(plane.sum())} points), {os.path.getsize(out) / 1e6:.1f} MB")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", default=DATA_DIR)
    args = ap.parse_args()
    extract(args.case, args.out)


if __name__ == "__main__":
    main()
