"""
Compares the GPU pilot (lbm_pilot.py) with OpenFOAM for the same setting: mean flow, CO2 at the last OpenFOAM
time, and run time. Optional figure of the breathing plane. Only reads the OpenFOAM case and the dataset file.
Usage: python3 compare_lbm.py --lbm data_lbm/N10_lbm_dx0.04.npz --case <OpenFOAM case> --data <dataset npz> --fig data_lbm/N10.png
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gnot_openfoam_rans")))
import common
from common import PPM_OUTDOOR


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lbm", required=True)
    ap.add_argument("--case", default=None, help="OpenFOAM case of the same setting (for its CO2 field)")
    ap.add_argument("--data", required=True, help="dataset file of the same setting (OpenFOAM mean flow)")
    ap.add_argument("--fig", default=None, help="png of the breathing plane")
    args = ap.parse_args()
    L = dict(np.load(args.lbm))
    D = dict(np.load(args.data))
    assert np.allclose(L["P"], D["P"]), "the two files are not on the same grid"
    assert np.allclose(L["V"], D["V"]), f"different settings: {L['V']} vs {D['V']}"
    pl = D["plane"]

    Ul, Uo = L["U_mean"], D["U_mean"]
    sl, so = np.linalg.norm(Ul, axis=1), np.linalg.norm(Uo, axis=1)
    print(f"mean flow 150-600 s (LBM vs OpenFOAM): mean speed {sl.mean():.3f} vs {so.mean():.3f} m/s, "
          f"velocity error {100 * rel(Ul, Uo):.1f} %, speed error {100 * rel(sl, so):.1f} %, "
          f"breathing plane speed error {100 * rel(sl[pl], so[pl]):.1f} %")

    ref, name = None, None
    if args.case:
        import check_co2_with_model_flow as L2
        from compare_with_openfoam import read_internal, time_dirs
        from compare_rans_pilot import load_case, to_grid
        meta, V, g, Cc, idx, td = load_case(os.path.abspath(args.case), read_internal, time_dirs, L2)
        t_end = max(t for t in td if t > 0)
        s = np.nan_to_num(to_grid(g, idx, read_internal(os.path.join(args.case, td[t_end], "s"), 1, len(Cc)).reshape(-1)))
        ref, name = s[g.fluid], f"OpenFOAM CO2 at {t_end:g} s"
    else:
        t_end = float(D["t_end_of"])
        k = int(np.argmin(np.abs(D["t_c"] - t_end)))
        ref, name = D["C"][k], f"dataset CO2 (our solver on the OpenFOAM flow) at {D['t_c'][k]:g} s"
    k = int(np.argmin(np.abs(L["t_snap"] - t_end)))
    cl = L["C_snap"][k]
    print(f"CO2 LBM at {L['t_snap'][k]:g} s vs {name}: error {100 * rel(cl, ref):.1f} %, total {cl.sum() / ref.sum():.3f}, "
          f"breathing plane {100 * rel(cl[pl], ref[pl]):.1f} %, paper metric (absolute ppm, plane) "
          f"{100 * rel(cl[pl] + PPM_OUTDOOR, ref[pl] + PPM_OUTDOOR):.1f} %")
    if "err_real_vs_OF" in D:
        print(f"  for comparison, our CO2 solver on the OpenFOAM flow vs OpenFOAM CO2: real flow "
              f"{100 * float(D['err_real_vs_OF']):.1f} %, dataset (B2) {100 * float(D['err_B2_vs_OF']):.1f} %, "
              f"paper metric {100 * float(D['abs_err_B2_z1p1']):.1f} %")
    print(f"LBM run: {int(L['steps'])} steps, {float(L['wall_s']) / 3600:.2f} h on {L['device']} for "
          f"{float(L['t_snap'][-1]):g} s (lattice {L['lbm_n'].tolist()}, dx {float(L['lbm_dx']):g} m)")

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        P = D["P"][pl]
        fig, ax = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        vmax = float(np.percentile(so[pl], 99))
        cmax = float(np.percentile(ref[pl], 99))
        for a, (v, title, vm, cm) in zip(ax.flat, [(so[pl], "OpenFOAM mean speed [m/s]", vmax, "viridis"),
                                                   (sl[pl], "LBM mean speed [m/s]", vmax, "viridis"),
                                                   (ref[pl], name + " [ppm]", cmax, "magma"),
                                                   (cl[pl], f"LBM CO2 at {L['t_snap'][k]:g} s [ppm]", cmax, "magma")]):
            im = a.scatter(P[:, 0], P[:, 1], c=v, s=6, marker="s", vmin=0, vmax=vm, cmap=cm)
            a.set_title(title)
            a.set_aspect("equal")
            fig.colorbar(im, ax=a)
        fig.suptitle(f"V = {L['V'].tolist()}, breathing plane z = {float(D['z_plane']):.2f} m")
        fig.savefig(args.fig, dpi=130)
        print(f"saved {args.fig}")


if __name__ == "__main__":
    main()
