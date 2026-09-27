"""
LEVEL 1 comparison: physics-informed GNOT vs an independent OpenFOAM solution of the SAME
equations (case made by make_openfoam_case.py, run by run_openfoam_case.sh).

Compares, for one window scenario:
  1. AIRFLOW  -- relative L2 error of the velocity (whole room and breathing plane z ~ 1.1 m)
                 at the chosen times; speed maps side by side.
  2. DOORS    -- OpenFOAM flow rate through each door and its split vs the model's alpha.
  3. CO2      -- CO2 transported through the OPENFOAM flow with the verified finite-volume
                 solver of check_co2_with_model_flow.py (same source, D, BCs; OpenFOAM's
                 conda build has no compiler for a coded Gaussian source), compared with the
                 model's CO2. A network-free reference for the open-window CO2.
Both sides use the identical cell layout (make_openfoam_case.py uses Grid's n = round(L/dx)).

Usage (training env, i.e. with torch):
  python3 compare_with_openfoam.py --case cases/W1_1ms_dx0.1 --checkpoint <model .pth>
  # checkpoint of an OLDER model version: point --code-dir at that version's code, e.g.
  python3 compare_with_openfoam.py --case cases/W1_1ms_dx0.1 \
      --checkpoint ~/BuildingControlCFD/experiments/gnot/checkpoints/v19_throughflow/gnot_v19_throughflow_final.pth \
      --code-dir ~/gnot_v19/experiments/gnot
Writes openfoam/results/<case>__<version>/ (log table, csv, figures).
"""
import argparse
import csv
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ OpenFOAM ascii readers
def _read_list_block(text, key, ncomp):
    """Parse 'key nonuniform List<...> N ( ... )' (or 'uniform v') -> (N, ncomp) array."""
    m = re.search(key + r"\s+nonuniform\s+List<\w+>\s*(\d+)\s*\(", text)
    if m is None:
        return None
    n = int(m.group(1))
    body = text[m.end():]
    end = body.index("\n)")
    arr = np.array(body[:end].replace("(", " ").replace(")", " ").split(), dtype=float)
    assert arr.size == n * ncomp, f"{key}: expected {n}x{ncomp} values, got {arr.size}"
    return arr.reshape(n, ncomp)


def read_internal(path, ncomp, n_cells=None):
    """internalField as (N, ncomp); a 'uniform' field (e.g. the initial state at rest in 0/)
    is expanded to n_cells rows."""
    with open(path) as f:
        text = f.read()
    arr = _read_list_block(text, "internalField", ncomp)
    if arr is not None:
        return arr
    m = re.search(r"internalField\s+uniform\s+\(?([^;)]*)\)?\s*;", text)
    if m is None or n_cells is None:
        raise ValueError(f"{path}: no readable internalField")
    val = np.array(m.group(1).split(), dtype=float).reshape(1, ncomp)
    return np.repeat(val, n_cells, axis=0)


def read_patch_sum(path, patch):
    """Sum of a surface field (phi) over one boundary patch."""
    with open(path) as f:
        text = f.read()
    m = re.search(r"\n\s*" + re.escape(patch) + r"\s*\n\s*\{", text)
    if m is None:
        return float("nan")
    block = text[m.end(): text.index("}", m.end())]
    vals = _read_list_block(block, "value", 1)
    if vals is None:
        mu = re.search(r"value\s+uniform\s+([-0-9.eE+]+)", block)
        return 0.0 if mu is None else float(mu.group(1))
    return float(vals.sum())


def time_dirs(case):
    out = []
    for d in os.listdir(case):
        try:
            t = float(d)
        except ValueError:
            continue
        if d != "0.orig" and os.path.isfile(os.path.join(case, d, "U")):
            out.append((t, d))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--code-dir", default=os.path.dirname(HERE),
                    help="experiments/gnot folder whose model code matches the checkpoint")
    ap.add_argument("--times", type=float, nargs="*", default=[30.0, 60.0, 120.0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-co2", action="store_true", help="skip the CO2 transport (faster)")
    args = ap.parse_args()
    code_dir = os.path.abspath(os.path.expanduser(args.code_dir))
    sys.path.insert(0, code_dir)
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gnot_model import GNOTOperator, check_checkpoint_compat
    from point_sampler import BREATHING_HEIGHT, ROOM_X, ROOM_Y
    from check_physics_consistency import evaluate
    import check_co2_with_model_flow as L2

    case = os.path.abspath(os.path.expanduser(args.case))
    meta = dict(line.split(None, 1) for line in open(os.path.join(case, "scenario.txt")).read().splitlines())
    V = [float(v) for v in meta["V"].split()]
    dx = float(meta["dx"])
    name = meta["name"]
    ckpt_path = os.path.abspath(os.path.expanduser(args.checkpoint))
    dev = args.device
    ckpt = torch.load(ckpt_path, map_location=dev)
    check_checkpoint_compat(ckpt, ckpt_path)
    model = GNOTOperator().to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    version = ckpt.get("version", "unknown")
    out_dir = os.path.join(HERE, "results", f"{os.path.basename(case)}__{version}")
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "comparison.log"), "w")

    def say(s=""):
        print(s, flush=True)
        log.write(s + "\n")
    say(f"OpenFOAM case {case}\n  scenario {name}: V = {V}, dx = {dx}\nmodel {ckpt_path} "
        f"(version={version}, iter={ckpt.get('iter', '?')}), code from {code_dir}")

    # ---- grid mapping: OpenFOAM cells -> (i, j, k) of the shared layout
    g = L2.Grid(dx, V)
    C = read_internal(os.path.join(case, "0", "C"), 3)
    idx = [np.clip(np.floor((C[:, a] - (ROOM_X[0], ROOM_Y[0], 0.0)[a]) / g.h[a]).astype(int), 0, g.n[a] - 1)
           for a in range(3)]
    occupied = np.zeros(g.X.shape, bool)
    occupied[idx[0], idx[1], idx[2]] = True
    mism = np.sum(occupied != g.fluid)
    say(f"grid: {len(C)} OpenFOAM cells on {g.n[0]}x{g.n[1]}x{g.n[2]}; fluid-mask mismatch {mism} cells")
    assert mism < 0.01 * g.fluid.sum(), "cell layouts do not match -- was the case made with the same dx?"

    def to_grid(vals):
        out = np.full(g.X.shape + vals.shape[1:], np.nan)
        out[idx[0], idx[1], idx[2]] = vals
        return out

    tdirs = time_dirs(case)
    tmax = max(t for t, _ in tdirs)
    say(f"saved OpenFOAM times: {len(tdirs)} (up to t = {tmax:g} s)")
    P = np.stack([g.X[g.fluid], g.Y[g.fluid], g.Z[g.fluid]], 1)
    kz = int(np.argmin(np.abs(g.c1d[2] - BREATHING_HEIGHT)))

    # ---- 1 + 2: airflow and doors
    rows = []
    say("\nAIRFLOW: model vs OpenFOAM (relative L2 of the velocity vector)")
    say(f"{'t [s]':>6s} | {'volume':>7s} {'plane':>7s} | {'|u| OF':>7s} {'|u| model':>9s} | "
        f"door1 / door2 [m^3/s] OF   split OF  alpha model")
    plane_maps = {}
    for t in args.times:
        match = [d for tt, d in tdirs if abs(tt - t) < 1e-6]
        if not match:
            say(f"{t:6.0f} | not saved by OpenFOAM (yet) -- skipped")
            continue
        U_of = to_grid(read_internal(os.path.join(case, match[0], "U"), 3, len(C)))
        f = evaluate(model, dev, P, t, V, 20.0)
        U_m = np.full(U_of.shape, np.nan)
        for a, key in enumerate(("u", "v", "w")):
            tmp = np.full(g.X.shape, np.nan)
            tmp[g.fluid] = f[key]
            U_m[..., a] = tmp
        fl = g.fluid
        e_vol = np.sqrt(np.nansum((U_m[fl] - U_of[fl]) ** 2) / np.nansum(U_of[fl] ** 2))
        pl = fl[:, :, kz]
        e_pl = np.sqrt(np.nansum((U_m[:, :, kz][pl] - U_of[:, :, kz][pl]) ** 2) / np.nansum(U_of[:, :, kz][pl] ** 2))
        s_of = np.nanmean(np.linalg.norm(U_of[fl], axis=1))
        s_m = np.nanmean(np.linalg.norm(U_m[fl], axis=1))
        phi_path = os.path.join(case, match[0], "phi")
        d1 = read_patch_sum(phi_path, "door1") if os.path.isfile(phi_path) else float("nan")
        d2 = read_patch_sum(phi_path, "door2") if os.path.isfile(phi_path) else float("nan")
        with torch.no_grad():
            alpha = model.door_split(torch.full((1, 1), t, device=dev),
                                     torch.tensor([V], dtype=torch.float32, device=dev)).item()
        split = d1 / (d1 + d2) if (d1 + d2) != 0 else float("nan")
        say(f"{t:6.0f} | {100 * e_vol:6.1f}% {100 * e_pl:6.1f}% | {s_of:7.3f} {s_m:9.3f} | "
            f"{d1:7.3f} / {d2:7.3f}   {split:6.3f}     {alpha:6.3f}")
        rows.append({"t": t, "vel_rel_L2_volume": e_vol, "vel_rel_L2_plane": e_pl, "mean_speed_OF": s_of,
                     "mean_speed_model": s_m, "door1_OF": d1, "door2_OF": d2, "split_OF": split,
                     "alpha_model": alpha})
        plane_maps[t] = (np.linalg.norm(U_of[:, :, kz], axis=-1), np.linalg.norm(U_m[:, :, kz], axis=-1),
                         U_of[:, :, kz], U_m[:, :, kz])

    # ---- 3: CO2 through the OpenFOAM flow (network-free reference) vs model CO2
    if not args.no_co2:
        say("\nCO2: model vs finite-volume CO2 transported by the OPENFOAM flow (N = 20)")
        snaps, times = [], []
        for t, d in tdirs:
            Ug = to_grid(read_internal(os.path.join(case, d, "U"), 3, len(C)))
            snaps.append([np.nan_to_num(Ug[..., a]) * g.fluid for a in range(3)])
            times.append(t)
        if times[0] > 0:                                   # at rest before the first save
            snaps.insert(0, [np.zeros(g.X.shape)] * 3)
            times.insert(0, 0.0)
        L2.T_SNAP = times                                  # velocity_at() interpolates on these
        L2.T_OUT = tuple(t for t in args.times if t <= times[-1])
        ref, dt_used, nsteps = L2.solve_co2(g, snaps)
        say(f"  FV transport: {nsteps} steps, dt = {dt_used:.4f} s, {len(times)} OpenFOAM snapshots")
        say(f"{'t [s]':>6s} | {'plane L2':>8s} {'volume L2':>9s} | {'mass model/ref':>14s}")
        for t in L2.T_OUT:
            c_m = evaluate(model, dev, P, t, V, 20.0)["c"]
            c_r = ref[t][g.fluid]
            pm = np.full(g.X.shape, np.nan); pm[g.fluid] = c_m
            e_vol = np.linalg.norm(c_m - c_r) / np.linalg.norm(c_r)
            pl = g.fluid[:, :, kz]
            e_pl = np.linalg.norm(pm[:, :, kz][pl] - ref[t][:, :, kz][pl]) / np.linalg.norm(ref[t][:, :, kz][pl])
            say(f"{t:6.0f} | {100 * e_pl:7.1f}% {100 * e_vol:8.1f}% | {c_m.sum() / c_r.sum():14.3f}")
            for r in rows:
                if r["t"] == t:
                    r.update(co2_rel_L2_plane=e_pl, co2_rel_L2_volume=e_vol, co2_mass_ratio=c_m.sum() / c_r.sum())
            if t in plane_maps:
                plane_maps[t] = plane_maps[t] + (ref[t][:, :, kz], pm[:, :, kz])

    if rows:
        with open(os.path.join(out_dir, "comparison.csv"), "w", newline="") as fh:
            keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "t", k))
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)

    # ---- figures: breathing plane at the last compared time
    if plane_maps:
        t = max(plane_maps)
        mp = plane_maps[t]
        ext = [ROOM_X[0], ROOM_X[1], ROOM_Y[0], ROOM_Y[1]]
        ncol = 3
        nrow = 2 if len(mp) > 4 else 1
        fig, axes = plt.subplots(nrow, ncol, figsize=(18, 4.6 * nrow), squeeze=False)
        vmax = np.nanmax(mp[0])
        for ax, F, title in ((axes[0, 0], mp[0], "speed, OpenFOAM"), (axes[0, 1], mp[1], f"speed, {version}"),
                             (axes[0, 2], mp[1] - mp[0], "speed difference (model - OpenFOAM)")):
            lim = vmax if "difference" not in title else np.nanmax(np.abs(mp[1] - mp[0]))
            im = ax.imshow(F.T, origin="lower", extent=ext, cmap="RdBu_r" if "difference" in title else "viridis",
                           vmin=-lim if "difference" in title else 0, vmax=lim)
            ax.set_title(f"{title} [m/s], z={BREATHING_HEIGHT} m, t={t:g} s")
            fig.colorbar(im, ax=ax, fraction=0.03)
        if nrow == 2:
            cmax = np.nanmax(mp[4])
            for ax, F, title in ((axes[1, 0], mp[4], "CO2, FV transport by OpenFOAM flow"),
                                 (axes[1, 1], mp[5], f"CO2, {version}"),
                                 (axes[1, 2], mp[5] - mp[4], "CO2 difference (model - reference)")):
                lim = cmax if "difference" not in title else np.nanmax(np.abs(mp[5] - mp[4]))
                im = ax.imshow(F.T, origin="lower", extent=ext, cmap="RdBu_r" if "difference" in title else "magma",
                               vmin=-lim if "difference" in title else 0, vmax=lim)
                ax.set_title(f"{title}, t={t:g} s")
                fig.colorbar(im, ax=ax, fraction=0.03)
        for ax in axes.ravel():
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
        fig.suptitle(f"{name}: physics-informed GNOT ({version}) vs OpenFOAM (same equations), breathing plane")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"breathing_plane_t{t:g}.png"), dpi=120)
    say(f"\nresults in {out_dir}")


if __name__ == "__main__":
    main()
