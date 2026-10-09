"""
Test without training: the built-in base flow (throughflow.py) alone compared with OpenFOAM.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--times", type=float, nargs="*", default=[60.0, 120.0])
    ap.add_argument("--depths", type=float, nargs="*", default=[], help="jet_<d> variants (v21 test; worse)")
    ap.add_argument("--eps-sets", nargs="*", default=["0.1,0.2", "0.2,0.3", "0.3,0.4"],
                    help="v22 check 3: opening edge widths 'EPS_WINDOW,EPS_DOOR' [m] tried with the smooth jet")
    ap.add_argument("--edges", type=float, nargs="*", default=[0.3],
                    help="v22 smooth-jet variants: tanh edge length JET_EDGE_W [m] (spread depth JET_SPREAD_L fixed)")
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--nu", type=float, default=None, help="nu for the viscous check (default: the case's nu)")
    ap.add_argument("--batch", type=int, default=20000)
    args = ap.parse_args()
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import throughflow as tfl
    from point_sampler import BREATHING_HEIGHT, ROOM_X, ROOM_Y
    import check_co2_with_model_flow as L2
    from compare_with_openfoam import read_internal, read_patch_sum, time_dirs

    case = os.path.abspath(os.path.expanduser(args.case))
    meta = dict(line.split(None, 1) for line in open(os.path.join(case, "scenario.txt")).read().splitlines())
    V = [float(v) for v in meta["V"].split()]
    dx = float(meta["dx"])
    out_dir = os.path.join(HERE, "results", f"{os.path.basename(case)}__bp_variants_v22")
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "bp_variants.log"), "w")

    def say(s=""):
        print(s, flush=True)
        log.write(s + "\n")

    g = L2.Grid(dx, V)
    C = read_internal(os.path.join(case, "0", "C"), 3)
    idx = [np.clip(np.floor((C[:, a] - (ROOM_X[0], ROOM_Y[0], 0.0)[a]) / g.h[a]).astype(int), 0, g.n[a] - 1)
           for a in range(3)]
    tdirs = dict((t, d) for t, d in time_dirs(case))
    kz = int(np.argmin(np.abs(g.c1d[2] - BREATHING_HEIGHT)))
    fl = g.fluid
    Pf = np.stack([g.X[fl], g.Y[fl], g.Z[fl]], 1)

    dev = args.device
    f64 = torch.float64

    def bp_velocity(t, alpha, blend, depth, smooth, edge):
        out = []
        Vt = torch.tensor([V], dtype=f64, device=dev)
        for i in range(0, len(Pf), args.batch):
            Q = torch.tensor(Pf[i:i + args.batch], dtype=f64, device=dev)
            x, y, z = (Q[:, j:j + 1].clone().requires_grad_(True) for j in range(3))
            n = x.shape[0]
            chi, psi = tfl.through_flow_potential(x, y, z, torch.full((n, 1), t, dtype=f64, device=dev),
                                                  Vt.expand(n, -1), torch.full((n, 1), alpha, dtype=f64, device=dev),
                                                  blend=blend, turn_depth=depth, smooth=smooth, edge_w=edge)
            gp = torch.autograd.grad(psi.sum(), (x, y), retain_graph=True)
            gc = torch.autograd.grad(chi.sum(), (y, z))
            u, v, w = gp[1], gc[1] - gp[0], -gc[0]
            out.append(torch.cat([u, v, w], 1).detach().cpu().numpy())
        U = np.full(g.X.shape + (3,), np.nan)
        U[fl] = np.concatenate(out)
        return U

    with torch.no_grad():
        a_pot = tfl.alpha_potential(torch.tensor([V], dtype=torch.float64)).item()
    say(f"case {case}: V = {V}, dx = {dx}; potential-flow alpha = {a_pot:.3f}")
    say("reference: trained v19 (W1 1 m/s, nu 0.01, t=60/120 s): velocity error 74% volume / 67% plane; "
        "trained v21 iter 3000 (nu 0.1): 136% / 109%\n")
    e0 = (tfl.EPS_WINDOW, tfl.EPS_DOOR)
    eps_sets = [tuple(float(v) for v in es.split(",")) for es in args.eps_sets]
    variants = ([("sharp", "linear", None, False, None, e0)]
                + [(f"sm{e:g}_ew{ew:g}_ed{ed:g}", "linear", None, True, e, (ew, ed))
                   for e in args.edges for ew, ed in eps_sets]
                + [(f"jet_{d:g}m", "jet", d, False, None, e0) for d in args.depths])

    def use_eps(ew, ed):
        assert ew <= 0.3 and ed <= 0.5, "edge wider than half the opening"
        tfl.EPS_WINDOW, tfl.EPS_DOOR = ew, ed
        tfl._DOOR_HN = tfl.DOOR_HEIGHT - ed / 2
    import point_sampler as ps
    import train_gnot as tg
    nu = args.nu if args.nu is not None else float(meta.get("nu", "0.01"))
    ps.FIXED_V = V
    torch.manual_seed(0)
    pts = [ps.sample_interior(2000, dev) for _ in range(5)]
    ps.FIXED_V = None

    def gz(f, v):
        if not f.requires_grad:
            return torch.zeros_like(v)
        r = torch.autograd.grad(f, v, grad_outputs=torch.ones_like(f), create_graph=True, allow_unused=True)[0]
        return torch.zeros_like(v) if r is None else r
    say(f"(1) viscous term of B_p ALONE, |nu lap(u)|^2 / (U_ref^2/L)^2 on 10000 training points, nu = {nu:g}"
        f"\n    (v21, sharp: this term was 99% of the NS loss; the network cannot cancel it)")
    say(f"{'variant':>20s} | {'mean':>9s} {'top-1%':>7s} | share of the total from: window zone (y > LY-1.3 m) / "
        f"door zone (y < DOOR_STRIP+1.3 m) / rest")
    for name, blend, depth, smooth, edge, (ew, ed) in variants:
        use_eps(ew, ed)
        vals, ys = [], []
        for (x0, y0, z0, t0, V0, _) in pts:
            x, y, z, t = (q.double().clone().requires_grad_(True) for q in (x0, y0, z0, t0))
            Vd = V0.double()
            al = tfl.alpha_potential(Vd)
            chi, psi = tfl.through_flow_potential(x, y, z, t, Vd, al, blend=blend, turn_depth=depth,
                                                  smooth=smooth, edge_w=edge)
            ub = (gz(psi, y), gz(chi, z) - gz(psi, x), -gz(chi, y))
            lap = torch.cat([gz(gz(c, x), x) + gz(gz(c, y), y) + gz(gz(c, z), z) for c in ub], 1)
            sc = tg.velocity_scale(Vd) ** 2 / tg.L_NS
            vals.append(((nu * lap / sc) ** 2).sum(1).detach().cpu())
            ys.append(y0.squeeze(1).detach().cpu())
        e, yy = torch.cat(vals), torch.cat(ys)
        k = max(1, e.numel() // 100)
        wz = yy > ROOM_Y[1] - 1.3
        dz = (yy < ROOM_Y[0] + tfl.DOOR_STRIP + 1.3) & ~wz
        tot = e.sum().item()
        say(f"{name:>20s} | {e.mean().item():9.4g} {e.sort(descending=True).values[:k].sum().item() / tot:7.1%} | "
            f"{e[wz].sum().item() / tot:6.1%} / {e[dz].sum().item() / tot:6.1%} / {e[~wz & ~dz].sum().item() / tot:6.1%}")
    use_eps(*e0)
    say("")
    say("(2) velocity of B_p ALONE vs OpenFOAM (relative L2)")
    say(f"{'t':>5s} {'variant':>20s} {'alpha':>11s} | {'volume':>7s} {'plane':>7s} | {'|u| B_p':>8s} {'|u| OF':>7s}")
    best = None
    maps = {}
    for t in args.times:
        d = tdirs.get(t)
        if d is None:
            say(f"{t:5.0f}  not saved by OpenFOAM -- skipped")
            continue
        U_of = np.full(g.X.shape + (3,), np.nan)
        U_of[idx[0], idx[1], idx[2]] = read_internal(os.path.join(case, d, "U"), 3, len(C))
        d1 = read_patch_sum(os.path.join(case, d, "phi"), "door1")
        d2 = read_patch_sum(os.path.join(case, d, "phi"), "door2")
        a_of = d1 / (d1 + d2)
        for name, blend, depth, smooth, edge, (ew, ed) in variants:
            use_eps(ew, ed)
            for a_lab, alpha in (("pot", a_pot),):
                U = bp_velocity(t, alpha, blend, depth, smooth, edge)
                e_v = np.sqrt(np.nansum((U[fl] - U_of[fl]) ** 2) / np.nansum(U_of[fl] ** 2))
                pl = fl[:, :, kz]
                e_p = np.sqrt(np.nansum((U[:, :, kz][pl] - U_of[:, :, kz][pl]) ** 2) / np.nansum(U_of[:, :, kz][pl] ** 2))
                say(f"{t:5.0f} {name:>20s} {a_lab + f'={alpha:.2f}':>11s} | {100 * e_v:6.1f}% {100 * e_p:6.1f}% | "
                    f"{np.nanmean(np.linalg.norm(U[fl], axis=1)):8.3f} {np.nanmean(np.linalg.norm(U_of[fl], axis=1)):7.3f}")
                if t == max(args.times) and a_lab == "pot":
                    maps[name] = np.linalg.norm(U[:, :, kz], axis=-1)
                    if best is None or e_v < best[1]:
                        best = (name, e_v)
        if t == max(args.times):
            maps["OpenFOAM"] = np.linalg.norm(U_of[:, :, kz], axis=-1)
    if maps:
        names = ["OpenFOAM"] + [v_[0] for v_ in variants]
        fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 3.6))
        vmax = np.nanmax(maps["OpenFOAM"])
        ext = [ROOM_X[0], ROOM_X[1], ROOM_Y[0], ROOM_Y[1]]
        for ax, n in zip(axes, names):
            im = ax.imshow(maps[n].T, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax)
            ax.set_title(n if n == "OpenFOAM" else f"B_p only: {n}")
            ax.set_xlabel("x [m]")
        axes[0].set_ylabel("y [m]")
        fig.colorbar(im, ax=axes, fraction=0.015, label="speed [m/s]")
        fig.suptitle(f"{meta['name']}: speed at z = {BREATHING_HEIGHT} m, t = {max(args.times):g} s -- "
                     f"through-flow field alone (no network) vs OpenFOAM")
        fig.savefig(os.path.join(out_dir, "bp_variants_speed.png"), dpi=110, bbox_inches="tight")
    if best:
        say(f"\nclosest B_p variant alone (potential alpha, t = {max(args.times):g} s): {best[0]} "
            f"({100 * best[1]:.1f}% volume error)")
    say(f"results in {out_dir}")


if __name__ == "__main__":
    main()
