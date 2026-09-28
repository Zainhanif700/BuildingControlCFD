"""
Where does the momentum (NS) loss come from?  v21 at nu = 0.1 logs NS losses of 25 ... 1700 that
jump from one iteration to the next (guide_w 1e2 - 7e3). Suspicion: a few points with a huge residual
dominate, e.g. next to the edges of the openings, where the analytic through-flow B_p has sharp
gradients (window/door taper 0.1-0.2 m) -> large viscous term nu*lap(u), and where the network
correction cannot help (it is multiplied by phi = 0 on the walls).

This evaluates the per-point residual on the TRAINING distribution (sample_interior, with the
checkpoint's own scenario) and reports:
  * mean normalised residual^2 at the checkpoint's nu and at nu = 0.01 (same points, same model);
  * share of the total carried by the top 1% / 0.1% points (heavy tail -> the jumpy loss);
  * where the top 1% are: distance to the nearest opening (window 1 / doors), early time t < 3 s;
  * how much of their residual is the viscous term nu*lap(u).
Self-check: the mean over the first batch is compared with train_gnot.physics_loss on the SAME points.

Usage (training env, CPU so a running training is not disturbed; ~5-10 min):
  CUDA_VISIBLE_DEVICES="" python3 diagnose_ns_residual.py checkpoints/v21_single/gnot_v21_single_iter3000.pth
"""
import argparse
import math

import torch

import point_sampler as ps
import train_gnot as tg
from gnot_model import GNOTOperator, check_checkpoint_compat


def residual_parts(model, x, y, z, t, V, N):
    """Same formulas as train_gnot.physics_loss (momentum part): returns per-point
    R0 = du/dt + (u.grad)u + grad p / rho  and  lap(u), each (n, 3), plus ns_scale (n, 1)."""
    g = tg.grad
    u, v, w, c, p = tg.get_velocity_and_derivs(model, x, y, z, t, V, N)
    R0, LAP = [], []
    dp = (g(p, x), g(p, y), g(p, z))
    for comp, dpc in zip((u, v, w), dp):
        dx_, dy_, dz_, dt_ = g(comp, x), g(comp, y), g(comp, z), g(comp, t)
        lap = g(dx_, x) + g(dy_, y) + g(dz_, z)
        R0.append((dt_ + u * dx_ + v * dy_ + w * dz_ + dpc / tg.RHO).detach())
        LAP.append(lap.detach())
    return torch.cat(R0, 1), torch.cat(LAP, 1), (tg.velocity_scale(V) ** 2 / tg.L_NS).detach()


def dist_to_openings(x, y, z):
    """Distance [m] from each point to the nearest open window (only those open in FIXED_V) or door
    rectangle (windows on y = LY, doors on y = 0)."""
    rects = [(a, b, c, d, ps.ROOM_Y[1]) for k, (a, b, c, d) in enumerate(ps.WINDOWS) if ps.FIXED_V[k] > 0]
    rects += [(a, b, c, d, ps.ROOM_Y[0]) for (a, b, c, d) in ps.DOORS]
    out = None
    for a, b, c, d, yp in rects:
        dx = (a - x).clamp_min(0) + (x - b).clamp_min(0)
        dz = (c - z).clamp_min(0) + (z - d).clamp_min(0)
        dd = torch.sqrt(dx ** 2 + dz ** 2 + (y - yp) ** 2)
        out = dd if out is None else torch.minimum(out, dd)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cpu"
    ck = torch.load(args.checkpoint, map_location=dev)
    check_checkpoint_compat(ck, args.checkpoint)
    model = GNOTOperator().to(dev)
    model.load_state_dict(ck["model_state"])
    model.eval()
    nu_ck = float(ck.get("nu", tg.NU))
    ps.FIXED_V = ck.get("scenario_V") or tg.SINGLE_SCENARIO_V
    print(f"{args.checkpoint}: iter {ck.get('iter')}, nu {nu_ck:g}, scenario V = {ps.FIXED_V}")
    torch.manual_seed(args.seed)

    rows = []
    for b in range(args.batches):
        x, y, z, t, V, N = ps.sample_interior(tg.POINTS_INTERIOR, dev)
        N = tg._co2_occupancy(N)
        for q in (x, y, z, t):
            q.requires_grad_(True)
        R0, LAP, sc = residual_parts(model, x, y, z, t, V, N)
        if b == 0:   # self-check against the training loss on the SAME points
            pts = (x.detach(), y.detach(), z.detach(), t.detach(), V, N)
            orig = tg.sample_interior
            tg.sample_interior = lambda n, device: tuple(q.clone() for q in pts)
            try:
                L_train = tg.physics_loss(model, dev, nu=nu_ck)[0].item()
            finally:
                tg.sample_interior = orig
            L_here = (((R0 - nu_ck * LAP) / sc) ** 2).sum(1).mean().item()
            print(f"self-check: physics_loss {L_train:.6g} vs per-point mean {L_here:.6g} "
                  f"({'OK' if abs(L_train - L_here) <= 1e-3 * abs(L_train) + 1e-9 else 'MISMATCH'})")
        rows.append((x.detach(), y.detach(), z.detach(), t.detach(), R0, LAP, sc))
        print(f"  batch {b + 1}/{args.batches} done", flush=True)

    x, y, z, t, R0, LAP, sc = (torch.cat([r[i] for r in rows], 0) for i in range(7))
    n = x.shape[0]

    def r2(nu):
        return (((R0 - nu * LAP) / sc) ** 2).sum(1)
    e_ck, e_01 = r2(nu_ck), r2(0.01)
    visc2 = ((nu_ck * LAP / sc) ** 2).sum(1)
    rest2 = ((R0 / sc) ** 2).sum(1)
    order = torch.argsort(e_ck, descending=True)
    k1, k01 = max(1, n // 100), max(1, n // 1000)
    top = order[:k1]
    d = dist_to_openings(x, y, z).squeeze(1)
    tt = t.squeeze(1)
    print(f"\n{n} points (training distribution, scenario {ps.FIXED_V})")
    print(f"mean normalised NS residual^2: {e_ck.mean().item():.4g} at nu = {nu_ck:g} | "
          f"{e_01.mean().item():.4g} at nu = 0.01 (same model, same points)")
    print(f"median {e_ck.median().item():.4g}; share of the total from the top 1%: "
          f"{e_ck[top].sum().item() / e_ck.sum().item():.1%}, top 0.1%: "
          f"{e_ck[order[:k01]].sum().item() / e_ck.sum().item():.1%}")
    for name, m in (("all points", torch.ones(n, dtype=torch.bool)),
                    ("top 1%", torch.zeros(n, dtype=torch.bool).index_fill_(0, top, True))):
        print(f"  {name:10s}: within 0.3 m of an opening {(d[m] < 0.3).float().mean().item():6.1%}, "
              f"within 1 m {(d[m] < 1.0).float().mean().item():6.1%}, t < 3 s {(tt[m] < 3).float().mean().item():6.1%}, "
              f"viscous/total (|nu lap|^2 / (|nu lap|^2 + |rest|^2)) "
              f"{(visc2[m].sum() / (visc2[m].sum() + rest2[m].sum())).item():.2f}")
    near = d < 0.3
    print(f"share of the total residual^2 from points within 0.3 m of an opening "
          f"({near.float().mean().item():.1%} of the points): {e_ck[near].sum().item() / e_ck.sum().item():.1%}")
    print("\nReading: a large top-1% share located at the openings with a high viscous share = B_p's sharp"
          "\nedge profile, which the network cannot correct there (phi = 0); it scales with nu^2, so it is"
          "\nworst in the nu = 0.1 stage and swamps the gradient (guide_w >> 10).")


if __name__ == "__main__":
    main()
