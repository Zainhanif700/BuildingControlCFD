"""
Diagnostic: where does the momentum (Navier-Stokes) loss come from?
"""
import argparse
import math

import torch

import point_sampler as ps
import train_gnot as tg
from gnot_model import GNOTOperator, check_checkpoint_compat


def residual_parts(model, x, y, z, t, V, N):
    """The terms of the momentum residual (same formulas as in training)."""
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
    """Distance from each point to the nearest open window or door."""
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
        if b == 0:
            pts = (x.detach(), y.detach(), z.detach(), t.detach(), V, N)
            orig = tg.sample_interior
            tg.sample_interior = lambda n, device: tuple(q.clone() for q in pts)
            try:
                L_train = tg.physics_loss(model, dev, nu=nu_ck)[0].item()
            finally:
                tg.sample_interior = orig
            n_sel = ps.interior_uniform_count(R0.shape[0]) if getattr(tg, "NS_UNIFORM_POINTS_ONLY", False) else R0.shape[0]
            r2_here = (((R0 - nu_ck * LAP) / sc) ** 2).sum(1)[:n_sel]
            L_here = (2.0 * (torch.sqrt(1.0 + r2_here) - 1.0)).mean().item() if getattr(tg, "NS_PSEUDO_HUBER", False) \
                else r2_here.mean().item()
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
    from throughflow import through_flow_potential, COLUMN_BLEND, DOOR_STRIP
    xt, yt, zt, t_t = (q[top].clone().requires_grad_(True) for q in (x, y, z, t))
    Vt = torch.tensor([ps.FIXED_V], dtype=xt.dtype).expand(xt.shape[0], -1)
    with torch.no_grad():
        alpha = model.door_split(t_t.detach(), Vt)
    chi, psi = through_flow_potential(xt, yt, zt, t_t, Vt, alpha)

    def g(f, v):
        if not f.requires_grad:
            return torch.zeros_like(v)
        r = torch.autograd.grad(f, v, grad_outputs=torch.ones_like(f), create_graph=True, allow_unused=True)[0]
        return torch.zeros_like(v) if r is None else r
    ub = (g(psi, yt), g(chi, zt) - g(psi, xt), -g(chi, yt))
    lap_bp = torch.cat([g(g(c, xt), xt) + g(g(c, yt), yt) + g(g(c, zt), zt) for c in ub], 1).detach()
    visc_tot = ((nu_ck * LAP[top] / sc[top]) ** 2).sum(1)
    visc_bp = ((nu_ck * lap_bp / sc[top]) ** 2).sum(1)
    print(f"\ntop 1%: viscous term of B_p ALONE / viscous term of the full model = "
          f"{(visc_bp.sum() / visc_tot.sum()).item():.2f}  (~1: B_p's own curvature; << 1 or >> 1: the network part)")
    xd, yd, zd = x[top].squeeze(1), y[top].squeeze(1), z[top].squeeze(1)
    col = torch.stack([torch.sqrt((xd - cx) ** 2 + (yd - cy) ** 2) - r - rb
                       for (cx, cy, r, _, _), rb in zip(ps.COLUMNS, COLUMN_BLEND)], 1).min(1).values
    in_door_x = torch.zeros_like(xd, dtype=torch.bool)
    for a, b, _, _ in ps.DOORS:
        in_door_x |= (xd > a - 0.1) & (xd < b + 0.1)
    wall = torch.stack([xd - ps.ROOM_X[0], ps.ROOM_X[1] - xd, yd - ps.ROOM_Y[0], ps.ROOM_Y[1] - yd,
                        zd - ps.ROOM_Z[0], ps.ROOM_Z[1] - zd], 1).min(1).values
    classes = [("window/door opening (< 0.3 m)", d[top] < 0.3),
               ("door-turn strip (door x-range, y < DOOR_STRIP)", in_door_x & (yd < DOOR_STRIP + 0.1)),
               ("column blend ring (psi blended to the column)", col < 0.1),
               ("wall layer (< 0.3 m from wall/floor/ceiling)", wall < 0.3),
               ("elsewhere in the room", torch.ones_like(xd, dtype=torch.bool))]
    taken = torch.zeros_like(xd, dtype=torch.bool)
    tot = e_ck[top].sum().item()
    for name, m in classes:
        m = m & ~taken
        taken |= m
        print(f"  {name:50s}: {m.float().mean().item():6.1%} of the top points, "
              f"{e_ck[top][m].sum().item() / tot:6.1%} of their residual^2")
    print("  10 largest (x, y, z [m], t [s], residual^2):")
    for i in range(10):
        print(f"    ({xd[i].item():6.2f}, {yd[i].item():5.2f}, {zd[i].item():5.2f}), t {tt[top][i].item():6.1f}: "
              f"{e_ck[top][i].item():.3g}")
    print("\nReading: a large top-1% share located at the openings with a high viscous share = B_p's sharp"
          "\nedge profile, which the network cannot correct there (phi = 0); it scales with nu^2, so it is"
          "\nworst in the nu = 0.1 stage and swamps the gradient (guide_w >> 10).")


if __name__ == "__main__":
    main()
