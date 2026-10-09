"""
Physics-only training of the GNOT model (no simulation data): Navier-Stokes and CO2 equation residuals at random points.
Usage: python3 train_gnot.py
"""
import math
import os
import time
import torch

from gnot_model import GNOTOperator, NONDIM_CHECKPOINT_KEY, MODEL_FORMAT_KEY, MODEL_FORMAT
from throughflow import bp_velocity_laplacian
from point_sampler import (
    sample_interior, sample_walls, sample_doors, sample_windows, sample_ic,
    sample_columns_surface, _generate_interior_batch, interior_uniform_count, USE_PERSISTENT_POOL,
    ROOM_X, ROOM_Y, ROOM_Z, NUM_WINDOWS, CO2_SOURCE_SIGMA, BREATHING_HEIGHT,
    EMISSION_PER_PERSON, S_REF, C_REF, TAU_RAMP, COLUMNS, N_PEOPLE_MAX, WINDOWS, DOORS,
)

NU = 0.01
RHO = 1.0
DIFFUSIVITY = 0.005
SIGMA = CO2_SOURCE_SIGMA
SOURCE_X = (ROOM_X[0] + ROOM_X[1]) / 2
SOURCE_Y = (ROOM_Y[0] + ROOM_Y[1]) / 2

POINTS_INTERIOR = 1000
POINTS_WALLS = 600
POINTS_COLUMNS_PER = 40
POINTS_WINDOWS_PER = 40
POINTS_DOORS = 200
POINTS_IC = 400
MAX_ITERS = 20000
LOG_EVERY = 10
CKPT_EVERY = 1000
LR = 1e-3

RESUME_FROM = None
LR_DECAY_START = None
LR_MIN = 1e-5


NU_SCHEDULE = [(None, NU)]
LR_EXP_DECAY = (0.9, 2000)


def nu_at(it):
    """Viscosity used in the momentum residual at iteration it."""
    for last, nu in NU_SCHEDULE:
        if last is None or it <= last:
            return nu
    return NU


def lr_at(it):
    """Learning rate at iteration it."""
    if LR_EXP_DECAY is not None:
        rate, steps = LR_EXP_DECAY
        return LR * rate ** (it / steps)
    if LR_DECAY_START is None or it <= LR_DECAY_START:
        return LR
    frac = min(1.0, (it - LR_DECAY_START) / (MAX_ITERS - LR_DECAY_START))
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1.0 + math.cos(math.pi * frac))


CO2_WEIGHT_MIN = 1.0
CO2_WEIGHT_MAX = 200.0
CO2_WEIGHT_EMA_ALPHA = 0.1
CO2_WEIGHT_WARMUP_ITERS = 500
CO2_WEIGHT_UPDATE_EVERY = 100
GRAD_CLIP_MAX_NORM = 10.0


USE_ADAPTIVE_CO2_WEIGHT = False


def guide_norm_ratio(grad_ns, grad_co2):
    """Ratio of the Navier-Stokes and CO2 loss gradients."""
    ns_sq = sum((g ** 2).sum() for g in grad_ns if g is not None)
    co2_sq = sum((g ** 2).sum() for g in grad_co2 if g is not None)
    if not torch.is_tensor(co2_sq) or not torch.is_tensor(ns_sq) or co2_sq.item() < 1e-30:
        return float("nan")
    return (ns_sq.sqrt() / co2_sq.sqrt()).item()


CO2_LOSS_AT_FULL_OCCUPANCY = True


def _co2_occupancy(N_people):
    """Number of people used inside the loss functions."""
    return torch.full_like(N_people, N_PEOPLE_MAX) if CO2_LOSS_AT_FULL_OCCUPANCY else N_people


def compute_param_grads(loss, params, retain_graph):
    """Gradient of one loss term with respect to the parameters (without touching .grad)."""
    return torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)


def gradnorm_weight_update(grad_ns, grad_co2, prev_weight):
    """Updates the CO2 loss weight from the ratio of the gradient sizes."""
    ns_abs = torch.cat([g.abs().flatten() for g in grad_ns if g is not None])
    co2_abs = torch.cat([g.abs().flatten() for g in grad_co2 if g is not None])
    if ns_abs.numel() == 0 or co2_abs.numel() == 0:
        return prev_weight
    co2_mean = co2_abs.mean()
    if co2_mean < 1e-12:
        return prev_weight
    target_weight = (ns_abs.max() / co2_mean).item()
    target_weight = max(CO2_WEIGHT_MIN, min(CO2_WEIGHT_MAX, target_weight))
    return (1 - CO2_WEIGHT_EMA_ALPHA) * prev_weight + CO2_WEIGHT_EMA_ALPHA * target_weight

VERSION = "v25_nsuniform"
SINGLE_SCENARIO_V = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

OPTIMIZER = "adam"
SOAP_BETAS = (0.99, 0.999)
SOAP_PRECONDITION_FREQUENCY = 2
SOAP_WEIGHT_DECAY = 0.0


def make_optimizer(params):
    """Builds the configured optimiser."""
    if OPTIMIZER == "adam":
        return torch.optim.Adam(params, lr=LR)
    if OPTIMIZER == "soap":
        from soap import SOAP
        return SOAP(params, lr=LR, betas=SOAP_BETAS, weight_decay=SOAP_WEIGHT_DECAY,
                    precondition_frequency=SOAP_PRECONDITION_FREQUENCY, eps=1e-8)
    raise ValueError(f"unknown OPTIMIZER {OPTIMIZER!r} (use 'adam' or 'soap')")

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints", VERSION)
os.makedirs(CKPT_DIR, exist_ok=True)


def grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def get_velocity_and_derivs(model, x, y, z, t, V, N_people):
    """Model output with the divergence-free velocity and the derivatives needed for the residuals."""
    A1, A2, A3, C, p = model(x, y, z, t, V, N_people)
    u, v, w = model.velocity_from_potential(A1, A2, A3, x, y, z)
    return u, v, w, C, p


USE_BP_RESIDUAL_WEIGHT = False
NS_PSEUDO_HUBER = True
WALLS_WEIGHT = 10.0
SKIP_IC_LOSS = True
BP_WEIGHT_TAU = 1.0
_NS_DIAG = {}
NS_UNIFORM_POINTS_ONLY = True


def physics_loss(model, device, nu=NU):
    """Navier-Stokes and CO2 residual losses at the interior points (returned separately)."""
    x, y, z, t, V, N_people = sample_interior(POINTS_INTERIOR, device)
    N_people = _co2_occupancy(N_people)
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True); t.requires_grad_(True)

    u, v, w, c, p = get_velocity_and_derivs(model, x, y, z, t, V, N_people)

    du_dx, du_dy, du_dz, du_dt = grad(u, x), grad(u, y), grad(u, z), grad(u, t)
    dv_dx, dv_dy, dv_dz, dv_dt = grad(v, x), grad(v, y), grad(v, z), grad(v, t)
    dw_dx, dw_dy, dw_dz, dw_dt = grad(w, x), grad(w, y), grad(w, z), grad(w, t)
    dc_dx, dc_dy, dc_dz, dc_dt = grad(c, x), grad(c, y), grad(c, z), grad(c, t)
    dp_dx, dp_dy, dp_dz = grad(p, x), grad(p, y), grad(p, z)

    d2u = grad(du_dx, x) + grad(du_dy, y) + grad(du_dz, z)
    d2v = grad(dv_dx, x) + grad(dv_dy, y) + grad(dv_dz, z)
    d2w = grad(dw_dx, x) + grad(dw_dy, y) + grad(dw_dz, z)
    d2c = grad(dc_dx, x) + grad(dc_dy, y) + grad(dc_dz, z)

    conv_u = u * du_dx + v * du_dy + w * du_dz
    conv_v = u * dv_dx + v * dv_dy + w * dv_dz
    conv_w = u * dw_dx + v * dw_dy + w * dw_dz
    conv_c = u * dc_dx + v * dc_dy + w * dc_dz

    res_u = du_dt + conv_u + (1.0 / RHO) * dp_dx - nu * d2u
    res_v = dv_dt + conv_v + (1.0 / RHO) * dp_dy - nu * d2v
    res_w = dw_dt + conv_w + (1.0 / RHO) * dp_dz - nu * d2w

    dist2 = (x - SOURCE_X) ** 2 + (y - SOURCE_Y) ** 2 + (z - BREATHING_HEIGHT) ** 2
    S = N_people * EMISSION_PER_PERSON * torch.exp(-dist2 / (SIGMA ** 2))
    res_c = dc_dt + conv_c - DIFFUSIVITY * d2c - S

    ns_scale = velocity_scale(V) ** 2 / L_NS
    r2 = (res_u / ns_scale) ** 2 + (res_v / ns_scale) ** 2 + (res_w / ns_scale) ** 2
    n_ns = r2.shape[0]
    if NS_UNIFORM_POINTS_ONLY:
        assert not USE_PERSISTENT_POOL, "NS_UNIFORM_POINTS_ONLY relies on the fresh batch's row order"
        n_ns = interior_uniform_count(r2.shape[0])
        r2 = r2[:n_ns]
    _NS_DIAG["n"] = n_ns
    if USE_BP_RESIDUAL_WEIGHT:
        with torch.no_grad():
            alpha = model.door_split(t.detach(), V)
            lap_bp = bp_velocity_laplacian(x, y, z, t, V, alpha)
            r_bp2 = ((nu * lap_bp / ns_scale) ** 2).sum(dim=1, keepdim=True)[:n_ns]
            wgt = 1.0 / (1.0 + r_bp2 / BP_WEIGHT_TAU)
        ns_loss = (wgt * r2).mean() / wgt.mean()
        _NS_DIAG.update(raw=r2.mean().item(), w_mean=wgt.mean().item(),
                        w_low=(wgt < 0.5).float().mean().item())
    elif NS_PSEUDO_HUBER:
        ns_loss = (2.0 * (torch.sqrt(1.0 + r2) - 1.0)).mean()
        _NS_DIAG.update(raw=r2.mean().item(), w_mean=1.0, w_low=0.0)
    else:
        ns_loss = r2.mean()
    co2_loss = ((res_c / S_REF) ** 2).mean()
    return ns_loss, co2_loss


_LX, _LY, _LZ = ROOM_X[1] - ROOM_X[0], ROOM_Y[1] - ROOM_Y[0], ROOM_Z[1] - ROOM_Z[0]
WINDOW_AREAS = [(x1 - x0) * (z1 - z0) for x0, x1, z0, z1 in WINDOWS]
A_SOLID = (2 * _LX * _LZ + 2 * _LY * _LZ + 2 * _LX * _LY
           - sum((x1 - x0) * (z1 - z0) for x0, x1, z0, z1 in DOORS) - sum(WINDOW_AREAS)
           - 2 * sum(math.pi * r ** 2 for _, _, r, _, _ in COLUMNS)
           + sum(2 * math.pi * r * (zh - zl) for _, _, r, zl, zh in COLUMNS))
Q_FLOOR = 0.1
V_REL_FLOOR = 0.1
USE_RELATIVE_WINDOW_LOSS = False
BC_SCALING_WARMUP = 2000


A_DOORS = sum((x1 - x0) * (z1 - z0) for x0, x1, z0, z1 in DOORS)
U_NS_FLOOR = 0.5
L_NS = _LZ
USE_WINDOW_NORMAL_TARGET = False


_ALPHA_PROBE_V = torch.tensor([[3.0] + [0.0] * 7, [0.0] * 7 + [3.0], [3.0] * 8])


def slip_scale(V):
    """Speed scale for the no-slip wall terms."""
    areas = torch.tensor(WINDOW_AREAS, device=V.device, dtype=V.dtype).view(1, -1)
    q = (V * areas).sum(dim=1, keepdim=True)
    a_open = ((V > 0).to(V.dtype) * areas).sum(dim=1, keepdim=True)
    u_win = q / a_open.clamp_min(1e-9)
    u_ref = velocity_scale(V)
    return torch.maximum(torch.minimum(u_ref, u_win), u_ref / SLIP_MAX_RATIO).clamp_min(V_REL_FLOOR)


SLIP_MAX_RATIO = 3.0


def velocity_scale(V):
    """Velocity scale of the scenario."""
    areas = torch.tensor(WINDOW_AREAS, device=V.device, dtype=V.dtype).view(1, -1)
    return ((V * areas).sum(dim=1, keepdim=True) / A_DOORS).clamp_min(U_NS_FLOOR)


def throughflow_scale(V, t=None):
    """Flow-rate scale of the scenario."""
    areas = torch.tensor(WINDOW_AREAS, device=V.device, dtype=V.dtype).view(1, -1)
    return (V * areas).sum(dim=1, keepdim=True).clamp_min(Q_FLOOR)


def _planar_normal_component(x, y, z, u, v, w):
    """Velocity component normal to the wall of each wall point."""
    on_x = (x == ROOM_X[0]) | (x == ROOM_X[1])
    on_y = (y == ROOM_Y[0]) | (y == ROOM_Y[1])
    return torch.where(on_x, u, torch.where(on_y, v, w))


def walls_loss(model, device, flux_weight=1.0):
    """No-slip and no-leak loss on the walls."""
    x, y, z, t, V, N_people = sample_walls(POINTS_WALLS, device)
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True)
    u, v, w, c, p = get_velocity_and_derivs(model, x, y, z, t, V, N_people)
    us2 = slip_scale(V) ** 2
    loss = (u ** 2 / us2).mean() + (v ** 2 / us2).mean() + (w ** 2 / us2).mean()
    un = _planar_normal_component(x.detach(), y.detach(), z.detach(), u, v, w)
    un_scaled = [A_SOLID * un / throughflow_scale(V, t)]

    xc, yc, zc, tc, Vc, Nc = sample_columns_surface(POINTS_COLUMNS_PER, device)
    xc.requires_grad_(True); yc.requires_grad_(True); zc.requires_grad_(True)
    uc, vc, wc, cc, pc = get_velocity_and_derivs(model, xc, yc, zc, tc, Vc, Nc)
    usc2 = slip_scale(Vc) ** 2
    loss = loss + (uc ** 2 / usc2).mean() + (vc ** 2 / usc2).mean() + (wc ** 2 / usc2).mean()
    centers = torch.tensor([[cx, cy] for cx, cy, _, _, _ in COLUMNS], device=device, dtype=xc.dtype)
    radii = torch.tensor([r for _, _, r, _, _ in COLUMNS], device=device, dtype=xc.dtype)
    xd_, yd_ = xc.detach(), yc.detach()
    k = ((xd_ - centers[:, 0]) ** 2 + (yd_ - centers[:, 1]) ** 2).argmin(dim=1)
    nx = ((xd_.squeeze(-1) - centers[k, 0]) / radii[k]).unsqueeze(-1)
    ny = ((yd_.squeeze(-1) - centers[k, 1]) / radii[k]).unsqueeze(-1)
    un_scaled.append(A_SOLID * (uc * nx + vc * ny) / throughflow_scale(Vc, tc))
    return loss + flux_weight * (torch.cat(un_scaled, dim=0) ** 2).mean()


def windows_loss(model, device, co2_weight, rel_weight=1.0):
    """Inflow loss on the open windows."""
    x, y, z, t, V, N_people, window_idx = sample_windows(POINTS_WINDOWS_PER, device)
    N_people = _co2_occupancy(N_people)
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True)
    u, v, w, c, p = get_velocity_and_derivs(model, x, y, z, t, V, N_people)

    V_at_point = V.gather(1, window_idx)
    target_v = -V_at_point * torch.tanh(3.0 * t / TAU_RAMP)

    is_open = (V_at_point > 0).float()
    if not USE_WINDOW_NORMAL_TARGET:
        us2 = slip_scale(V) ** 2
        return ((u ** 2 / us2).mean() + (w ** 2 / us2).mean()
                + co2_weight * (is_open * (c / C_REF) ** 2).mean())
    scale2 = ((1.0 - rel_weight) + rel_weight * target_v.abs().clamp_min(V_REL_FLOOR)) ** 2
    return ((u ** 2 / scale2).mean() + ((v - target_v) ** 2 / scale2).mean() + (w ** 2 / scale2).mean()
            + co2_weight * (is_open * (c / C_REF) ** 2).mean())


def doors_loss(model, device):
    x, y, z, t, V, N_people = sample_doors(POINTS_DOORS, device)
    _, _, _, _, p = model(x, y, z, t, V, N_people)
    return ((p / velocity_scale(V) ** 2) ** 2).mean()


CO2_GRAD_REF = C_REF / SIGMA


def _planar_wall_normal_derivative(x, y, z, dc_dx, dc_dy, dc_dz):
    """dc/dn on the 6 planar room faces."""
    on_x = (x == ROOM_X[0]) | (x == ROOM_X[1])
    on_y = (y == ROOM_Y[0]) | (y == ROOM_Y[1])
    on_z = (z == ROOM_Z[0]) | (z == ROOM_Z[1])
    dc_dn = torch.where(on_x, dc_dx, torch.where(on_y, dc_dy, dc_dz))
    return dc_dn, (on_x | on_y | on_z)


def co2_boundary_loss(model, device, co2_weight):
    """CO2 boundary conditions: c = 0 at open windows, no flux through the walls."""
    x, y, z, t, V, N_people = sample_walls(POINTS_WALLS, device)
    N_people = _co2_occupancy(N_people)
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True)
    _, _, _, c, _ = model(x, y, z, t, V, N_people)
    dn_walls, _ = _planar_wall_normal_derivative(x, y, z, grad(c, x), grad(c, y), grad(c, z))

    xc, yc, zc, tc, Vc, Nc = sample_columns_surface(POINTS_COLUMNS_PER, device)
    xc.requires_grad_(True); yc.requires_grad_(True); zc.requires_grad_(True)
    _, _, _, cc, _ = model(xc, yc, zc, tc, Vc, _co2_occupancy(Nc))
    centers = torch.tensor([[cx, cy] for cx, cy, _, _, _ in COLUMNS], device=device, dtype=xc.dtype)
    radii = torch.tensor([r for _, _, r, _, _ in COLUMNS], device=device, dtype=xc.dtype)
    xd_, yd_ = xc.detach(), yc.detach()
    dist2_axis = (xd_ - centers[:, 0]) ** 2 + (yd_ - centers[:, 1]) ** 2
    k = dist2_axis.argmin(dim=1)
    nx = (xd_.squeeze(-1) - centers[k, 0]) / radii[k]
    ny = (yd_.squeeze(-1) - centers[k, 1]) / radii[k]
    dn_cols = grad(cc, xc) * nx.unsqueeze(-1) + grad(cc, yc) * ny.unsqueeze(-1)

    xd, yd, zd, td, Vd, Nd = sample_doors(POINTS_DOORS, device)
    yd.requires_grad_(True)
    _, _, _, cd, _ = model(xd, yd, zd, td, Vd, _co2_occupancy(Nd))
    dn_doors = grad(cd, yd)

    xw, yw, zw, tw, Vw, Nw, idxw = sample_windows(POINTS_WINDOWS_PER, device)
    yw.requires_grad_(True)
    _, _, _, cw, _ = model(xw, yw, zw, tw, Vw, _co2_occupancy(Nw))
    is_closed = (Vw.gather(1, idxw) == 0).squeeze(1)
    dn_closed_windows = grad(cw, yw)[is_closed]

    dn = torch.cat([dn_walls, dn_cols, dn_doors, dn_closed_windows], dim=0) / CO2_GRAD_REF
    return co2_weight * (dn ** 2).mean()


def ic_loss(model, device, co2_weight):
    x, y, z, t, V, N_people = sample_ic(POINTS_IC, device)
    N_people = _co2_occupancy(N_people)
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True)
    u, v, w, c, p = get_velocity_and_derivs(model, x, y, z, t, V, N_people)
    return (u ** 2).mean() + (v ** 2).mean() + (w ** 2).mean() + co2_weight * ((c / C_REF) ** 2).mean()


def trivial_co2_floor(device, n=200000):
    """CO2 loss of a constant CO2 field (reference value for the logs)."""
    x, y, z, t, V, N_people = _generate_interior_batch(n, device)
    N_people = _co2_occupancy(N_people)
    dist2 = (x - SOURCE_X) ** 2 + (y - SOURCE_Y) ** 2 + (z - BREATHING_HEIGHT) ** 2
    S = N_people * EMISSION_PER_PERSON * torch.exp(-dist2 / (SIGMA ** 2))
    return ((S / S_REF) ** 2).mean().item()


def main():
    import argparse
    global VERSION, CKPT_DIR, MAX_ITERS, RESUME_FROM
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=None, help="override MAX_ITERS (dry run)")
    ap.add_argument("--tag", default=None, help="suffix for VERSION / checkpoint folder (dry run)")
    ap.add_argument("--resume", default=None,
                    help="continue from an iter-numbered checkpoint (model + optimizer state), path relative to "
                         "this file; use with --tag so the continuation gets its own folder")
    args = ap.parse_args()
    if args.resume:
        RESUME_FROM = args.resume
    if args.tag:
        VERSION = f"{VERSION}_{args.tag}"
        CKPT_DIR = os.path.join(HERE, "checkpoints", VERSION)
        os.makedirs(CKPT_DIR, exist_ok=True)
    if args.iters:
        MAX_ITERS = args.iters
    print(f"VERSION={VERSION}, MAX_ITERS={MAX_ITERS}, checkpoints -> {CKPT_DIR}")
    existing = [f for f in os.listdir(CKPT_DIR) if f.endswith(".pth")]
    if existing:
        raise SystemExit(f"{CKPT_DIR} already contains {len(existing)} checkpoint(s) -- set a NEW "
                         f"VERSION in train_gnot.py before training, so existing results are not overwritten.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    import point_sampler
    point_sampler.FIXED_V = SINGLE_SCENARIO_V
    print(f"[v21] scenarios: {'FIXED V = ' + str(SINGLE_SCENARIO_V) if SINGLE_SCENARIO_V else 'training mix'}; "
          f"nu curriculum: {', '.join(f'{nu:g} (to iter {last})' if last else f'{nu:g} (to the end)' for last, nu in NU_SCHEDULE)}; "
          f"CO2 window factor: {'ON' if __import__('gnot_model').USE_CO2_WINDOW_FACTOR else 'OFF'}")

    model = GNOTOperator().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GNOT parameters: {n_params:,}")
    floor = trivial_co2_floor(device)
    print(f"[v8_nondim] Trivial-solution CO2(scaled) reference = {floor:.4f}  "
          f"(CO2(scaled) near this = still stuck on C=const; clearly below = learning CO2)")
    print(f"[v8_nondim] adaptive CO2 weighting: {'ON' if USE_ADAPTIVE_CO2_WEIGHT else 'OFF (co2_weight fixed at 1.0)'}; "
          f"guide_w column = norm-balancing weight the Expert's Guide rule would pick (diagnostic)")

    optimizer = make_optimizer(model.parameters())
    params = list(model.parameters())
    if OPTIMIZER == "soap":
        print(f"Optimizer: SOAP (betas={SOAP_BETAS}, precondition_frequency={SOAP_PRECONDITION_FREQUENCY}, "
              f"weight_decay={SOAP_WEIGHT_DECAY}); its first step only initialises the preconditioner")
    else:
        print("Optimizer: Adam")

    co2_weight = 1.0

    start_iter = 0
    if RESUME_FROM is not None:
        from gnot_model import check_checkpoint_compat
        resume_path = os.path.join(HERE, RESUME_FROM)
        if not os.path.isfile(resume_path):
            raise SystemExit(f"RESUME_FROM checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        check_checkpoint_compat(ckpt, resume_path)
        if "optimizer_state" not in ckpt:
            raise SystemExit(f"{resume_path} has no optimizer_state -- resume from an "
                             f"iter-numbered checkpoint, not _final/_best")
        if ckpt.get("optimizer", "adam") != OPTIMIZER:
            raise SystemExit(f"{resume_path} was trained with {ckpt.get('optimizer', 'adam')!r}, but "
                             f"OPTIMIZER = {OPTIMIZER!r} -- optimizer states are not interchangeable")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        co2_weight = ckpt.get("co2_weight", 1.0)
        start_iter = ckpt["iter"] + 1
        print(f"[v11] resumed from {resume_path} (iter={ckpt['iter']}, version={ckpt.get('version')}); "
              f"continuing at iter {start_iter}")
    if LR_EXP_DECAY is not None:
        print(f"LR schedule: exponential {LR:g} * {LR_EXP_DECAY[0]}**(it/{LR_EXP_DECAY[1]}) "
              f"-> {lr_at(MAX_ITERS):.2e} at iter {MAX_ITERS}")
    elif LR_DECAY_START is None:
        print(f"LR schedule: constant {LR:g}")
    else:
        print(f"LR schedule: {LR:g} constant until iter {LR_DECAY_START}, cosine to {LR_MIN:g} at iter {MAX_ITERS}")
    guide_w = float("nan")
    walls_ratio = float("nan")

    best_total_val = float("inf")

    start = time.time()
    for it in range(start_iter, MAX_ITERS + 1):
        cur_lr = lr_at(it)
        for group in optimizer.param_groups:
            group["lr"] = cur_lr
        optimizer.zero_grad()

        cur_nu = nu_at(it)
        L_ns, L_co2 = physics_loss(model, device, nu=cur_nu)
        grad_ns = compute_param_grads(L_ns, params, retain_graph=True)
        grad_co2 = compute_param_grads(L_co2, params, retain_graph=False)
        for p, g_ns, g_co2 in zip(params, grad_ns, grad_co2):
            total_grad = None
            if g_ns is not None:
                total_grad = g_ns
            if g_co2 is not None:
                weighted = co2_weight * g_co2
                total_grad = weighted if total_grad is None else total_grad + weighted
            if total_grad is None:
                continue
            p.grad = total_grad.clone() if p.grad is None else p.grad + total_grad

        bc_lam = min(1.0, it / BC_SCALING_WARMUP)
        L_walls_raw = walls_loss(model, device, flux_weight=bc_lam)
        L_walls = WALLS_WEIGHT * L_walls_raw
        if it % CO2_WEIGHT_UPDATE_EVERY == 0:
            g_walls = compute_param_grads(L_walls, params, retain_graph=True)
            walls_ratio = guide_norm_ratio(grad_ns, g_walls)
            del g_walls
        L_walls.backward()

        L_windows = windows_loss(model, device, co2_weight,
                                 rel_weight=bc_lam if USE_RELATIVE_WINDOW_LOSS else 0.0)
        L_windows.backward()

        L_doors = doors_loss(model, device)
        L_doors.backward()

        if SKIP_IC_LOSS:
            L_ic = torch.zeros((), device=device)
        else:
            L_ic = ic_loss(model, device, co2_weight)
            L_ic.backward()

        L_co2bc = co2_boundary_loss(model, device, co2_weight)
        L_co2bc.backward()

        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_MAX_NORM)

        optimizer.step()

        if it % CO2_WEIGHT_UPDATE_EVERY == 0:
            guide_w = guide_norm_ratio(grad_ns, grad_co2)
            if USE_ADAPTIVE_CO2_WEIGHT and it >= CO2_WEIGHT_WARMUP_ITERS:
                co2_weight = gradnorm_weight_update(grad_ns, grad_co2, co2_weight)

        total_val = (L_ns.item() + co2_weight * L_co2.item() + L_walls.item()
                     + L_windows.item() + L_doors.item() + L_ic.item() + L_co2bc.item())

        unweighted_total = (L_ns.item() + L_co2.item() + L_walls_raw.item()
                            + L_windows.item() + L_doors.item() + L_ic.item() + L_co2bc.item())

        if it % LOG_EVERY == 0:
            elapsed = time.time() - start
            speed = (it - start_iter + 1) / elapsed if elapsed > 0 else 0.0
            with torch.no_grad():
                a_probe = model.door_split(torch.full((3, 1), 60.0, device=device), _ALPHA_PROBE_V.to(device))
            print(f"[Iter {it:05d}/{MAX_ITERS}] Total={total_val:.5f} | "
                  f"NS={L_ns.item():.5f} CO2(scaled)={L_co2.item():.5f} CO2_weight={co2_weight:.2f} guide_w={guide_w:.3g} "
                  f"Walls={L_walls_raw.item():.5f}(x{WALLS_WEIGHT:g}) g_ns/g_walls={walls_ratio:.3g} "
                  f"Windows={L_windows.item():.5f} Doors={L_doors.item():.5f} "
                  f"IC={L_ic.item():.5f} CO2_BC={L_co2bc.item():.5f} LR={cur_lr:.2e} nu={cur_nu:g} "
                  + (f"NSraw={_NS_DIAG.get('raw', float('nan')):.4g} w_mean={_NS_DIAG.get('w_mean', float('nan')):.3f} "
                     f"w<0.5={_NS_DIAG.get('w_low', float('nan')):.1%} " if USE_BP_RESIDUAL_WEIGHT
                     else f"NSraw={_NS_DIAG.get('raw', float('nan')):.4g} " if NS_PSEUDO_HUBER else "") +
                  f"alpha(W1/W8/all)={a_probe[0].item():.2f}/{a_probe[1].item():.2f}/{a_probe[2].item():.2f} | {speed:.2f} it/s")

        if it > start_iter and cur_nu != nu_at(it - 1):
            best_total_val = float("inf")
        if it % LOG_EVERY == 0 and unweighted_total < best_total_val:
            best_total_val = unweighted_total
            best_path = os.path.join(CKPT_DIR, f"gnot_{VERSION}_best.pth")
            torch.save({"iter": it, "version": VERSION, "co2_weight": co2_weight,
                        "unweighted_total": best_total_val, NONDIM_CHECKPOINT_KEY: True, MODEL_FORMAT_KEY: MODEL_FORMAT, "lr": cur_lr,
                        "nu": cur_nu, "scenario_V": SINGLE_SCENARIO_V,
                        "model_state": model.state_dict()}, best_path)

        if it % CKPT_EVERY == 0 and it > 0:
            ckpt_path = os.path.join(CKPT_DIR, f"gnot_{VERSION}_iter{it}.pth")
            torch.save({"iter": it, "version": VERSION, "co2_weight": co2_weight,
                        NONDIM_CHECKPOINT_KEY: True, MODEL_FORMAT_KEY: MODEL_FORMAT, "lr": cur_lr,
                        "nu": cur_nu, "scenario_V": SINGLE_SCENARIO_V,
                        "optimizer": OPTIMIZER, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict()}, ckpt_path)
            print(f"  -> saved checkpoint: {ckpt_path}")

    final_path = os.path.join(CKPT_DIR, f"gnot_{VERSION}_final.pth")
    torch.save({"iter": MAX_ITERS, "version": VERSION, "co2_weight": co2_weight,
                NONDIM_CHECKPOINT_KEY: True, MODEL_FORMAT_KEY: MODEL_FORMAT, "lr": cur_lr,
                "nu": cur_nu, "scenario_V": SINGLE_SCENARIO_V,
                "model_state": model.state_dict()}, final_path)
    print(f"Training complete. Final checkpoint: {final_path}")
    print(f"Best checkpoint (lowest unweighted_total={best_total_val:.5f}): "
          f"{os.path.join(CKPT_DIR, f'gnot_{VERSION}_best.pth')}")


if __name__ == "__main__":
    main()
