"""
Built-in base flow that meets the window and door boundary conditions exactly; the network adds a correction.
"""
import math

import torch

from point_sampler import ROOM_X, ROOM_Y, ROOM_Z, WINDOWS, DOORS, COLUMNS, NUM_WINDOWS, TAU_RAMP

LX, LY, LZ = ROOM_X[1] - ROOM_X[0], ROOM_Y[1] - ROOM_Y[0], ROOM_Z[1] - ROOM_Z[0]
EPS_WINDOW = 0.10
EPS_DOOR = 0.20
DOOR_STRIP = 2.0
COLUMN_BLEND = [0.5, 0.5, 0.5, 0.25]
PHI_L = 0.5
PHI_OPEN_AMP = 2.0
V_OPEN = 0.1
_I1 = 0.5
DOOR_HEIGHT = DOORS[0][3]
assert all(abs(d[3] - DOOR_HEIGHT) < 1e-9 and d[2] == ROOM_Z[0] for d in DOORS), "doors must share height, start at floor"
assert all(w[2] == ROOM_Z[0] and w[3] == ROOM_Z[1] for w in WINDOWS), "windows must be floor-to-ceiling"
_DOOR_HN = DOOR_HEIGHT - EPS_DOOR / 2


def _S(t):
    """Smooth step from 0 to 1 on [0, 1]."""
    t = t.clamp(0.0, 1.0)
    return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)


def _Sint(t):
    t = t.clamp(0.0, 1.0)
    return t ** 5 * (7 - 14 * t + 10 * t ** 2 - 2.5 * t ** 3)


def _R(s):
    """Integral of the smooth step."""
    return torch.where(s <= 0, torch.zeros_like(s), torch.where(s < 1, _Sint(s), _I1 + (s - 1)))


def _plateau(x, a, b, e):
    return _S((x - a) / e) - _S((x - (b - e)) / e)


def _cumplat(x, a, b, e):
    """Integral of the smooth plateau."""
    return e * (_R((x - a) / e) - _R((x - (b - e)) / e))


def ramp(t):
    return torch.tanh(3.0 * t / TAU_RAMP)


_CONST_CACHE = {}


def _consts(ref):
    """Opening geometry as tensors on the right device."""
    key = (str(ref.device), ref.dtype)
    if key not in _CONST_CACHE:
        mk = lambda vals: torch.tensor(vals, device=ref.device, dtype=ref.dtype).view(1, -1)
        _CONST_CACHE[key] = (mk([w[0] for w in WINDOWS]), mk([w[1] for w in WINDOWS]),
                             mk([d[0] for d in DOORS]), mk([d[1] for d in DOORS]))
    return _CONST_CACHE[key]


def _F_top(x, V, rt):
    """Cumulative inflow per unit height along the window wall (x from 0), shape (B,1)."""
    wa, wb, _, _ = _consts(x)
    kappa = (wb - wa) / (wb - wa - EPS_WINDOW)
    return rt * (V * kappa * _cumplat(x, wa, wb, EPS_WINDOW)).sum(dim=1, keepdim=True)


def _q(V, rt):
    wa, wb, _, _ = _consts(V)
    return rt * (V * (wb - wa)).sum(dim=1, keepdim=True)


def _door_weights(alpha):
    return torch.cat([alpha, 1.0 - alpha], dim=1)


def _F_bot(x, q, alpha):
    _, _, da, db = _consts(x)
    return q * (_door_weights(alpha) * _cumplat(x, da, db, EPS_DOOR) / (db - da - EPS_DOOR)).sum(dim=1, keepdim=True)


def _dF_bot(x, q, alpha):
    _, _, da, db = _consts(x)
    return q * (_door_weights(alpha) * _plateau(x, da, db, EPS_DOOR) / (db - da - EPS_DOOR)).sum(dim=1, keepdim=True)


def _G(z):
    """Height profile integral of the door outflow."""
    return z - (LZ / _DOOR_HN) * (z - EPS_DOOR * _R((z - DOOR_HEIGHT + EPS_DOOR) / EPS_DOOR))


PSI_BLEND = "linear"
PSI_TURN_DEPTH = 2.0

JET_SMOOTH = True
JET_EDGE_W = 0.3
JET_SPREAD_L = 1.0


def _lncosh(u):
    """ln cosh(u) without overflow."""
    return u + torch.nn.functional.softplus(-2.0 * u, beta=1.0, threshold=50.0) - math.log(2.0)


def _F_top_smooth(x, V, rt, w):
    """Cumulative window inflow along the wall, with smooth edges."""
    wa, wb, _, _ = _consts(x)
    x0 = torch.full_like(x, ROOM_X[0])
    x1 = torch.full_like(x, ROOM_X[1])

    def C(xx):
        return 0.5 * w * (_lncosh((xx - wa) / w) - _lncosh((xx - wb) / w))
    frac = (C(x) - C(x0)) / (C(x1) - C(x0))
    return rt * (V * (wb - wa) * frac).sum(dim=1, keepdim=True)


def _F_bot_smooth(x, q, alpha, w):
    """Cumulative door outflow along the wall, with smooth edges."""
    _, _, da, db = _consts(x)
    x0 = torch.full_like(x, ROOM_X[0])
    x1 = torch.full_like(x, ROOM_X[1])

    def C(xx):
        return 0.5 * w * (_lncosh((xx - da) / w) - _lncosh((xx - db) / w))
    frac = (C(x) - C(x0)) / (C(x1) - C(x0))
    return q * (_door_weights(alpha) * frac).sum(dim=1, keepdim=True)


def through_flow_potential(x, y, z, t, V, alpha, blend=None, turn_depth=None,
                           smooth=None, edge_w=None, spread_l=None):
    """Vector potential of the built-in base flow."""
    blend = PSI_BLEND if blend is None else blend
    turn_depth = PSI_TURN_DEPTH if turn_depth is None else turn_depth
    smooth = JET_SMOOTH if smooth is None else smooth
    edge_w = JET_EDGE_W if edge_w is None else edge_w
    spread_l = JET_SPREAD_L if spread_l is None else spread_l
    rt = ramp(t)
    q = _q(V, rt)
    F_top = _F_top(x, V, rt)
    if smooth:
        beta = _S((ROOM_Y[1] - y) / spread_l)
        F_top = F_top + beta * (_F_top_smooth(x, V, rt, edge_w) - F_top)
    F_bot = _F_bot(x, q, alpha)
    if smooth:
        beta_d = _S((y - ROOM_Y[0] - DOOR_STRIP) / spread_l)
        F_bot = F_bot + beta_d * (_F_bot_smooth(x, q, alpha, edge_w) - F_bot)
    if blend == "linear":
        eta = (y - ROOM_Y[0]) / LY
    elif blend == "jet":
        eta = _S((y - ROOM_Y[0]) / turn_depth)
    else:
        raise ValueError(f"unknown blend {blend!r} (use 'linear' or 'jet')")
    psi = (1.0 - eta) * F_bot + eta * F_top
    for (cx, cy, r, _, _), rb in zip(COLUMNS, COLUMN_BLEND):
        xc = torch.full_like(x, cx)
        c = _F_top(xc, V, rt) if cy > ROOM_Y[0] + LY / 2 else _F_bot(xc, q, alpha)
        d = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2 + 1e-12) - r
        blend = 1.0 - _S(d / rb)
        psi = psi + blend * (c - psi)
    m = 1.0 - _S((y - ROOM_Y[0]) / DOOR_STRIP)
    chi = _dF_bot(x, q, alpha) * m * _G(z - ROOM_Z[0])
    return chi, psi


def solid_distance_phi(x, y, z, V):
    """Smooth distance function: 0 on walls, columns and closed windows, about 1 inside the room."""
    L = PHI_L
    phi = (torch.tanh((x - ROOM_X[0]) / L) * torch.tanh((ROOM_X[1] - x) / L)
           * torch.tanh((z - ROOM_Z[0]) / L) * torch.tanh((ROOM_Z[1] - z) / L))
    wa, wb, da, db = _consts(x)
    zdoor = 1.0 - _S((z - (DOOR_HEIGHT - EPS_DOOR)) / EPS_DOOR)
    door_hole = _plateau(x, da, db, EPS_DOOR).sum(dim=1, keepdim=True) * zdoor
    win_hole = (torch.tanh(V / V_OPEN) * _plateau(x, wa, wb, EPS_WINDOW)).sum(dim=1, keepdim=True)
    phi = phi * torch.tanh((y - ROOM_Y[0]) / L + PHI_OPEN_AMP * door_hole)
    phi = phi * torch.tanh((ROOM_Y[1] - y) / L + PHI_OPEN_AMP * win_hole)
    for cx, cy, r, _, _ in COLUMNS:
        phi = phi * torch.tanh(((x - cx) ** 2 + (y - cy) ** 2 - r ** 2) / (2 * r * L))
    return phi


DOOR1_SHARE_PER_WINDOW = [0.5935, 0.5654, 0.5526, 0.4914, 0.4491, 0.4355, 0.3989, 0.3857]
L_CO2_WINDOW = 0.3


def alpha_potential(V):
    """Potential-flow share of the air leaving through door 1."""
    wa, wb, _, _ = _consts(V)
    r = torch.tensor(DOOR1_SHARE_PER_WINDOW, device=V.device, dtype=V.dtype).view(1, -1)
    q = V * (wb - wa)
    tot = q.sum(dim=1, keepdim=True)
    return torch.where(tot > 0, (q * r).sum(dim=1, keepdim=True) / tot.clamp_min(1e-12),
                       torch.full_like(tot, 0.5))


def co2_window_factor(x, y, V):
    """Factor that forces c = 0 on the open windows."""
    wa, wb, _, _ = _consts(x)
    hole = (torch.tanh(V / V_OPEN) * _plateau(x, wa, wb, EPS_WINDOW)).sum(dim=1, keepdim=True)
    return 1.0 - hole * (1.0 - torch.tanh((ROOM_Y[1] - y) / L_CO2_WINDOW))


def target_window_flux(V, t):
    """Prescribed outward flux of each window (negative = inflow), (B, 8)."""
    areas = torch.tensor([(b - a) * (d - c) for a, b, c, d in WINDOWS], device=V.device, dtype=V.dtype)
    return -ramp(t) * V * areas.view(1, -1)


def bp_velocity_laplacian(x, y, z, t, V, alpha):
    """Laplacian of the base-flow velocity (no gradient to the network)."""
    x, y, z, t = (q.detach().clone().requires_grad_(True) for q in (x, y, z, t))
    with torch.enable_grad():
        chi, psi = through_flow_potential(x, y, z, t, V.detach(), alpha.detach())

        def g(f, v):
            if not f.requires_grad:
                return torch.zeros_like(v)
            r = torch.autograd.grad(f, v, grad_outputs=torch.ones_like(f), create_graph=True, allow_unused=True)[0]
            return torch.zeros_like(v) if r is None else r
        ub = (g(psi, y), g(chi, z) - g(psi, x), -g(chi, y))
        lap = torch.cat([g(g(c, x), x) + g(g(c, y), y) + g(g(c, z), z) for c in ub], 1)
    return lap.detach()
