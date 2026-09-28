"""
v19_throughflow: EXACT velocity boundary conditions by construction.

Velocity of the model:   u = curl( B_p + s(V) * phi * A_net )

B_p   = (chi, 0, psi): an analytic "through-flow" vector potential. Its flow carries
        exactly the prescribed air in through every open window and out through the
        doors, and has exactly zero normal velocity on ALL solid surfaces (walls,
        floor, ceiling, columns, closed windows, the wall strips above the doors).
phi   = smooth distance-like function, 0 on all solid surfaces (incl. closed windows),
        > 0 in the room and at open windows / doors.
A_net = the network's vector potential (s(V) = RMS(V)/V_MAX, as since v9).

Why: every soft-BC version leaked the air through the walls (v13, v16) or collapsed
to no flow when the leak was penalised strongly (v17). Here that is impossible:
  - u stays exactly divergence-free (it is a curl);
  - curl(s*phi*A) has zero NORMAL velocity wherever phi = 0 (grad(phi) is parallel
    to the normal there), and by Stokes' theorem zero NET flux through every opening
    (its rim lies on the wall, where phi = 0): the network can reshape the flow but
    can never leak through a solid surface or change how much air enters or leaves;
  - B_p fixes the net fluxes exactly: window k delivers V_k * tanh(3t/TAU_RAMP) * A_k,
    and the doors release the same total. A zero-flow collapse is not representable.
An exact hard wall constraint WITHOUT B_p was rejected (zero through-flow, Stokes);
this particular-solution + correction structure is the standard remedy (Lagaris et
al. 1998; lifting functions as in Lee et al. 2026, arXiv:2608.08114).

Construction of B_p (verified numerically: normal velocity on every solid surface
<= 2e-8 m/s, window/door fluxes exact, see smoke-test stage 0f):
  psi(x, y): transfinite interpolation between the cumulative inflow along the window
      wall, F_top(x), and the cumulative outflow along the door wall, F_bot(x):
      psi = (1-y/Ly) F_bot + (y/Ly) F_top. curl(0,0,psi) = (dpsi/dy, -dpsi/dx, 0) is a
      z-independent horizontal flow: w = 0 (floor/ceiling exact), psi = 0 / q on the end
      walls (exact), v = -F_top' at the window wall (the prescribed inflow profile).
      Around each column psi is blended smoothly to a constant, so the column surface
      is a streamline (exact). Each column's constant equals the wall streamline value
      next to it and its blend ring never reaches a SOLID wall segment where psi varies,
      nor the door x-ranges at y = 0 (chi's cancellation needs psi(x,0) = F_bot(x)).
      Checked for this geometry (independent review): column 1 meets y = LY at x in
      [13.537, 14.163] (F_top constant; window 7 ends 13.50), column 2 meets y = 0 at
      [5.097, 6.383], column 4 (blend 0.25 m) meets y = 0 at [13.425, 14.235] (door 2 ends
      13.38). Column 3's ring touches window 3's edge taper [5.30, 5.40] -- an opening, not a
      wall, and psi(5.40) = its constant, so window 3's flux stays exact.
  chi(x, y, z) = F_bot'(x) m(y) G(z): the doors are only 2.11 m high, but the psi-flow
      reaches the door wall at full height. curl(chi,0,0) = (0, dchi/dz, -dchi/dy)
      redirects that air down into the door opening inside a DOOR_STRIP-deep strip:
      at y = 0 the total normal velocity is -F_bot'(x) g(z) (g = door-height profile,
      0 above the door); G(0) = G(Lz) = 0 keeps floor/ceiling exact.
  Door split: F_bot = q [alpha C1(x) + (1-alpha) C2(x)] with a LEARNED alpha(V, t) in
      (0,1) -- the correction cannot change per-door fluxes (Stokes), so the split is a
      degree of freedom of B_p, decided by the physics through the loss.
  Opening profiles are smooth plateaus (C3 7th-order smoothstep edges, widths EPS_WINDOW
  and EPS_DOOR), renormalised so each opening carries exactly its full-area flux.
  ASSUMPTION to state in the thesis: the net inflow of each window is fixed to
  V_k * A_k (uniform-speed equivalent) and the flow enters normal to the window; the
  network shapes everything else.
"""
import math

import torch

from point_sampler import ROOM_X, ROOM_Y, ROOM_Z, WINDOWS, DOORS, COLUMNS, NUM_WINDOWS, TAU_RAMP

LX, LY, LZ = ROOM_X[1] - ROOM_X[0], ROOM_Y[1] - ROOM_Y[0], ROOM_Z[1] - ROOM_Z[0]
EPS_WINDOW = 0.10         # m, edge width of the window inflow profile
EPS_DOOR = 0.20           # m, edge width of the door outflow profile (x and top edge)
DOOR_STRIP = 2.0          # m, depth of the strip where chi redirects flow into the doors. Review of
# v19: B_p's own NS residual concentrates in this strip (1 m: training-mix mean |res|^2 232, 77% of it
# here); 2 m halves the turning rate -> 88. Safe: chi's x-support (door x-ranges) touches no column.
COLUMN_BLEND = [0.5, 0.5, 0.5, 0.25]   # m; column 4 is 6.5 cm from the door wall
PHI_L = 0.5               # m, length scale of phi near solid surfaces (phi ~ distance/PHI_L)
PHI_OPEN_AMP = 2.0        # phi at the centre of an opening ~ tanh(2) = 0.96 (interior-like)
V_OPEN = 0.1              # m/s, window 'open' scale in phi: hole amplitude tanh(V_k / V_OPEN)
_I1 = 0.5                 # integral of the smoothstep over [0, 1]
DOOR_HEIGHT = DOORS[0][3]
assert all(abs(d[3] - DOOR_HEIGHT) < 1e-9 and d[2] == ROOM_Z[0] for d in DOORS), "doors must share height, start at floor"
assert all(w[2] == ROOM_Z[0] and w[3] == ROOM_Z[1] for w in WINDOWS), "windows must be floor-to-ceiling"
_DOOR_HN = DOOR_HEIGHT - EPS_DOOR / 2    # integral over z of the door-height plateau


def _S(t):
    """C3 smoothstep 35t^4 - 84t^5 + 70t^6 - 20t^7 on [0,1], 0 below, 1 above; S(1-t) = 1-S(t)."""
    t = t.clamp(0.0, 1.0)
    return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)


def _Sint(t):
    t = t.clamp(0.0, 1.0)
    return t ** 5 * (7 - 14 * t + 10 * t ** 2 - 2.5 * t ** 3)


def _R(s):
    """integral_{-inf}^{s} S."""
    return torch.where(s <= 0, torch.zeros_like(s), torch.where(s < 1, _Sint(s), _I1 + (s - 1)))


def _plateau(x, a, b, e):
    return _S((x - a) / e) - _S((x - (b - e)) / e)


def _cumplat(x, a, b, e):
    """integral_{-inf}^{x} plateau; total = b - a - e."""
    return e * (_R((x - a) / e) - _R((x - (b - e)) / e))


def ramp(t):
    return torch.tanh(3.0 * t / TAU_RAMP)


_CONST_CACHE = {}


def _consts(ref):
    """Opening geometry as (1, K) tensors on ref's device/dtype (all 8 windows / both doors
    are evaluated at once by broadcasting -- far fewer autograd ops than a Python loop).
    Cached per (device, dtype) to avoid host->device copies on every call."""
    key = (str(ref.device), ref.dtype)
    if key not in _CONST_CACHE:
        mk = lambda vals: torch.tensor(vals, device=ref.device, dtype=ref.dtype).view(1, -1)
        _CONST_CACHE[key] = (mk([w[0] for w in WINDOWS]), mk([w[1] for w in WINDOWS]),
                             mk([d[0] for d in DOORS]), mk([d[1] for d in DOORS]))
    return _CONST_CACHE[key]


def _F_top(x, V, rt):
    """Cumulative inflow per unit height along the window wall (x from 0), shape (B,1)."""
    wa, wb, _, _ = _consts(x)
    kappa = (wb - wa) / (wb - wa - EPS_WINDOW)            # full-area flux despite the tapered edges
    return rt * (V * kappa * _cumplat(x, wa, wb, EPS_WINDOW)).sum(dim=1, keepdim=True)


def _q(V, rt):
    wa, wb, _, _ = _consts(V)
    return rt * (V * (wb - wa)).sum(dim=1, keepdim=True)


def _door_weights(alpha):
    return torch.cat([alpha, 1.0 - alpha], dim=1)            # (B, 2)


def _F_bot(x, q, alpha):
    _, _, da, db = _consts(x)
    return q * (_door_weights(alpha) * _cumplat(x, da, db, EPS_DOOR) / (db - da - EPS_DOOR)).sum(dim=1, keepdim=True)


def _dF_bot(x, q, alpha):
    _, _, da, db = _consts(x)
    return q * (_door_weights(alpha) * _plateau(x, da, db, EPS_DOOR) / (db - da - EPS_DOOR)).sum(dim=1, keepdim=True)


def _G(z):
    """integral_0^z (1 - g), g = door-height profile normalised to integral LZ; G(0) = G(LZ) = 0."""
    return z - (LZ / _DOOR_HN) * (z - EPS_DOOR * _R((z - DOOR_HEIGHT + EPS_DOOR) / EPS_DOOR))


# How psi blends from the door wall (F_bot) to the window wall (F_top) across the room depth:
#   "linear" (v19, v20): psi = (1 - y/LY) F_bot + (y/LY) F_top -- the part of the inflow that must
#       reach the OTHER door drifts slowly across the whole room and then runs down a straight band
#       to that door (x ~ 12.5-13.4 m for door 2). OpenFOAM (W1 at 1 m/s) shows instead a straight
#       jet from the window and a turn ALONG THE DOOR WALL; the network correction did not remove
#       the band (a straight band barely violates the equations -> almost no loss signal).
#   "jet" (candidate for v21): psi = F_bot + (F_top - F_bot) * S(y / PSI_TURN_DEPTH): above the
#       turning layer psi = F_top(x) (straight jets from the windows across the room, v = -F_top'),
#       and the redistribution towards the doors happens in a layer of depth PSI_TURN_DEPTH next to
#       the door wall (u = (F_top - F_bot) S'). Generic jet behaviour (a jet crosses the room and
#       spreads along the wall it hits), not a fit to one OpenFOAM case. Same exactness: psi = F_bot
#       at y = 0, F_top at y = LY, 0 / q on the end walls, columns blended to the adjacent wall value.
# The DEFAULT stays "linear": v20 checkpoints must be evaluated with the field they were trained
# with. A version that switches must bump gnot_model.MODEL_FORMAT.
PSI_BLEND = "linear"
PSI_TURN_DEPTH = 2.0      # m (same order as DOOR_STRIP, where chi turns the flow into the doors)

# v22_smoothjet: SMOOTH window-jet profile inside the room.
# Found in v21 (diagnose_ns_residual.py, W1 1 m/s): 99.4% of the momentum loss came from 1% of the
# points, ALL at x = 2.10-2.68 m (= window 1) and at every depth y and height z; their residual was
# 100% the viscous term of B_p ITSELF. Cause: the linear psi blend carries the window's sharp inflow
# profile (0.1 m C3 edges, EPS_WINDOW) as a sheet through the whole room depth -> nu*lap(v) ~ nu*V/eps^2
# along two 0.1 m-thin shear layers, far finer than the network can represent (Fourier features down
# to ~0.6 m) -> an irreducible, spiky loss (v21 NS loss 25-1700 between iterations) that swamps the
# gradient; the same points give ~8 at nu = 0.01, the whole measured NS loss of that case (v19 too).
# Fix: away from the window wall, psi uses the SAME window fluxes with a smooth tanh-edged profile
# (length JET_EDGE_W); the sharp profile is kept at the window wall and blended out over JET_SPREAD_L:
#     F_top_in(x, y) = F_top(x) + beta(y) (F_top_smooth(x) - F_top(x)),  beta = S((LY - y)/JET_SPREAD_L).
# Exactness is unchanged: beta = beta' = 0 at y = LY (psi and dpsi/dy at the window wall as before);
# F_top_smooth(0) = 0 and F_top_smooth(LX) = q exactly (normalised), so the end walls stay exact; the
# door wall (y = 0) and chi are untouched; column constants are unchanged (their rings meet the walls,
# where psi is unchanged). Only the free interior shape of the through-flow changes (a spread jet).
JET_SMOOTH = True
JET_EDGE_W = 0.3          # m, tanh edge length of the smooth jet profile (v22 check 2: best vs OpenFOAM at nu 0.01)
JET_SPREAD_L = 1.0        # m, depth from the window wall over which the sharp profile blends out


def _lncosh(u):
    return torch.log(torch.cosh(u))   # |u| <= LX / JET_EDGE_W ~ 31 here: no overflow in float32


def _F_top_smooth(x, V, rt, w):
    """Cumulative inflow per unit height with smooth tanh edges of length w, (B,1): window k
    contributes V_k (b_k - a_k) (C_k(x) - C_k(0)) / (C_k(LX) - C_k(0)), C_k = integral of
    (tanh((x-a)/w) - tanh((x-b)/w)) / 2 -> exactly 0 at x = ROOM_X[0], exactly q/rt at ROOM_X[1]."""
    wa, wb, _, _ = _consts(x)
    x0 = torch.full_like(x, ROOM_X[0])
    x1 = torch.full_like(x, ROOM_X[1])

    def C(xx):
        return 0.5 * w * (_lncosh((xx - wa) / w) - _lncosh((xx - wb) / w))
    frac = (C(x) - C(x0)) / (C(x1) - C(x0))
    return rt * (V * (wb - wa) * frac).sum(dim=1, keepdim=True)


def _F_bot_smooth(x, q, alpha, w):
    """Door-side counterpart of _F_top_smooth: cumulative outflow q [alpha, 1-alpha] with tanh edges
    of length w, exactly 0 at ROOM_X[0] and q at ROOM_X[1]. (v22, check 1: with only the window side
    smoothed, B_p's viscous term fell 7.5x but was the SAME for every edge length -- the door sheet,
    carried with weight (1 - y/LY) through the room with its 0.2 m edges, and the window-wall zone.)"""
    _, _, da, db = _consts(x)
    x0 = torch.full_like(x, ROOM_X[0])
    x1 = torch.full_like(x, ROOM_X[1])

    def C(xx):
        return 0.5 * w * (_lncosh((xx - da) / w) - _lncosh((xx - db) / w))
    frac = (C(x) - C(x0)) / (C(x1) - C(x0))
    return q * (_door_weights(alpha) * frac).sum(dim=1, keepdim=True)


def through_flow_potential(x, y, z, t, V, alpha, blend=None, turn_depth=None,
                           smooth=None, edge_w=None, spread_l=None):
    """(chi, psi), each (B,1): B_p = (chi, 0, psi). Inputs physical units; alpha in (0,1), (B,1).
    blend / turn_depth default to PSI_BLEND / PSI_TURN_DEPTH (see above); smooth / edge_w /
    spread_l to JET_SMOOTH / JET_EDGE_W / JET_SPREAD_L (v22)."""
    blend = PSI_BLEND if blend is None else blend
    turn_depth = PSI_TURN_DEPTH if turn_depth is None else turn_depth
    smooth = JET_SMOOTH if smooth is None else smooth
    edge_w = JET_EDGE_W if edge_w is None else edge_w
    spread_l = JET_SPREAD_L if spread_l is None else spread_l
    rt = ramp(t)
    q = _q(V, rt)
    F_top = _F_top(x, V, rt)
    if smooth:   # v22: the interior sees a smooth jet profile; exact sharp profile at the window wall
        beta = _S((ROOM_Y[1] - y) / spread_l)
        F_top = F_top + beta * (_F_top_smooth(x, V, rt, edge_w) - F_top)
    F_bot = _F_bot(x, q, alpha)
    if smooth:   # v22: same for the door sheet, but only BEYOND the door-turn strip (y > DOOR_STRIP),
        # where chi = 0: inside the strip chi's sharp door profile must meet the sharp psi (exact
        # door outflow at y = 0 needs psi(x, 0) = F_bot(x)); beta_d = 0 for y <= DOOR_STRIP.
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
        d = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2 + 1e-12) - r   # +1e-12: finite gradient at the axis (inside the solid)
        blend = 1.0 - _S(d / rb)
        psi = psi + blend * (c - psi)
    m = 1.0 - _S((y - ROOM_Y[0]) / DOOR_STRIP)
    chi = _dF_bot(x, q, alpha) * m * _G(z - ROOM_Z[0])
    return chi, psi


def solid_distance_phi(x, y, z, V):
    """phi (B,1): 0 on walls/floor/ceiling/columns and on CLOSED windows (V_k = 0),
    ~ distance/PHI_L near them, ~1 in the room, ~0.96 at the centre of doors and open windows."""
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


# v20: potential-flow share of each window's air that leaves through door 1, from
# compute_door_split.py (lap(theta) = 0, unit inflow at window k, no flux elsewhere, theta = 0 at
# both doors). dx = 0.12 m values: at this spacing both doors (and all windows) are resolved with
# equal cell counts (at dx = 0.1 door 2 came out 10% larger than door 1, biasing r_k by ~0.004).
# Across dx = 0.075-0.2 the shares scatter by about +-0.006 (independent review) -- small, and the
# model adds a learned correction on top (gnot_model.door_split).
DOOR1_SHARE_PER_WINDOW = [0.5935, 0.5654, 0.5526, 0.4914, 0.4491, 0.4355, 0.3989, 0.3857]
L_CO2_WINDOW = 0.3        # m, depth over which the CO2 window factor recovers from 0 to 1


def alpha_potential(V):
    """v20: potential-flow door-1 share of the scenario's through-flow, (B,1):
    sum_k V_k A_k r_k / sum_k V_k A_k (0.5 when all windows are closed -- then there is no flow)."""
    wa, wb, _, _ = _consts(V)
    r = torch.tensor(DOOR1_SHARE_PER_WINDOW, device=V.device, dtype=V.dtype).view(1, -1)
    q = V * (wb - wa)
    tot = q.sum(dim=1, keepdim=True)
    return torch.where(tot > 0, (q * r).sum(dim=1, keepdim=True) / tot.clamp_min(1e-12),
                       torch.full_like(tot, 0.5))


def co2_window_factor(x, y, V):
    """v20: omega(x, y; V) in [0, 1] with omega = 0 on the core of every OPEN window (exact clean
    inflow, c = 0) and omega = 1 on closed windows, walls and away from the window wall:
        omega = 1 - hole(x, V) * (1 - tanh((LY - y) / L_CO2_WINDOW)),
        hole  = sum_k tanh(V_k / V_OPEN) * plateau_k(x)      (same opening profile as the inflow).
    At y = LY, omega = 1 - hole: 0 where the window is fully open (hole -> 1 for V_k >> V_OPEN),
    partial only in the 0.1 m edge taper, where the inflow velocity also tapers to 0. Multiplies
    the CO2 output -- it keeps C = 0 at t = 0 and C proportional to N exactly.
    Exactness on the window core: the remainder is (1 - tanh(V_k/0.1)) C -- 0.5% at V = 0.3 m/s,
    1e-4 at 0.5 m/s, exactly 0 in float32 for V >= 1 m/s. So: 'exact for V >~ 0.5 m/s'; slower
    windows (and the edge tapers) are still covered by the soft c = 0 term in windows_loss.
    omega = 1 exactly (and d omega/dn = 0) on walls, mullions and closed windows, so the CO2
    no-flux conditions and the closed-room case are unchanged (independent review)."""
    wa, wb, _, _ = _consts(x)
    hole = (torch.tanh(V / V_OPEN) * _plateau(x, wa, wb, EPS_WINDOW)).sum(dim=1, keepdim=True)
    return 1.0 - hole * (1.0 - torch.tanh((ROOM_Y[1] - y) / L_CO2_WINDOW))


def target_window_flux(V, t):
    """Prescribed outward flux of each window (negative = inflow), (B, 8)."""
    areas = torch.tensor([(b - a) * (d - c) for a, b, c, d in WINDOWS], device=V.device, dtype=V.dtype)
    return -ramp(t) * V * areas.view(1, -1)


def bp_velocity_laplacian(x, y, z, t, V, alpha):
    """v22: lap(curl B_p), (B,3), DETACHED (own leaf copies of the inputs, no graph to the network):
    the viscous term of the analytic through-flow alone. Used to down-weight the momentum residual
    where B_p itself carries an irreducible viscous error (train_gnot.physics_loss)."""
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
