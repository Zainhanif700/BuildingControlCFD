"""
Physics-only training loop for GNOT on the real room.

No simulation data anywhere -- exactly like pino_parametric_3d_test.py and
Alexander's own train_parametric_multi_window_tanh.py. The network is
checked against the Navier-Stokes + CO2 transport equations at random
points, with random (t, V1..V8, N_people) resampled every iteration.

Physical constants below are copied from Alexander's own
config_multi_window_tanh.yaml so our physics matches his setup (useful for
later cross-checking against his trained model):
    nu=0.01, rho=1.0, diffusivity=0.005, emission_per_person=1.15e-4,
    sigma=2.5, breathing_height=1.10, tau_ramp=2.0
"""
import os
import time
import torch

from gnot_model import GNOTOperator
from point_sampler import (
    sample_interior, sample_walls, sample_doors, sample_windows, sample_ic,
    sample_columns_surface,
    ROOM_X, ROOM_Y, ROOM_Z, NUM_WINDOWS, CO2_SOURCE_SIGMA,
)

# --- physical constants (matching Alexander's config exactly) ---
NU = 0.01
RHO = 1.0
DIFFUSIVITY = 0.005
EMISSION_PER_PERSON = 1.15e-4
SIGMA = CO2_SOURCE_SIGMA  # single source of truth lives in point_sampler.py now
BREATHING_HEIGHT = 1.10
TAU_RAMP = 2.0
SOURCE_X = (ROOM_X[0] + ROOM_X[1]) / 2
SOURCE_Y = (ROOM_Y[0] + ROOM_Y[1]) / 2

# --- training config ---
# NOTE: these are much smaller than Alexander's own point counts (8000
# interior, etc.) on purpose. His trainer uses a plain MLP; ours uses
# cross-attention, and computing the Laplacian terms needs SECOND-order
# autograd through that attention -- measured to need roughly 4x more GPU
# memory per point than a plain MLP would.
#
# RE-MEASURED after adding the multi-octave Fourier feature encoding (see
# FourierFeatures in gnot_model.py): the extra sin/cos nonlinearities roughly
# DOUBLE the second-order-autograd memory cost per interior point (~7.2MB/pt
# now vs ~3.75MB/pt before), measured directly via torch.cuda.max_memory_allocated()
# on the RTX A2000 (12GB) by sweeping POINTS_INTERIOR in isolation:
#   200 -> 1.45GB, 500 -> 3.60GB, 1000 -> 7.18GB, 1500 -> OOM
# Also confirmed empirically that ONLY the interior/Laplacian term (physics_loss)
# is expensive -- walls/columns/windows/doors/IC only need FIRST-order autograd
# (the curl trick), so they cost almost nothing by comparison: running the full
# pipeline with POINTS_INTERIOR=1000 and walls/columns/windows/doors/IC all at
# their original (larger) sizes below still peaked at exactly 7.18GB, stable
# with no growth across 10 iterations -- so only POINTS_INTERIOR was reduced
# here (1500 -> 1000), everything else kept at full size for training quality.
POINTS_INTERIOR = 1000
POINTS_WALLS = 600
POINTS_COLUMNS_PER = 40   # x 4 columns = 160 -- no-slip on the columns' curved surfaces
POINTS_WINDOWS_PER = 40   # x 8 windows = 320
POINTS_DOORS = 200
POINTS_IC = 400
MAX_ITERS = 20000
LOG_EVERY = 10
CKPT_EVERY = 1000
LR = 1e-3

# FIX (found by verification): CO2 values are tiny (~0.02) compared to
# velocity (~0.3-1 m/s), so when combined into one physics loss, the CO2
# residual got numerically drowned out and the network defaulted to an
# overly smooth, wrong spatial pattern instead of the correct small,
# localized source bump. This is a well-documented PINN failure mode
# ("loss imbalance" / "spectral bias" -- see literature).
#
# A first version of this fix used a FIXED weight of 100.0 -- arbitrary, not
# principled. A SECOND version adaptively weighted based on the ratio of raw
# LOSS VALUES (ns_loss / co2_loss, EMA-smoothed) -- this was ALSO wrong and
# caused a real training collapse: the CO2 residual looks spuriously tiny at
# initialization because most randomly-sampled interior points are far from
# the small Gaussian CO2 source, so the averaged MSE residual is near-zero
# before the network has learned anything. That drove the weight to its
# 10000 ceiling within ~100 iterations and the network collapsed to the
# trivial zero-velocity solution by iteration ~6500 (confirmed from the
# training log: NS/Walls/Doors/IC all hit exactly 0.0, which is what a
# motionless room trivially satisfies, while Windows loss stayed high since
# it demands nonzero inflow velocity that the collapsed solution can't give).
#
# FIX (literature-grounded, per Wang, Teng & Perdikaris 2021, "Understanding
# and Mitigating Gradient Flow Pathologies in Physics-Informed Neural
# Networks", SIAM J. Sci. Comput. 43(5)): weight by GRADIENT NORMS w.r.t. the
# shared network parameters, not raw loss values. Gradient norms reflect how
# hard a loss term actually pulls on the shared parameters -- they don't have
# the "looks small because of sparse sampling" blind spot that loss values do.
# Their Algorithm 2.1: lambda_hat = max|grad(L_ns)| / mean|grad(L_co2)|,
# EMA-smoothed with alpha=0.1 (their recommended value; higher alpha here than
# the old 0.01 since gradient norms are a much more reliable signal, so faster
# adaptation is safe).
CO2_WEIGHT_MIN = 1.0
CO2_WEIGHT_MAX = 10_000.0    # clamp range so a bad ratio can't destabilize training
CO2_WEIGHT_EMA_ALPHA = 0.1   # Wang et al. 2021's recommended EMA rate
CO2_WEIGHT_WARMUP_ITERS = 500  # keep weight=1.0 until the network has learned
# *something* first -- early-training gradients (like early loss ratios) are
# noisy/unreliable, and the literature on curriculum/staged PINN training
# (e.g. causality-based and R3 adaptive-sampling methods) supports delaying
# aggressive reweighting until training has stabilized a bit.
GRAD_CLIP_MAX_NORM = 10.0  # defense-in-depth: caps how much any single
# iteration's combined gradient can move the shared trunk, regardless of root
# cause. Standard practice in general deep learning (RNN/transformer training)
# and reported as a complementary safeguard in recent PINN adaptive-weighting
# work; there's no single canonical value for PINNs specifically, so this is
# a permissive, not tightly-tuned, default.


def compute_param_grads(loss, params, retain_graph):
    """torch.autograd.grad (NOT .backward()) so we get each loss term's
    gradient in isolation, without touching .grad / accumulating -- needed to
    compare gradient MAGNITUDES between loss terms before deciding how to
    combine them (Wang et al.'s gradient-norm weighting)."""
    return torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)


def gradnorm_weight_update(grad_ns, grad_co2, prev_weight):
    """lambda_hat = max|grad(L_ns)| / mean|grad(L_co2)|, EMA-smoothed --
    see the module-level comment above for why this replaces the earlier
    loss-VALUE ratio."""
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

# --- version tag: keeps checkpoints/figures from different physics/model
# revisions from ever being confused with each other. Bump this any time the
# physics loss, model architecture, or point sampling meaningfully changes.
#   v1_smooth_co2  -- original run: plain random-Fourier query encoding (scale=1.0),
#                     fixed CO2_LOSS_WEIGHT=100.0. Trained to 20k iters. Diagnosed
#                     (via closed-window test) to predict a physically-impossible
#                     room-wide smooth CO2 gradient instead of a localized source --
#                     KNOWN BAD, kept only for before/after comparison.
#   v2_co2_fix     -- multi-octave NeRF-style Fourier features (literature-grounded
#                     frequency band) + adaptive EMA CO2 loss weighting. Current.
VERSION = "v2_co2_fix"

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints", VERSION)
os.makedirs(CKPT_DIR, exist_ok=True)


def grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def get_velocity_and_derivs(model, x, y, z, t, V, N_people):
    """Forward pass + curl trick (via the model's OWN velocity_from_potential
    method -- not a re-implemented copy -- so training and inference can
    never silently drift apart if the curl-trick formula ever changes).
    x,y,z,t must have requires_grad=True."""
    A1, A2, A3, C, p = model(x, y, z, t, V, N_people)
    u, v, w = model.velocity_from_potential(A1, A2, A3, x, y, z)
    return u, v, w, C, p


def physics_loss(model, device):
    """Returns (ns_loss, co2_loss) SEPARATELY -- see the module-level comment
    above (CO2_WEIGHT_* constants) for why these are no longer combined into
    one number."""
    x, y, z, t, V, N_people = sample_interior(POINTS_INTERIOR, device)
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True); t.requires_grad_(True)

    u, v, w, c, p = get_velocity_and_derivs(model, x, y, z, t, V, N_people)

    # first derivatives needed for convection + pressure gradient
    du_dx, du_dy, du_dz, du_dt = grad(u, x), grad(u, y), grad(u, z), grad(u, t)
    dv_dx, dv_dy, dv_dz, dv_dt = grad(v, x), grad(v, y), grad(v, z), grad(v, t)
    dw_dx, dw_dy, dw_dz, dw_dt = grad(w, x), grad(w, y), grad(w, z), grad(w, t)
    dc_dx, dc_dy, dc_dz, dc_dt = grad(c, x), grad(c, y), grad(c, z), grad(c, t)
    dp_dx, dp_dy, dp_dz = grad(p, x), grad(p, y), grad(p, z)

    # second derivatives (Laplacians) for viscosity/diffusion terms
    d2u = grad(du_dx, x) + grad(du_dy, y) + grad(du_dz, z)
    d2v = grad(dv_dx, x) + grad(dv_dy, y) + grad(dv_dz, z)
    d2w = grad(dw_dx, x) + grad(dw_dy, y) + grad(dw_dz, z)
    d2c = grad(dc_dx, x) + grad(dc_dy, y) + grad(dc_dz, z)

    conv_u = u * du_dx + v * du_dy + w * du_dz
    conv_v = u * dv_dx + v * dv_dy + w * dv_dz
    conv_w = u * dw_dx + v * dw_dy + w * dw_dz
    conv_c = u * dc_dx + v * dc_dy + w * dc_dz

    res_u = du_dt + conv_u + (1.0 / RHO) * dp_dx - NU * d2u
    res_v = dv_dt + conv_v + (1.0 / RHO) * dp_dy - NU * d2v
    res_w = dw_dt + conv_w + (1.0 / RHO) * dp_dz - NU * d2w

    # CO2 source: Gaussian around room center at breathing height, scaled by occupancy
    dist2 = (x - SOURCE_X) ** 2 + (y - SOURCE_Y) ** 2 + (z - BREATHING_HEIGHT) ** 2
    S = N_people * EMISSION_PER_PERSON * torch.exp(-dist2 / (SIGMA ** 2))
    res_c = dc_dt + conv_c - DIFFUSIVITY * d2c - S

    ns_loss = (res_u ** 2).mean() + (res_v ** 2).mean() + (res_w ** 2).mean()
    co2_loss = (res_c ** 2).mean()
    return ns_loss, co2_loss


def walls_loss(model, device):
    # planar room faces (walls, floor, ceiling; door/window openings excluded)
    x, y, z, t, V, N_people = sample_walls(POINTS_WALLS, device)
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True)
    u, v, w, c, p = get_velocity_and_derivs(model, x, y, z, t, V, N_people)
    loss = (u ** 2).mean() + (v ** 2).mean() + (w ** 2).mean()

    # FIX (found by audit): the 4 columns are solid, floor-to-ceiling pillars --
    # no-slip must also hold on their curved surfaces, or nothing stops the
    # network from predicting flow straight through them.
    xc, yc, zc, tc, Vc, Nc = sample_columns_surface(POINTS_COLUMNS_PER, device)
    xc.requires_grad_(True); yc.requires_grad_(True); zc.requires_grad_(True)
    uc, vc, wc, cc, pc = get_velocity_and_derivs(model, xc, yc, zc, tc, Vc, Nc)
    loss = loss + (uc ** 2).mean() + (vc ** 2).mean() + (wc ** 2).mean()
    return loss


def windows_loss(model, device, co2_weight):
    x, y, z, t, V, N_people, window_idx = sample_windows(POINTS_WINDOWS_PER, device)
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True)
    u, v, w, c, p = get_velocity_and_derivs(model, x, y, z, t, V, N_people)

    V_at_point = V.gather(1, window_idx)  # (B,1) -- this point's own window's speed
    target_v = -V_at_point * torch.tanh(3.0 * t / TAU_RAMP)  # inflow into the room (-y direction)

    # c=0 (clean air in) gets the same adaptive CO2 weighting as the interior
    # residual, for the same reason -- otherwise it's numerically tiny next to
    # the velocity terms and gets neglected during training.
    return (u ** 2).mean() + ((v - target_v) ** 2).mean() + (w ** 2).mean() + co2_weight * (c ** 2).mean()


def doors_loss(model, device):
    x, y, z, t, V, N_people = sample_doors(POINTS_DOORS, device)
    _, _, _, _, p = model(x, y, z, t, V, N_people)
    return (p ** 2).mean()


def ic_loss(model, device, co2_weight):
    x, y, z, t, V, N_people = sample_ic(POINTS_IC, device)
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True)
    u, v, w, c, p = get_velocity_and_derivs(model, x, y, z, t, V, N_people)
    return (u ** 2).mean() + (v ** 2).mean() + (w ** 2).mean() + (p ** 2).mean() + co2_weight * (c ** 2).mean()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = GNOTOperator().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GNOT parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    params = list(model.parameters())

    # Adaptive CO2 loss weight -- starts at a neutral 1.0 and is rebalanced
    # every iteration (after a warm-up) by gradnorm_weight_update(). Tracked
    # as running state across iterations, so it must live here in main(),
    # not as a module-level constant.
    co2_weight = 1.0

    # NOTE on memory: each loss term below is backward()-ed IMMEDIATELY after
    # being computed (instead of summing all 5 into one `total` and calling
    # backward() once at the end). Gradients still accumulate correctly into
    # .grad either way -- the only difference is that this way, each term's
    # computation graph (which can be large: curl-trick + second derivatives
    # through the attention layers) is freed right after its own backward()
    # instead of all 5 graphs being held in memory simultaneously. This cut
    # peak GPU memory a lot and fixed an out-of-memory error we hit even
    # though nvidia-smi showed several GB still free (classic "several
    # medium graphs at once" problem, not a hard memory ceiling).
    start = time.time()
    for it in range(MAX_ITERS + 1):
        optimizer.zero_grad()

        # L_ns and L_co2 both come from the SAME forward pass in physics_loss()
        # (one model(...) call at the same interior points, then split into a
        # tuple) -- so they share the same underlying computation graph, unlike
        # walls/windows/doors/ic below which each do their own independent
        # forward pass.
        #
        # We use torch.autograd.grad (not .backward()) for these two so we can
        # inspect each term's gradient magnitude BEFORE combining them (needed
        # for gradnorm_weight_update -- see module-level comment for why).
        # retain_graph=True on the first call keeps the shared graph alive for
        # the second; the second call (default retain_graph=False) frees it
        # afterward, same peak-memory behavior as separate backward() calls.
        L_ns, L_co2 = physics_loss(model, device)
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

        L_walls = walls_loss(model, device)
        L_walls.backward()

        L_windows = windows_loss(model, device, co2_weight)
        L_windows.backward()

        L_doors = doors_loss(model, device)
        L_doors.backward()

        L_ic = ic_loss(model, device, co2_weight)
        L_ic.backward()

        # Defense-in-depth: cap the combined gradient's norm before stepping,
        # regardless of root cause (see GRAD_CLIP_MAX_NORM comment above).
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_MAX_NORM)

        optimizer.step()

        # Rebalance the CO2 weight for the NEXT iteration using gradient-norm
        # ratios (Wang et al. 2021), not raw loss values -- see module-level
        # comment for why the loss-value version caused a training collapse.
        # Held at a neutral 1.0 during the warm-up window.
        if it >= CO2_WEIGHT_WARMUP_ITERS:
            co2_weight = gradnorm_weight_update(grad_ns, grad_co2, co2_weight)

        total_val = (L_ns.item() + co2_weight * L_co2.item() + L_walls.item()
                     + L_windows.item() + L_doors.item() + L_ic.item())

        if it % LOG_EVERY == 0:
            elapsed = time.time() - start
            speed = (it + 1) / elapsed if elapsed > 0 else 0.0
            print(f"[Iter {it:05d}/{MAX_ITERS}] Total={total_val:.5f} | "
                  f"NS={L_ns.item():.5f} CO2(raw)={L_co2.item():.6f} CO2_weight={co2_weight:.2f} CO2(weighted)={co2_weight * L_co2.item():.5f} "
                  f"Walls={L_walls.item():.5f} Windows={L_windows.item():.5f} Doors={L_doors.item():.5f} "
                  f"IC={L_ic.item():.5f} | {speed:.2f} it/s")

        if it % CKPT_EVERY == 0 and it > 0:
            ckpt_path = os.path.join(CKPT_DIR, f"gnot_{VERSION}_iter{it}.pth")
            torch.save({"iter": it, "version": VERSION, "co2_weight": co2_weight,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict()}, ckpt_path)
            print(f"  -> saved checkpoint: {ckpt_path}")

    final_path = os.path.join(CKPT_DIR, f"gnot_{VERSION}_final.pth")
    torch.save({"iter": MAX_ITERS, "version": VERSION, "co2_weight": co2_weight,
                "model_state": model.state_dict()}, final_path)
    print(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
