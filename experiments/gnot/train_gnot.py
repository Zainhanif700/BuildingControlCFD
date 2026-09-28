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
import math
import os
import time
import torch

from gnot_model import GNOTOperator, NONDIM_CHECKPOINT_KEY, MODEL_FORMAT_KEY, MODEL_FORMAT
from point_sampler import (
    sample_interior, sample_walls, sample_doors, sample_windows, sample_ic,
    sample_columns_surface, _generate_interior_batch,
    ROOM_X, ROOM_Y, ROOM_Z, NUM_WINDOWS, CO2_SOURCE_SIGMA, BREATHING_HEIGHT,
    EMISSION_PER_PERSON, S_REF, C_REF, TAU_RAMP, COLUMNS, N_PEOPLE_MAX, WINDOWS, DOORS,
)

# --- physical constants (matching Alexander's config exactly) ---
NU = 0.01
RHO = 1.0
DIFFUSIVITY = 0.005
# EMISSION_PER_PERSON now lives in point_sampler.py (v8_nondim) -- imported above.
SIGMA = CO2_SOURCE_SIGMA  # single source of truth lives in point_sampler.py now
# TAU_RAMP now lives in point_sampler.py (v8_nondim) -- imported above.
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
MAX_ITERS = 50000  # v21: 10k (nu 0.1) + 10k (nu 0.03) + 30k (nu 0.01); v10-v20 used 20000
LOG_EVERY = 10
CKPT_EVERY = 1000
LR = 1e-3

# --- v11_latedecay: two-phase learning-rate schedule ---
# Constant LR while the physics is being learned, then a cosine decay to refine
# it. v10 was validated against a grid-converged finite-difference reference
# (milestones/v10_hardic/README.md): 2% error at the source but ~14% relative
# L2 over the breathing-height plane, concentrated in the tails. A late decay
# is the standard way to reduce that remaining error: a constant LR of 1e-3
# keeps Adam's steps too large to settle fine detail. v6_lr_decay failed
# because it decayed from iteration 0 and throttled CO2 BEFORE CO2 had been
# learned at all (CO2 stayed on the trivial solution then); by iter 20000 of
# v10 CO2 is learned and validated, so decaying only from there is a
# different, well-founded experiment.
#
# RESUME_FROM: continue from v10's iter-20000 checkpoint (it stores the Adam
# state, so there is no optimizer restart transient) instead of redoing
# v10's 20000 iterations. Set to None to train from scratch -- the SAME
# schedule then applies (constant LR to LR_DECAY_START, decay afterwards).
#
# DEFAULTS RESET after v11 (see version history): v11 showed the late decay does
# NOT reduce the error, so the live defaults are v10's validated setup again --
# fresh run, constant LR. The mechanism is kept for future experiments:
#   v11 used: RESUME_FROM = "checkpoints/v10_hardic/gnot_v10_hardic_iter20000.pth",
#             MAX_ITERS = 30000, LR_DECAY_START = 20000, LR_MIN = 1e-5
RESUME_FROM = None       # e.g. "checkpoints/<version>/gnot_<version>_iter20000.pth" (relative to this file)
LR_DECAY_START = None    # None = constant LR throughout; an iteration number = cosine decay after it
LR_MIN = 1e-5


# --- v21_single: single scenario + viscosity curriculum + exponential LR decay ---------------
# Literature (deep review before v21; details in the VERSION history below):
#  * Wang, Sankaran, Wang & Perdikaris 2023 (Expert's Guide, arXiv:2308.08468, sec. 7.5): trained
#    directly at high Re, PINNs are unstable / converge to wrong solutions; a Reynolds-number
#    curriculum (cavity Re 100 -> 400 -> 1000 -> 3200, each stage initialised from the previous one,
#    most iterations in the last stage) reaches 15.8% at Re 3200. PirateNets (arXiv:2402.00326): same
#    idea, 4.2%. Krishnapriyan et al. 2021 (NeurIPS): curriculum ~2 orders of magnitude lower error.
#  * Expert's Guide: LR 1e-3 with EXPONENTIAL decay (rate 0.9 every 2000 steps), no weight decay.
#  * Every successful high-Re cavity result is ONE scenario; data-free physics-informed operators at
#    Re ~ 500 fail (PINO Kolmogorov Re 500, 0 data: 74% error, arXiv:2111.03794 Table 7).
# Viscosity instead of Re: our geometry and inflow are fixed by the scenario, so Re = U L / nu is
# lowered by raising nu (identical meaning). Stage boundaries are INCLUSIVE: nu_at(10000) = 0.1,
# so the iter-10000 checkpoint belongs to the nu = 0.1 stage and can be checked against the
# OpenFOAM case with nu = 0.1.
NU_SCHEDULE = [(10000, 0.1), (20000, 0.03), (None, NU)]   # (last iteration of the stage, nu)
LR_EXP_DECAY = (0.9, 2000)   # (rate, steps): lr = LR * rate**(it/steps); None -> lr_at's v11 logic


def nu_at(it):
    """Kinematic viscosity used in the momentum residual at iteration `it` (v21 curriculum).
    Ends at NU (the physical value that all checks and OpenFOAM use)."""
    for last, nu in NU_SCHEDULE:
        if last is None or it <= last:
            return nu
    return NU


def lr_at(it):
    """LR for iteration `it`. v21: LR_EXP_DECAY set -> LR * rate**(it/steps) (Expert's Guide).
    Otherwise v11 logic: LR until LR_DECAY_START, then cosine from LR down
    to exactly LR_MIN at MAX_ITERS. LR_DECAY_START=None -> constant LR."""
    if LR_EXP_DECAY is not None:
        rate, steps = LR_EXP_DECAY
        return LR * rate ** (it / steps)
    if LR_DECAY_START is None or it <= LR_DECAY_START:
        return LR
    frac = min(1.0, (it - LR_DECAY_START) / (MAX_ITERS - LR_DECAY_START))
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1.0 + math.cos(math.pi * frac))

# REVERTED (v7_higher_co2_weight): v6_lr_decay tested a cosine LR decay to
# address the CO2 magnitude oscillation seen in v5 (see
# milestones/v6_lr_decay/README.md for full data). Result: it did NOT fix
# the oscillation -- instead, CO2's magnitude stayed pinned near zero for
# the entire second half of training (never reaching even v5's peak
# magnitude), while the "improvement" in closed-window velocity noise was
# actually just normal variance in an already-fixed baseline, not a real
# fix. Diagnosis: velocity and CO2 share one optimizer/LR, but CO2 needs
# more/larger updates for longer (spectral bias, localized source term);
# decaying the shared LR to stabilize the (already-fine) velocity term
# likely choked off CO2's ability to keep improving. Back to a constant LR
# here. (v8 note: the deeper cause turned out to be input/output scaling --
# see point_sampler.py's S_REF/C_REF comment.)

# v8 NOTE: everything in this block is HISTORY of the adaptive CO2 weighting
# (v2-v7). As of v8 the real cause of the CO2 "imbalance" was found to be
# missing non-dimensionalization (see point_sampler.py S_REF/C_REF), and the
# adaptive weight is switched OFF -- see USE_ADAPTIVE_CO2_WEIGHT below.
#
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
# SECOND ATTEMPT AT THIS FIX ALSO FAILED (smoke-tested before committing to a
# full run, per our "verify everything" rule): updating the gradient-norm
# weight EVERY iteration created a feedback loop -- a big weight jump
# immediately reshapes the shared trunk, which changes the next gradient
# reading, producing another big jump, compounding into outright divergence
# (observed: Windows/IC losses spiking into the hundreds within 25 iterations
# once the weight started climbing). Wang et al. 2021's actual algorithm
# updates this weight only PERIODICALLY (their paper anneals every several
# iterations, not every single step) for exactly this reason -- updating
# every step was our own oversimplification, not what the paper does.
CO2_WEIGHT_MIN = 1.0
CO2_WEIGHT_MAX = 200.0      # v8_nondim: REVERTED to v5's value (v7 had raised
# it to 500, but v7 was never run -- after literature review, that ceiling
# turned out to be this project's own invention; Wang et al. 2021's
# Algorithm 1 has no cap at all). Kept at v5's 200 so v8 differs from the
# v5 baseline (which has full diagnostic data) in ONE thing only: the
# non-dimensionalization. (Only used if USE_ADAPTIVE_CO2_WEIGHT is True,
# which it is NOT in v8 -- see that flag's comment for why.)
CO2_WEIGHT_EMA_ALPHA = 0.1   # Wang et al. 2021's recommended EMA rate
CO2_WEIGHT_WARMUP_ITERS = 500  # keep weight=1.0 until the network has learned
# *something* first -- early-training gradients (like early loss ratios) are
# noisy/unreliable, and the literature on curriculum/staged PINN training
# (e.g. causality-based and R3 adaptive-sampling methods) supports delaying
# aggressive reweighting until training has stabilized a bit.
CO2_WEIGHT_UPDATE_EVERY = 100  # only recompute/EMA-update the weight every N
# iterations (matches Wang et al.'s actual periodic-annealing practice) --
# give the network time to adapt to a given weight before revising it again,
# instead of compounding a fresh, possibly-noisy weight every single step.
GRAD_CLIP_MAX_NORM = 10.0  # defense-in-depth: caps how much any single
# iteration's combined gradient can move the shared trunk, regardless of root
# cause. Standard practice in general deep learning (RNN/transformer training)
# and reported as a complementary safeguard in recent PINN adaptive-weighting
# work; there's no single canonical value for PINNs specifically, so this is
# a permissive, not tightly-tuned, default.


# v8_nondim: ADAPTIVE CO2 WEIGHTING SWITCHED OFF (co2_weight fixed at 1.0).
# Found by independent audit of v8: the adaptive weight above uses Wang et
# al. 2021's max|grad NS| / mean|grad CO2| statistic, which is >> 1 BY
# CONSTRUCTION (a maximum over ~320k parameters divided by a mean) even when
# the two losses are perfectly balanced. Before v8 that didn't matter -- the
# CO2 loss really was ~1e4x too small, so the weight pinned at its ceiling
# either way. After v8's scaling, the CO2 loss is O(0.1), comparable to NS,
# and that same statistic would push CO2 up to 200x ABOVE velocity, likely
# wrecking the already-working velocity field. The later, refined recipe in
# the same group's Expert's Guide (Wang, Sankaran, Wang & Perdikaris 2023,
# arXiv:2308.08468) balances by making the gradient NORMS of the weighted
# terms equal -- a statistic that is ~1 when losses are balanced. So for v8:
#   - co2_weight = 1.0 (the plain unweighted PINN baseline, appropriate once
#     all residuals are O(1) after non-dimensionalization);
#   - the norm-equalizing weight the Guide's rule WOULD choose,
#     ||grad L_ns|| / ||grad L_co2||, is computed and LOGGED only (column
#     "guide_w"), so this run also produces evidence on whether balancing is
#     needed at all: if guide_w stays within roughly 0.1-10, equal weights are
#     fine; if it sits far outside, turn norm balancing on in the next run.
USE_ADAPTIVE_CO2_WEIGHT = False


def guide_norm_ratio(grad_ns, grad_co2):
    """||grad L_ns||_2 / ||grad L_co2||_2 over all shared parameters -- the
    weight on L_co2 that would make both terms' gradient norms equal
    (Expert's Guide loss balancing). Diagnostic only in v8."""
    ns_sq = sum((g ** 2).sum() for g in grad_ns if g is not None)
    co2_sq = sum((g ** 2).sum() for g in grad_co2 if g is not None)
    if not torch.is_tensor(co2_sq) or not torch.is_tensor(ns_sq) or co2_sq.item() < 1e-30:
        return float("nan")
    return (ns_sq.sqrt() / co2_sq.sqrt()).item()


# v13: evaluate ALL CO2 losses at full occupancy N = N_MAX.
# Diagnosed from v12 (milestones/v12_linear_n/README.md): with C exactly
# proportional to N (v12), each sample's CO2 residual is (N/N_MAX) * r_hat with
# r_hat independent of N, so the squared CO2 loss silently weighted every sample
# by (N/N_MAX)^2 -- on average only 1/3 of the intended weight, and an
# effective sample size of ~0.56 (samples with few people contributed almost
# nothing). That matches v12's late guide_w of ~20 and its stalled accuracy at
# 20-50 people. Because C is EXACTLY linear in N and the flow does not depend
# on N, the residual at N = N_MAX is exactly r_hat: training on it loses
# nothing and gives every sample full weight, with no division by small N.
# Applied to physics_loss (NS part unaffected: flow is N-independent),
# windows_loss, ic_loss, co2_boundary_loss and the trivial-floor reference.
CO2_LOSS_AT_FULL_OCCUPANCY = True


def _co2_occupancy(N_people):
    """Occupancy used inside the loss functions -- see CO2_LOSS_AT_FULL_OCCUPANCY."""
    return torch.full_like(N_people, N_PEOPLE_MAX) if CO2_LOSS_AT_FULL_OCCUPANCY else N_people


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
#   v2_co2_fix     -- multi-octave NeRF-style Fourier features (SEPARABLE per-axis,
#                     literature-grounded frequency band) + adaptive EMA CO2 loss
#                     weighting. Trained to 20k iters (checkpoints iter16000/final).
#                     Diagnosed (closed-window test) to still show a CO2 "band"
#                     artifact -- localized in one axis but not the other. KNOWN
#                     PARTIAL FIX, kept for before/after comparison.
#   v3_isotropic_ff -- (tested via standalone partial_run scripts, not through this
#                     file's main() yet) isotropic random Fourier features (fixing
#                     the v2 separable-encoding limitation) + source_proximity input
#                     feature (see gnot_model.py's QueryEncoder). Diagnosed to fix
#                     the "band" shape but the CO2 field stayed undertrained
#                     (near-zero/negative) at 3k-10k iterations with pure uniform
#                     interior sampling.
#   v4_source_sampling / v4b_source_sampling_tuned -- (tested via partial_run_v4.py,
#                     not through this file's main() yet) adds source-concentrated
#                     interior sampling on top of v3 (see point_sampler.py's
#                     SOURCE_SAMPLE_FRAC/XY_STD/Z_STD) so the network sees the CO2
#                     source region far more often per iteration. v4 (frac=0.4,
#                     std=sigma) still showed a rotated band artifact; v4b (frac=0.6,
#                     std=sigma/2, tighter concentration) is being evaluated now.
#   v5_closed_window_fix -- (partial_run_v4.py + resume_v5.py, not through this
#                     file's main() yet) adds correlated closed/partial-closed
#                     window-scenario oversampling (point_sampler.py's
#                     CLOSED_SCENARIO_FRAC/PARTIAL_CLOSED_SCENARIO_FRAC) on top of
#                     v4b. CONFIRMED: fixes the spurious closed-window velocity
#                     artifact, durable across iter10000-20000, no regression on
#                     open-window behavior. STILL OPEN: CO2 source localization --
#                     magnitude oscillates late in training instead of converging
#                     (see milestones/v5_closed_window_fix/README.md).
#   v6_lr_decay     -- first run through this file's own main(), now with a
#                     cosine learning-rate decay schedule (since removed -- see milestones/v6_lr_decay/) added
#                     on top of everything in v5, to test whether the late-training
#                     CO2 oscillation was caused by a constant LR overshooting a
#                     near-good solution. Fresh 20k-iteration run (not a resume of
#                     v5), so it's directly comparable to v5's own checkpoint
#                     history at matching iteration counts. RESULT (see
#                     milestones/v6_lr_decay/README.md): did NOT fix the CO2
#                     oscillation -- CO2 magnitude stayed pinned near zero for the
#                     whole second half of training instead. The "improved"
#                     closed-window velocity numbers were just normal variance in
#                     an already-fixed v5 baseline, not a real additional fix.
#                     LR decay likely choked off CO2's still-needed large updates
#                     while stabilizing the already-converged velocity term.
#   v7_higher_co2_weight -- NEVER RUN. Reverted the v6 LR decay and raised
#                     CO2_WEIGHT_MAX 200->500; abandoned before training after
#                     literature review showed the ceiling itself was this
#                     project's own invention (no basis in Wang et al. 2021).
#                     Its best-loss checkpointing was kept.
#   v8_nondim       -- ROOT-CAUSE FIX: non-dimensionalization (Wang, Sankaran,
#                     Wang & Perdikaris 2023, arXiv:2308.08468, step 1). Found by
#                     checking that the CO2 residual loss in EVERY prior run sat
#                     at ~3e-6 from iteration ~10 on -- exactly the loss of the
#                     trivial constant-C solution (mean(S^2) = 3.08e-6 over our
#                     sampling distribution). Inputs t, V, N_people and token
#                     positions are now scaled to ~[0,1] inside the model; C is
#                     output as C_REF * C_hat; the CO2 residual is divided by
#                     S_REF and the boundary/IC CO2 terms by C_REF (see
#                     point_sampler.py S_REF/C_REF). Two changes that follow
#                     directly from the scaling (both found by independent
#                     audit): (a) a second time input tanh(3t/TAU_RAMP), since
#                     scaling t by 120 s alone would squash the 2 s inflow ramp
#                     and risk regressing the already-working velocity; (b) the
#                     adaptive CO2 weight is OFF (fixed 1.0), since its max/mean
#                     statistic is >>1 by construction and would over-weight the
#                     now properly-scaled CO2 term ~200x (Expert's Guide's
#                     norm-balancing weight is logged as guide_w instead).
#                     Otherwise identical to v5: constant LR, 100%-fresh
#                     sampling (Stage 1 pool switched off).
#                     RESULT (iter 1000-2000): first run ever to leave the
#                     trivial CO2 solution (CO2(scaled) ~0.01 vs floor 0.093;
#                     guide_w ~1, balanced; velocity healthy; C(source) rising
#                     0.012 -> 0.021) -- but the CO2 maximum sat pinned against
#                     the door wall (11.9, 0.1) with a room-wide band.
#                     LATER (iter 3000-5000, see milestones/v8_nondim/README.md):
#                     the wall pinning resolved on its own, but CO2 barely grows
#                     in time (~3% of the physical rate). co2_residual_breakdown.py
#                     showed why: with windows CLOSED the network balances the
#                     source by CONVECTION (0.332 vs S 0.318) through a spurious
#                     ~0.07 m/s flow, not by accumulation (dc/dt 0.012).
#   v9_zeroflow_bc  -- v8 + two exact-physics fixes, each with its own diagnostic:
#                     (1) HARD zero-flow constraint: velocity potential multiplied
#                     by s(V)=RMS(V)/V_MAX in gnot_model.py, so u=0 EXACTLY when all
#                     windows are closed -- removes the loophole above (check: the
#                     breakdown's closed-window u.grad(c) must be 0 and dc/dt must
#                     carry the source). (2) the missing CO2 boundary conditions:
#                     no-flux dc/dn=0 on walls/floor/ceiling/columns, zero-gradient
#                     outflow at doors (co2_boundary_loss; logged as CO2_BC).
#                     RESULT (iter 7000, milestones/v9_zeroflow_bc/README.md): FIRST
#                     physically correct CO2 -- closed windows balanced by
#                     accumulation (dc/dt 0.233 vs S 0.318, u.grad(c)=0), growth peak
#                     at (7.57,4.92) vs true (7.76,4.58), half-max width 21/11 vs
#                     physical 18/10, growth ~81% of the physical value. Remaining:
#                     constant offset C(t=0) ~ -0.039 everywhere (soft IC too cheap).
#   v10_hardic      -- v9 + HARD initial condition: C = C_REF*(t/T_MAX)*C_hat in
#                     gnot_model.py, so C(t=0)=0 exactly (Lagaris et al. 1998).
#                     Single change vs v9. The CO2 part of ic_loss is now identically
#                     0 (kept; harmless). Check: probe_co2_time.py C(t=0) row = 0,
#                     far-corner CO2 ~0 instead of -0.04, growth unchanged or better.
#                     RESULT: validated against a grid-converged finite-difference
#                     reference (fd_reference_closed_room.py): 2% error at the source,
#                     identical peak location, ~14% relative L2 over the breathing-height
#                     plane (error in the tails). No training collapse.
#   v11_latedecay   -- v10 + a LATE learning-rate decay: resume v10 at iter 20000 (with
#                     its Adam state) and cosine-decay LR 1e-3 -> 1e-5 over iters
#                     20001-30000 (see RESUME_FROM / lr_at). Single change vs v10.
#                     Check: FD-reference plane error vs v10's ~14%.
#                     RESULT (code: git 3cb59d8): NEGATIVE. Relative L2 vs the FD
#                     reference at t=60 s: 18.1% (iter 22000), 15.8% (25000), 15.2%
#                     (28000), 15.5% (final) vs v10's 13.9%. The decay settled the
#                     error at ~15% instead of lowering it; checkpoint-to-checkpoint
#                     spread is a few %, so this is 'no measurable change'. The
#                     ~14-15% (in the low-concentration tails) is systematic, not
#                     optimizer noise. v10 remains the best model. Live defaults
#                     reset to v10's setup (RESUME_FROM/LR_DECAY_START = None).
#   v12_linear_n    -- v10 + HARD proportionality of CO2 to occupancy:
#                     C = C_REF*(t/T_MAX)*(N/N_MAX)*C_hat (exact for this model, see
#                     gnot_model.py). Found from validate_closed_room.py on v10 (27
#                     cases): plane L2 11-14% at 50 people, 14-17% at 20, but 47-54% at
#                     5 -- an N-independent error component dominating at low
#                     occupancy. Single change vs v10 (fresh run, constant LR).
#                     Check: validate_closed_room.py. NOTE: with exact linearity the
#                     relative error is IDENTICAL at every N by construction, so 'the
#                     5-people rows improve' is automatic, not a success test. The real
#                     test: is that common error at or below v10's BEST case (11-14% plane
#                     L2 at 50 people) -- i.e. did removing the N-dependence make the
#                     field itself better, not just rescale it?
#                     RESULT (milestones/v12_linear_n/README.md): mixed. Source error
#                     1.9% (best), plane L2 19-26% at EVERY N (5 people: 50% -> 22%), but
#                     worse than v10 at 20-50 people (v10: 11-17%). Cause: CO2 residual
#                     of each sample scaled by N/N_MAX, so the loss weighted samples by
#                     (N/N_MAX)^2 -- ~1/3 of the intended CO2 weight (guide_w ~20 late).
#   v13_fullocc     -- v12 + all CO2 losses evaluated at N = N_MAX
#                     (CO2_LOSS_AT_FULL_OCCUPANCY): exact by linearity, gives every
#                     sample full weight. Single change vs v12. Trivial-floor reference
#                     is 0.279 (= 3 x 0.093). Check: validate_closed_room.py plane L2 at
#                     or below v10's best (11-14%), and guide_w back in ~1-10.
#                     RESULT (milestones/v13_fullocc): plane L2 14.4-17.9% (mean 15.9%)
#                     at every N, guide_w 2-5 -- best all-round model; source ~5-7% low.
#                     diagnose_residual_map.py: error = accumulated residual (corr 0.93);
#                     far field holds 91% of the squared error but only 44% of the loss.
#   v14_uniform     -- v13 + UNIFORM interior sampling (point_sampler.SOURCE_SAMPLE_FRAC
#                     0.6 -> 0), so the loss weights the room like the error metric.
#                     Single change vs v13. Trivial-floor reference becomes 0.0527.
#                     Check: validate_closed_room.py plane L2 below v13's 15.9%;
#                     diagnose_residual_map.py far-field error share down.
#                     RESULT (negative, not a milestone): plane L2 17.8% mean (v13 15.9%),
#                     source -10 to -25% (v13 -5 to -7%). Far field improved, source
#                     region starved (near-source share of error^2 9% -> 61%). Reverted
#                     to SOURCE_SAMPLE_FRAC = 0.6. Same night, lbfgs_probe.py on v13: a
#                     fixed-batch loss 46,000x lower and plane L2 14.6% -> 7.4%, but
#                     held-out loss 2x HIGHER -> the optimizer is a real limit, yet a
#                     second-order fit to one fixed batch overfits. Hence v15.
#   v15_soap        -- v13 setup + SOAP optimizer instead of Adam (OPTIMIZER = "soap").
#                     SOAP = Adam run in the eigenbasis of a Shampoo preconditioner
#                     (Vyas et al. 2024, arXiv:2409.11321); for PINNs it gave large
#                     gains over Adam with points RESAMPLED every step (Wang, Bhartari,
#                     Li & Perdikaris 2025, arXiv:2502.00604) -- i.e. the curvature
#                     benefit L-BFGS showed, without the fixed batch. Settings from that
#                     paper: betas (0.99, 0.999), precondition_frequency 2; weight_decay
#                     0 (the library default 0.01 is not used in PINN training). Same LR
#                     1e-3, same clipping, same 20k iterations -- single change vs v13.
#                     Check: plane L2 below v13's 15.9% at every N and source error
#                     not worse; expect each iteration ~1.1-1.5x slower.
#                     RESULT (mid-run, CPU validation): 1.16 vs 1.18 it/s (SOAP nearly free).
#                     iter 5000: plane L2 21.8% (v13 at 5000: 29.4%) -- faster start; but
#                     iter 10000: 21.6% (v13 at 10000: 15.4%) -- stalled. Not better than v13.
#                     Confounded by the sampler bug found the same day (see v16).
#   v16_shuffle     -- v13 setup (Adam) + BUG FIX in point_sampler.sample_scenario: scenario
#                     rows are now shuffled. Before, scenario TYPE was tied to point ORDER
#                     (uniform | all-closed | partial blocks vs. window 1..8 / wall-by-face /
#                     uniform-then-source point layout): windows 5-6 never trained open, the
#                     x-max wall + floor + column 3 never had no-slip with flow, far-field
#                     interior never saw closed windows. Found by check_physics_consistency.py
#                     (air leaks through walls, ~0 through doors) + crosscheck_windows.py
#                     (inflow error 13% on the training mix, 59% on random V at the same
#                     points). Affects v4-v15. (Started as v16_shuffle, stopped after a few
#                     minutes to add the fixes below -- run as v16_fixes.)
#   v16_fixes       -- v16_shuffle + the CONFIRMED bugs from an independent code audit
#                     (all correctness fixes, no tuning):
#                     (a) closed windows were CO2 sinks (c = 0 at every window): now c = 0 only
#                         at OPEN windows, no-flux dc/dn = 0 at closed ones (windows_loss,
#                         co2_boundary_loss) -- matches the FD reference ('noflux').
#                     (b) ic_loss no longer forces p = 0 at t = 0 (contradicted the start-up
#                         momentum balance; no pressure IC in incompressible flow).
#                     (c) source-concentrated points are rejected outside the room instead of
#                         clamped (~5% of interior points sat exactly on the floor).
#                     (d) sample_walls: points per face proportional to net area; floor/
#                         ceiling points inside column footprints rejected. sample_ic:
#                         uniform instead of source-concentrated.
#                     Check (post_training_checks.sh): closed-room plane L2 vs v13's 15.9%;
#                     window inflow / door outflow / leakage / CO2 budget in the level-3 check.
#                     RESULT at iter 5000 (stopped there for v17): closed room 25.6% (v13 at
#                     5000: 29.4%); window inflow 70-98% of target at 1-5 m/s (v13: 6-60%), but
#                     85-93% error at 0.2 m/s; door outflow still ~0.01 m^3/s -- air leaves
#                     through the WALLS (soft no-slip too cheap for a spread-out leak).
#   v17_fluxbc      -- v16_fixes + flux-based scaling of the velocity BCs (see the v17_fluxbc
#                     comment above walls_loss): wall/column normal velocity scaled by
#                     A_SOLID / Q_scale(V,t) (term = (leak flux / inflow)^2), window inflow error
#                     relative to the window's own speed. An exact hard wall constraint was
#                     checked numerically and rejected (Stokes: zero net flux per opening).
#                     Check: level 3 door outflow ~ window inflow, leak < 10% of inflow; 0.2 m/s
#                     rows like the 1-5 m/s rows; closed room not worse than v16 at the same iter.
#                     RESULT at iter 5000 (NEGATIVE, stopped at ~7000): the flow COLLAPSED -- inflow
#                     0.3% of target with one window open (v16: 71%), 22% with all open; door
#                     outflow still only ~5% of what enters. Windows loss flat at 0.35-0.40 from
#                     iter 3000, guide_w fell to ~0.1-0.2 (NS gradient ~0 = almost no flow). Closed
#                     room 18.3% (v16 at 5000: 25.6%), but with no flow to learn, not a v17 success.
#   v18_fluxonly    -- v16_fixes + ONLY the flux-scaled leak term (USE_RELATIVE_WINDOW_LOSS = False,
#                     absolute window error as in v16). Isolates the two v17 changes: if v18 keeps
#                     the inflow and the leak shrinks, the relative window error caused the collapse;
#                     if not, the cause is deeper (door jets / representability) -> exact route.
#                     NOT RUN: a literature review (Sun et al. 2020 CMAME: soft BCs 'completely
#                     wrong' for internal flows; Daw et al. 2023 / Rohrhofer et al. 2023: trivial
#                     solutions as loss minima; Lagaris 1998 / Lee et al. 2026: particular solution
#                     + correction) ranked the soft route low -> went straight to the exact route.
#   v19_throughflow -- v16_fixes losses (absolute window error) + EXACT velocity BCs by
#                     construction (gnot_model.MODEL_FORMAT 'v19_throughflow', throughflow.py):
#                     u = curl(B_p + s(V)*phi*A). B_p = analytic through-flow potential: exact
#                     inflow V_k*A_k per open window, exact zero normal velocity on walls, floor,
#                     ceiling, columns, closed windows and above the doors, door outflow = inflow,
#                     learned door split alpha(V,t). phi = 0 on solid surfaces -> the network part
#                     cannot leak or change any opening's net flux (Stokes). Leak and zero-flow
#                     collapse are impossible by construction. Tangential no-slip, window profile,
#                     door p = 0, NS and CO2 stay soft. (The v17 flux leak term is kept but is ~0.)
#                     After an independent review: (i) momentum residual, wall slip and window
#                     tangential terms non-dimensionalised PER SCENARIO by U_ref = door jet speed
#                     (velocity_scale; B_p's own residual would otherwise swamp all terms);
#                     (ii) DOOR_STRIP 1 -> 2 m (halves B_p's residual); (iii) the correction is
#                     multiplied by the ramp -> u(t=0) = 0 exact; (iv) no normal-velocity target at
#                     windows (flux exact; USE_WINDOW_NORMAL_TARGET = False). Old v12-v18 checkpoints
#                     are checked with the git tag code-v12-format.
#                     Check: level 3 leak ~0 and door outflow = inflow (exact, sanity); closed room
#                     vs v13's 15.9%; NS residual level; level 2 CO2 consistency. NOTE: level 3
#                     'winErr' compares with a UNIFORM profile, so the 0.1 m edge taper of B_p shows
#                     up there by design; the net window flux (Q_in vs Q_target) is the exact one.
#                     RESULT at iter 5000 (run continued to 20000): air balance EXACT in every
#                     scenario (inflow = door outflow, leak 0 -- first time); closed room 14.9% (worst
#                     16.2%, already below v13's FINAL 15.9%), source 4.5%, closed CO2 budget +6.5%.
#                     BUT: open-window CO2 budget +50..+230% (CO2 carried IN through the open
#                     windows: soft c = 0 violated), wall slip flat at 12-57% of the window speed
#                     from iter 1000 to 5000, door split barely moved (0.47/0.50/0.53).
#   v20_co2window   -- v19 + three exact/physical fixes for exactly those points:
#                     (a) C x co2_window_factor: c = 0 EXACTLY on open-window cores (gnot_model,
#                         MODEL_FORMAT 'v20_co2window'); (b) door split alpha = sigmoid(logit(
#                         alpha_pot) + learned correction), alpha_pot from potential flow
#                         (compute_door_split.py; W1 -> 0.59 ... W8 -> 0.38 through door 1);
#                     (c) wall slip (and window tangential terms) measured against
#                         max(min(U_ref, window speed), U_ref/3) (slip_scale; at most 9x v19's weight).
#                     c = 0 is exact on open-window cores for V >~ 0.5 m/s (soft term below / edges).
#                     v19 checkpoints: git tag code-v19-format.
#                     Check: level-3 CO2 budget within +-5..10% for open windows; level 2; slip falls;
#                     closed room not worse than v19; air balance stays exact.
#                     RESULT at iter 5000 (NEGATIVE, stopped): CO2 collapsed to almost nothing --
#                     closed-room plane error 95.4%, source -97.6%, closed CO2 budget -83%, rel_CO2
#                     ~0.96, guide_w ~1e-3. Cause: u.grad(omega)*C_hat next to the open windows makes
#                     the CO2 residual huge there; C_hat ~ 0 is the cheapest answer. v19 stays best.
#                     Also found (OpenFOAM, W1 1 m/s, openfoam/): v19's velocity error 74% vs B_p
#                     alone 71% -- the network adds nothing to the flow (correction alignment 0.14,
#                     rel_NS ~ 1): the momentum equation is not learned (diagnose_flow_correction.py).
#   v21_single      -- DIAGNOSTIC STEP after a literature review of that failure (documented regime:
#                     data-free PINNs / PI-operators at Re ~ 300-1000 converge to smooth wrong
#                     solutions; every high-Re success is single-scenario, uses a Re curriculum,
#                     1e5+ iterations and a decaying LR). v19 losses and model, plus:
#                     (a) ONE scenario only: SINGLE_SCENARIO_V = window 1 at 1 m/s (the OpenFOAM
#                         case), point_sampler.FIXED_V; t and N stay random;
#                     (b) viscosity curriculum nu 0.1 -> 0.03 -> 0.01 (NU_SCHEDULE, nu_at);
#                     (c) exponential LR decay 1e-3 * 0.9**(it/2000) (LR_EXP_DECAY), 50k iterations;
#                     (d) v20's CO2 window factor OFF (gnot_model.USE_CO2_WINDOW_FACTOR); v20's
#                         potential-flow door split prior kept.
#                     Question answered: CAN this network learn the jet + recirculation at all?
#                     Check: compare_with_openfoam.py + diagnose_flow_correction.py against the
#                     OpenFOAM case with the SAME nu at each stage end (iter 10000: nu 0.1, 20000:
#                     0.03, final: 0.01). Success = velocity error clearly below B_p's 71% and
#                     alignment clearly above 0.14. Closed-room checks are NOT meaningful for v21
#                     (never trained closed).
#
# IMPORTANT: this VERSION variable (and CKPT_DIR below) is what train_gnot.py's own
# main() uses for a FULL 20k-iteration production run. Bump this to match whichever
# fix combination is confirmed working via the closed-window diagnostic BEFORE
# launching the next full run through this file, so production checkpoints aren't
# mislabeled with stale physics/sampling.
VERSION = "v21_single"
SINGLE_SCENARIO_V = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # v21: W1 1 m/s (= openfoam case W1_1ms); None = mix

# v15: optimizer switch. "adam" = v1-v14 behaviour; "soap" = soap.py (official
# implementation, github.com/nikhilvyas/SOAP, MIT licence, unmodified copy).
OPTIMIZER = "adam"   # v16: back to v13's optimizer so the sampler fix is the only change
SOAP_BETAS = (0.99, 0.999)        # Wang et al. 2025 PINN setting (library default (0.95, 0.95))
SOAP_PRECONDITION_FREQUENCY = 2   # Wang et al. 2025 (library default 10)
SOAP_WEIGHT_DECAY = 0.0           # library default 0.01 -- off, as with Adam in v1-v14


def make_optimizer(params):
    """Build the optimizer selected by OPTIMIZER. Shared by main() and the smoke test."""
    if OPTIMIZER == "adam":
        return torch.optim.Adam(params, lr=LR)
    if OPTIMIZER == "soap":
        from soap import SOAP
        return SOAP(params, lr=LR, betas=SOAP_BETAS, weight_decay=SOAP_WEIGHT_DECAY,
                    precondition_frequency=SOAP_PRECONDITION_FREQUENCY, eps=1e-8)
    raise ValueError(f"unknown OPTIMIZER {OPTIMIZER!r} (use 'adam' or 'soap')")
# main() refuses to start if checkpoints for this VERSION already exist, so an old
# result can never be overwritten by forgetting to bump this.

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


def physics_loss(model, device, nu=NU):
    """Returns (ns_loss, co2_loss) SEPARATELY -- see the module-level comment
    above (CO2_WEIGHT_* constants) for why these are no longer combined into
    one number. nu: viscosity of the momentum residual (v21 curriculum, nu_at(it));
    default = the physical NU."""
    x, y, z, t, V, N_people = sample_interior(POINTS_INTERIOR, device)
    N_people = _co2_occupancy(N_people)  # v13
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

    res_u = du_dt + conv_u + (1.0 / RHO) * dp_dx - nu * d2u
    res_v = dv_dt + conv_v + (1.0 / RHO) * dp_dy - nu * d2v
    res_w = dw_dt + conv_w + (1.0 / RHO) * dp_dz - nu * d2w

    # CO2 source: Gaussian around room center at breathing height, scaled by occupancy
    dist2 = (x - SOURCE_X) ** 2 + (y - SOURCE_Y) ** 2 + (z - BREATHING_HEIGHT) ** 2
    S = N_people * EMISSION_PER_PERSON * torch.exp(-dist2 / (SIGMA ** 2))
    res_c = dc_dt + conv_c - DIFFUSIVITY * d2c - S

    # v19: momentum residual non-dimensionalised PER SCENARIO by U_ref^2 / L_NS (U_ref = the
    # scenario's door jet speed, see velocity_scale) -- the same idea v8 applied to CO2. With the
    # exact through-flow B_p (throughflow.py) the flow speed spans 0.5-17 m/s across scenarios and
    # the raw residual ~U^2: unnormalised, B_p's own residual (training-mix mean ~1e2) would swamp
    # every other term and make the fast scenarios dominate (independent review of v19).
    ns_scale = velocity_scale(V) ** 2 / L_NS
    ns_loss = ((res_u / ns_scale) ** 2).mean() + ((res_v / ns_scale) ** 2).mean() + ((res_w / ns_scale) ** 2).mean()
    # v8_nondim: divide by S_REF so the CO2 residual is O(1) instead of
    # O(6e-3) -- every term in res_c (dc/dt, conv_c, D*lap(c), S) has units
    # of concentration/second, so this is a pure rescaling of the same
    # equation, not a change to the physics. With this, the trivial C=0
    # solution scores ~0.093 (was 3.08e-6), comparable to NS -- so the
    # network finally has a real incentive to leave it.
    co2_loss = ((res_c / S_REF) ** 2).mean()
    return ns_loss, co2_loss


# --- v17_fluxbc: flux-based scaling of the velocity boundary conditions -------------
# Found by check_physics_consistency.py on v16_fixes (iter 5000): air entered through the
# windows (70-98% of the target) but left through the WALLS, not the doors (door outflow
# ~0.01 m^3/s). Cause: the pointwise no-slip loss mean(u^2) makes a leak of ~0.01 m/s cost
# only ~1e-4 per point, yet spread over ~440 m^2 of solid surface it carries as much air as
# the windows. (An exact hard constraint u = curl(phi^2 A) with phi = 0 on the walls was
# checked numerically first and REJECTED: by Stokes' theorem the net flux through every
# opening is then the circulation of phi^2 A around its rim, which is 0 -- no through-flow
# possible without an additional particular solution.)
# Fix = measure the leak as what it physically is, a FLUX relative to the scenario's
# through-flow: the normal velocity at each solid-surface point is scaled by
# A_SOLID / Q_scale(V, t), so the term equals (leaked flux / window inflow)^2 -- a leak as
# large as the inflow costs ~1, the same as a completely wrong window. With the velocity
# exactly divergence-free, no leak means the air MUST leave through the doors.
# Same idea for the window inflow: the error is measured RELATIVE to the window's own
# speed (floor V_REL_FLOOR), so a 0.2 m/s window counts as much as a 5 m/s one (before, the
# loss scaled with V^2: slow openings were ~600x under-weighted, audit suspicion S1;
# v16 at iter 5000 had 85-93% inflow error at 0.2 m/s).
# Both are non-dimensionalizations by the scenario's own scales (as v8 did globally and
# v13 for occupancy), not new loss terms.
_LX, _LY, _LZ = ROOM_X[1] - ROOM_X[0], ROOM_Y[1] - ROOM_Y[0], ROOM_Z[1] - ROOM_Z[0]
WINDOW_AREAS = [(x1 - x0) * (z1 - z0) for x0, x1, z0, z1 in WINDOWS]
A_SOLID = (2 * _LX * _LZ + 2 * _LY * _LZ + 2 * _LX * _LY
           - sum((x1 - x0) * (z1 - z0) for x0, x1, z0, z1 in DOORS) - sum(WINDOW_AREAS)
           - 2 * sum(math.pi * r ** 2 for _, _, r, _, _ in COLUMNS)
           + sum(2 * math.pi * r * (zh - zl) for _, _, r, zl, zh in COLUMNS))  # ~441 m^2 (closed windows excluded)
Q_FLOOR = 0.1        # m^3/s: floor on the flux scale (very slow openings; all-closed rows have u = 0 exactly)
V_REL_FLOOR = 0.1    # m/s: floor on the window-speed scale in the relative inflow error
USE_RELATIVE_WINDOW_LOSS = False   # v18: OFF. v17 (relative window error ON) collapsed to almost no
# flow: zero flow satisfies NS exactly, removes the leak term, and cost only ~0.5 under the relative
# window error (closed windows at weight 1/0.1^2 = 100). With the absolute error (v16), zero flow costs
# ~5, so collapsing is no escape. v18 isolates the flux leak term.
BC_SCALING_WARMUP = 2000   # iterations over which both v17 scalings are blended in (review: at random
# init the flux term is O(10-1000) vs O(1) for the rest and would swamp the PDE terms / push the flow to 0)


A_DOORS = sum((x1 - x0) * (z1 - z0) for x0, x1, z0, z1 in DOORS)   # 4.47 m^2
U_NS_FLOOR = 0.5     # m/s: floor on the per-scenario velocity scale (closed / very slow scenarios)
L_NS = _LZ           # m: length scale of the momentum-residual normalisation (room height)
USE_WINDOW_NORMAL_TARGET = False   # v19: the window inflow FLUX is exact by construction (B_p); a
# uniform-profile target for v would only fight B_p's tapered profile (review). Kept: u = w = 0
# (air enters normal to the window) and c = 0 at open windows.


_ALPHA_PROBE_V = torch.tensor([[3.0] + [0.0] * 7, [0.0] * 7 + [3.0], [3.0] * 8])   # logged door splits


def slip_scale(V):
    """v20: speed scale for the no-slip (tangential) wall terms, (B,1):
    max(min(U_ref, U_win), V_REL_FLOOR), U_win = sum_k V_k A_k / sum_{open k} A_k = mean inflow
    speed of the open windows. v19 measured slip against the door-jet speed U_ref, which is up to
    3.4x the window speed when all windows are open -> slip was ~12x under-weighted there and did
    not improve at all from iter 1000 to 5000 (57% of the window speed). Taking the smaller of the
    two speeds never weakens the v19 weighting (single windows keep U_ref < U_win)."""
    areas = torch.tensor(WINDOW_AREAS, device=V.device, dtype=V.dtype).view(1, -1)
    q = (V * areas).sum(dim=1, keepdim=True)
    a_open = ((V > 0).to(V.dtype) * areas).sum(dim=1, keepdim=True)
    u_win = q / a_open.clamp_min(1e-9)
    u_ref = velocity_scale(V)
    # cap (independent review): never below U_ref / 3, i.e. at most 9x heavier than in v19 (all 8
    # windows open would otherwise be ~12x) -- keeps the wall term from swamping the NS residual
    return torch.maximum(torch.minimum(u_ref, u_win), u_ref / SLIP_MAX_RATIO).clamp_min(V_REL_FLOOR)


SLIP_MAX_RATIO = 3.0


def velocity_scale(V):
    """v19: U_ref(V) = max(sum_k V_k A_k / A_DOORS, U_NS_FLOOR), (B,1): the mean door jet speed of
    the scenario's steady through-flow -- the characteristic speed used to non-dimensionalise the
    momentum residual and the velocity boundary terms (every scenario then counts O(1))."""
    areas = torch.tensor(WINDOW_AREAS, device=V.device, dtype=V.dtype).view(1, -1)
    return ((V * areas).sum(dim=1, keepdim=True) / A_DOORS).clamp_min(U_NS_FLOOR)


def throughflow_scale(V, t=None):
    """Q_scale = max(sum_k V_k A_k, Q_FLOOR): the steady inflow of the open windows for this
    scenario, per point, shape (B, 1). Deliberately WITHOUT the tanh(3t/TAU_RAMP) ramp
    (independent review of v17): the network's velocity does not ramp with t, so a ramped
    scale made the weight explode for t -> 0 (loss spikes up to ~1e4). The true normal velocity
    is 0 at every wall point at every time, so a time-independent scale is equally correct.
    (t is accepted and ignored, so callers need not change.)"""
    areas = torch.tensor(WINDOW_AREAS, device=V.device, dtype=V.dtype).view(1, -1)
    return (V * areas).sum(dim=1, keepdim=True).clamp_min(Q_FLOOR)


def _planar_normal_component(x, y, z, u, v, w):
    """Velocity component normal to the planar face each wall point lies on (exact
    coordinate equality, as in _planar_wall_normal_derivative; sign irrelevant, squared)."""
    on_x = (x == ROOM_X[0]) | (x == ROOM_X[1])
    on_y = (y == ROOM_Y[0]) | (y == ROOM_Y[1])
    return torch.where(on_x, u, torch.where(on_y, v, w))


def walls_loss(model, device, flux_weight=1.0):
    """flux_weight: v17 warm-up factor for the flux-scaled leak term (0 -> 1 over BC_SCALING_WARMUP)."""
    # planar room faces (walls, floor, ceiling; door/window openings excluded)
    x, y, z, t, V, N_people = sample_walls(POINTS_WALLS, device)
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True)
    u, v, w, c, p = get_velocity_and_derivs(model, x, y, z, t, V, N_people)
    us2 = slip_scale(V) ** 2                                       # v20: slip relative to min(U_ref, U_win)
    loss = (u ** 2 / us2).mean() + (v ** 2 / us2).mean() + (w ** 2 / us2).mean()
    un = _planar_normal_component(x.detach(), y.detach(), z.detach(), u, v, w)
    un_scaled = [A_SOLID * un / throughflow_scale(V, t)]           # v17: leak as flux ratio

    # FIX (found by audit): the 4 columns are solid, floor-to-ceiling pillars --
    # no-slip must also hold on their curved surfaces, or nothing stops the
    # network from predicting flow straight through them.
    xc, yc, zc, tc, Vc, Nc = sample_columns_surface(POINTS_COLUMNS_PER, device)
    xc.requires_grad_(True); yc.requires_grad_(True); zc.requires_grad_(True)
    uc, vc, wc, cc, pc = get_velocity_and_derivs(model, xc, yc, zc, tc, Vc, Nc)
    usc2 = slip_scale(Vc) ** 2
    loss = loss + (uc ** 2 / usc2).mean() + (vc ** 2 / usc2).mean() + (wc ** 2 / usc2).mean()
    # v17: radial (normal) velocity on the column surfaces, same flux scaling
    centers = torch.tensor([[cx, cy] for cx, cy, _, _, _ in COLUMNS], device=device, dtype=xc.dtype)
    radii = torch.tensor([r for _, _, r, _, _ in COLUMNS], device=device, dtype=xc.dtype)
    xd_, yd_ = xc.detach(), yc.detach()
    k = ((xd_ - centers[:, 0]) ** 2 + (yd_ - centers[:, 1]) ** 2).argmin(dim=1)
    nx = ((xd_.squeeze(-1) - centers[k, 0]) / radii[k]).unsqueeze(-1)
    ny = ((yd_.squeeze(-1) - centers[k, 1]) / radii[k]).unsqueeze(-1)
    un_scaled.append(A_SOLID * (uc * nx + vc * ny) / throughflow_scale(Vc, tc))
    return loss + flux_weight * (torch.cat(un_scaled, dim=0) ** 2).mean()


def windows_loss(model, device, co2_weight, rel_weight=1.0):
    """rel_weight: v17 warm-up blend from the absolute (0) to the relative (1) inflow error."""
    x, y, z, t, V, N_people, window_idx = sample_windows(POINTS_WINDOWS_PER, device)
    N_people = _co2_occupancy(N_people)  # v13
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True)
    u, v, w, c, p = get_velocity_and_derivs(model, x, y, z, t, V, N_people)

    V_at_point = V.gather(1, window_idx)  # (B,1) -- this point's own window's speed
    target_v = -V_at_point * torch.tanh(3.0 * t / TAU_RAMP)  # inflow into the room (-y direction)

    # c=0 (clean air in) gets the same co2_weight as the interior residual
    # (fixed at 1.0 in v8 -- see USE_ADAPTIVE_CO2_WEIGHT).
    # v8_nondim: measured in units of C_REF (dimensionless), consistent with
    # the scaled interior residual.
    # v16 fix (audit): c = 0 only where THIS window is OPEN (clean inflow). A closed
    # window is a wall: its velocity target is already 0 (V_k = 0 -> no-slip) and its
    # CO2 condition is no-flux, enforced in co2_boundary_loss. Before, c = 0 was imposed
    # on closed windows too, turning them into CO2 sinks (FD: 0.6-1.4% plane error,
    # ~0.5% of the CO2 lost by 120 s). Averaged over ALL window points, so each open
    # point keeps the same weight as before.
    is_open = (V_at_point > 0).float()
    # v17: velocity error RELATIVE to the window's own speed (floor V_REL_FLOOR; closed
    # windows, target 0, are measured against the floor) -- see the v17_fluxbc comment above.
    if not USE_WINDOW_NORMAL_TARGET:
        # v19: flux exact by construction; only the tangential components (air enters normal to
        # the window), plus clean inflow c = 0 at open windows. v20: tangential terms on the same
        # slip scale as the walls (consistency, review).
        us2 = slip_scale(V) ** 2
        return ((u ** 2 / us2).mean() + (w ** 2 / us2).mean()
                + co2_weight * (is_open * (c / C_REF) ** 2).mean())
    scale2 = ((1.0 - rel_weight) + rel_weight * target_v.abs().clamp_min(V_REL_FLOOR)) ** 2
    return ((u ** 2 / scale2).mean() + ((v - target_v) ** 2 / scale2).mean() + (w ** 2 / scale2).mean()
            + co2_weight * (is_open * (c / C_REF) ** 2).mean())


def doors_loss(model, device):
    x, y, z, t, V, N_people = sample_doors(POINTS_DOORS, device)
    _, _, _, _, p = model(x, y, z, t, V, N_people)
    return (p ** 2).mean()


# v9_co2_bc: length scale used to make the CO2 normal-gradient dimensionless.
# CO2 varies over the source width (sigma = 2.5 m), so a gradient of order
# C_REF / SIGMA is the natural "O(1)" scale -- consistent with how S_REF and
# C_REF non-dimensionalize the interior residual and the Dirichlet terms.
CO2_GRAD_REF = C_REF / SIGMA


def _planar_wall_normal_derivative(x, y, z, dc_dx, dc_dy, dc_dz):
    """dc/dn on the 6 planar room faces. sample_walls() places every point
    EXACTLY on its face via torch.full_like(..., ROOM_*[k]), so the face (and
    hence the normal axis) is recovered by exact coordinate equality. Only
    the axis matters, not the sign, since the loss squares dc/dn. Returns
    (dc_dn, on_any_face) -- on_any_face is checked by the smoke test."""
    on_x = (x == ROOM_X[0]) | (x == ROOM_X[1])
    on_y = (y == ROOM_Y[0]) | (y == ROOM_Y[1])
    on_z = (z == ROOM_Z[0]) | (z == ROOM_Z[1])
    dc_dn = torch.where(on_x, dc_dx, torch.where(on_y, dc_dy, dc_dz))
    return dc_dn, (on_x | on_y | on_z)


def co2_boundary_loss(model, device, co2_weight):
    """v9_co2_bc: the CO2 boundary conditions that were MISSING in v1-v8.

    Found from v8's diagnostics: once non-dimensionalization let the network
    leave the trivial CO2 solution, its CO2 maximum sat pinned against the
    door wall (x~11.9, y~0.1) instead of the room-centre source. The CO2
    advection-diffusion equation needs a condition on EVERY boundary to be
    well-posed; until now only the windows (c=0 inflow) and t=0 (c=0) had
    one, so walls/floor/ceiling/columns/doors were free -- the network could
    let CO2 pile up against, or flow through, solid walls and still satisfy
    the interior PDE.

    Standard conditions (textbook advection-diffusion / CFD, not a PINN
    trick):
      - solid walls, floor, ceiling, columns: no CO2 flux through the wall.
        The velocity there is already 0 (no-slip), so the total flux reduces
        to the diffusive part: -D dc/dn = 0  ->  dc/dn = 0.
      - doors (outflow): zero normal gradient dc/dn = 0, the standard CFD
        outflow condition -- CO2 leaves with the air by advection, with no
        artificial diffusive flux imposed at the opening.
    Windows keep their existing Dirichlet c=0 (clean inflow) in windows_loss.

    Each term is dc/dn / CO2_GRAD_REF (dimensionless), squared and averaged
    over all boundary points, weighted by co2_weight (1.0 in v8/v9)."""
    # planar faces (door/window openings excluded by sample_walls)
    x, y, z, t, V, N_people = sample_walls(POINTS_WALLS, device)
    N_people = _co2_occupancy(N_people)  # v13
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True)
    _, _, _, c, _ = model(x, y, z, t, V, N_people)
    dn_walls, _ = _planar_wall_normal_derivative(x, y, z, grad(c, x), grad(c, y), grad(c, z))

    # column side surfaces: outward radial normal from each point's own column
    # axis (normals built from DETACHED coordinates -- they're geometry, not
    # something to differentiate through)
    xc, yc, zc, tc, Vc, Nc = sample_columns_surface(POINTS_COLUMNS_PER, device)
    xc.requires_grad_(True); yc.requires_grad_(True); zc.requires_grad_(True)
    _, _, _, cc, _ = model(xc, yc, zc, tc, Vc, _co2_occupancy(Nc))  # v13
    centers = torch.tensor([[cx, cy] for cx, cy, _, _, _ in COLUMNS], device=device, dtype=xc.dtype)
    radii = torch.tensor([r for _, _, r, _, _ in COLUMNS], device=device, dtype=xc.dtype)
    xd_, yd_ = xc.detach(), yc.detach()
    dist2_axis = (xd_ - centers[:, 0]) ** 2 + (yd_ - centers[:, 1]) ** 2   # (P, 4)
    k = dist2_axis.argmin(dim=1)                                           # nearest column axis
    nx = (xd_.squeeze(-1) - centers[k, 0]) / radii[k]
    ny = (yd_.squeeze(-1) - centers[k, 1]) / radii[k]
    dn_cols = grad(cc, xc) * nx.unsqueeze(-1) + grad(cc, yc) * ny.unsqueeze(-1)

    # doors, on the y = ROOM_Y[0] wall: normal is the y axis
    xd, yd, zd, td, Vd, Nd = sample_doors(POINTS_DOORS, device)
    yd.requires_grad_(True)
    _, _, _, cd, _ = model(xd, yd, zd, td, Vd, _co2_occupancy(Nd))  # v13
    dn_doors = grad(cd, yd)

    # v16 fix (audit): CLOSED windows are walls -> no-flux dc/dn = 0 there (normal = y).
    # Open windows keep c = 0 (windows_loss). Only the closed-window points are appended
    # (not zeros for the open ones), so the wall/column/door terms are not diluted.
    xw, yw, zw, tw, Vw, Nw, idxw = sample_windows(POINTS_WINDOWS_PER, device)
    yw.requires_grad_(True)
    _, _, _, cw, _ = model(xw, yw, zw, tw, Vw, _co2_occupancy(Nw))
    is_closed = (Vw.gather(1, idxw) == 0).squeeze(1)
    dn_closed_windows = grad(cw, yw)[is_closed]          # (k, 1), k ~ 40% of the window points

    dn = torch.cat([dn_walls, dn_cols, dn_doors, dn_closed_windows], dim=0) / CO2_GRAD_REF
    return co2_weight * (dn ** 2).mean()


def ic_loss(model, device, co2_weight):
    x, y, z, t, V, N_people = sample_ic(POINTS_IC, device)
    N_people = _co2_occupancy(N_people)  # v13
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True)
    u, v, w, c, p = get_velocity_and_derivs(model, x, y, z, t, V, N_people)
    # v8_nondim: CO2 IC term in units of C_REF, consistent with windows_loss.
    # v16 fix (audit): NO pressure term. Incompressible flow has no pressure initial
    # condition (the doors fix the gauge, p = 0). At t = 0 the windows start to
    # accelerate the air (dv/dt = -1.5 V_k), which needs grad(p) != 0 inside the room;
    # forcing p = 0 at t = 0 contradicted the momentum equation during start-up.
    # (The CO2 term is exactly 0 anyway since v10's hard IC; kept for completeness.)
    return (u ** 2).mean() + (v ** 2).mean() + (w ** 2).mean() + co2_weight * ((c / C_REF) ** 2).mean()


def trivial_co2_floor(device, n=200000):
    """v8_nondim: the CO2(scaled) loss a network would get by outputting a
    constant (trivial) CO2 field -- then dc/dt, grad(c) and lap(c) are all 0,
    so res_c = -S and co2_loss = mean((S/S_REF)^2) over our sampling
    distribution. Printed at startup so the log can be read directly:
    CO2(scaled) staying near this number means the network is still stuck on
    the trivial solution (what happened in every run v1-v6); CO2(scaled)
    dropping clearly BELOW it means it is actually learning CO2.
    Uses _generate_interior_batch directly (not sample_interior) so this
    one-off estimate never touches the persistent pool state."""
    x, y, z, t, V, N_people = _generate_interior_batch(n, device)
    N_people = _co2_occupancy(N_people)  # v13: same occupancy as the training loss
    dist2 = (x - SOURCE_X) ** 2 + (y - SOURCE_Y) ** 2 + (z - BREATHING_HEIGHT) ** 2
    S = N_people * EMISSION_PER_PERSON * torch.exp(-dist2 / (SIGMA ** 2))
    return ((S / S_REF) ** 2).mean().item()


def main():
    # v19: optional short dry run (--iters N --tag dry) into checkpoints/<VERSION>_<tag>/, to check
    # loss balance, door split and speed before committing the GPU to a full run.
    import argparse
    global VERSION, CKPT_DIR, MAX_ITERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=None, help="override MAX_ITERS (dry run)")
    ap.add_argument("--tag", default=None, help="suffix for VERSION / checkpoint folder (dry run)")
    args = ap.parse_args()
    if args.tag:
        VERSION = f"{VERSION}_{args.tag}"
        CKPT_DIR = os.path.join(HERE, "checkpoints", VERSION)
        os.makedirs(CKPT_DIR, exist_ok=True)
    if args.iters:
        MAX_ITERS = args.iters
    print(f"VERSION={VERSION}, MAX_ITERS={MAX_ITERS}, checkpoints -> {CKPT_DIR}")
    # Refuse to overwrite an existing result: if this VERSION's checkpoint folder
    # already holds checkpoints, the user forgot to bump VERSION.
    existing = [f for f in os.listdir(CKPT_DIR) if f.endswith(".pth")]
    if existing:
        raise SystemExit(f"{CKPT_DIR} already contains {len(existing)} checkpoint(s) -- set a NEW "
                         f"VERSION in train_gnot.py before training, so existing results are not overwritten.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # v21: single-scenario mode (must be set before ANY sampling, incl. trivial_co2_floor)
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

    # CO2 loss weight -- 1.0 and held there in v8 (USE_ADAPTIVE_CO2_WEIGHT is
    # False, see its comment). If re-enabled, it's rebalanced periodically
    # (after a warm-up) by gradnorm_weight_update(); tracked as running state
    # across iterations, so it lives here in main().
    co2_weight = 1.0

    # v11: optionally resume (model weights + Adam state) -- see RESUME_FROM.
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
        if ckpt.get("optimizer", "adam") != OPTIMIZER:   # pre-v15 checkpoints carry no tag = Adam
            raise SystemExit(f"{resume_path} was trained with {ckpt.get('optimizer', 'adam')!r}, but "
                             f"OPTIMIZER = {OPTIMIZER!r} -- optimizer states are not interchangeable")
        model.load_state_dict(ckpt["model_state"])        # in place: optimizer keeps the same tensors
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
    guide_w = float("nan")  # diagnostic only: Expert's Guide norm-balancing weight

    # Best-loss checkpointing (added in v7): a safety net that keeps whichever
    # checkpoint had the LOWEST equal-weight total loss seen so far, so a late
    # destabilization doesn't leave us with only a worse final checkpoint.
    best_total_val = float("inf")

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
    for it in range(start_iter, MAX_ITERS + 1):
        cur_lr = lr_at(it)                      # v11: set explicitly every iteration
        for group in optimizer.param_groups:    # (no scheduler object -> no scheduler state to resume)
            group["lr"] = cur_lr
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
        cur_nu = nu_at(it)                      # v21: viscosity curriculum
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

        # v17: blend in the flux-scaled leak term and the relative inflow error over the
        # first BC_SCALING_WARMUP iterations (see BC_SCALING_WARMUP comment)
        bc_lam = min(1.0, it / BC_SCALING_WARMUP)
        L_walls = walls_loss(model, device, flux_weight=bc_lam)
        L_walls.backward()

        L_windows = windows_loss(model, device, co2_weight,
                                 rel_weight=bc_lam if USE_RELATIVE_WINDOW_LOSS else 0.0)
        L_windows.backward()

        L_doors = doors_loss(model, device)
        L_doors.backward()

        L_ic = ic_loss(model, device, co2_weight)
        L_ic.backward()

        # v9_co2_bc: CO2 no-flux (walls/floor/ceiling/columns) + zero-gradient
        # outflow (doors) -- see co2_boundary_loss() docstring.
        L_co2bc = co2_boundary_loss(model, device, co2_weight)
        L_co2bc.backward()

        # Defense-in-depth: cap the combined gradient's norm before stepping,
        # regardless of root cause (see GRAD_CLIP_MAX_NORM comment above).
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_MAX_NORM)

        optimizer.step()

        # CO2 weight: fixed at 1.0 in v8 (USE_ADAPTIVE_CO2_WEIGHT=False). If
        # re-enabled, the old periodic gradient-norm rebalancing (Wang et al.
        # 2021, see module-level history comment) runs after the warm-up.
        if it % CO2_WEIGHT_UPDATE_EVERY == 0:
            # v8: always compute the Expert's Guide norm-balancing weight for
            # the log (cheap: reuses grads already computed this iteration).
            guide_w = guide_norm_ratio(grad_ns, grad_co2)
            if USE_ADAPTIVE_CO2_WEIGHT and it >= CO2_WEIGHT_WARMUP_ITERS:
                co2_weight = gradnorm_weight_update(grad_ns, grad_co2, co2_weight)

        total_val = (L_ns.item() + co2_weight * L_co2.item() + L_walls.item()
                     + L_windows.item() + L_doors.item() + L_ic.item() + L_co2bc.item())

        # Equal-weight total used to pick the "best" checkpoint. v8: with
        # every residual now O(1) after non-dimensionalization, equal weights
        # are the natural comparison scale (the old version multiplied L_co2 by
        # CO2_WEIGHT_MAX=200, which after v8's scaling would have made "best"
        # track almost nothing but CO2 -- found by audit). Note L_windows/L_ic
        # contain co2_weight internally; with co2_weight fixed at 1.0 in v8 this
        # is exactly the equal-weight sum. If adaptive weighting is re-enabled,
        # revisit this.
        unweighted_total = (L_ns.item() + L_co2.item() + L_walls.item()
                            + L_windows.item() + L_doors.item() + L_ic.item() + L_co2bc.item())

        if it % LOG_EVERY == 0:
            elapsed = time.time() - start
            speed = (it - start_iter + 1) / elapsed if elapsed > 0 else 0.0
            with torch.no_grad():   # v19: door split for three probe scenarios (W1 only / W8 only / all 3 m/s)
                a_probe = model.door_split(torch.full((3, 1), 60.0, device=device), _ALPHA_PROBE_V.to(device))
            print(f"[Iter {it:05d}/{MAX_ITERS}] Total={total_val:.5f} | "
                  f"NS={L_ns.item():.5f} CO2(scaled)={L_co2.item():.5f} CO2_weight={co2_weight:.2f} guide_w={guide_w:.3g} "
                  f"Walls={L_walls.item():.5f} Windows={L_windows.item():.5f} Doors={L_doors.item():.5f} "
                  f"IC={L_ic.item():.5f} CO2_BC={L_co2bc.item():.5f} LR={cur_lr:.2e} nu={cur_nu:g} "
                  f"alpha(W1/W8/all)={a_probe[0].item():.2f}/{a_probe[1].item():.2f}/{a_probe[2].item():.2f} | {speed:.2f} it/s")

        # v7_higher_co2_weight: save a new best-loss checkpoint any time
        # unweighted_total hits a new low, overwriting the previous best each
        # time (not versioned by iteration -- this is a running "best so far"
        # pointer, not part of the regular iter-numbered checkpoint history).
        # Gated to every LOG_EVERY iterations (not every single iteration) --
        # FIX (found by audit): checking/saving every iteration would trigger
        # torch.save (GPU->CPU copy + disk I/O) very often during the fast
        # early-loss-drop phase, a real throughput hit for a safety net that
        # doesn't need iteration-exact precision.
        # v21 (review): losses at different nu are not comparable -> the best-loss pointer restarts
        # at every curriculum stage, so _best.pth always comes from the latest (finally: nu = NU) stage
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
