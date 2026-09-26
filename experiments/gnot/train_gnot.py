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
    EMISSION_PER_PERSON, S_REF, C_REF, TAU_RAMP, COLUMNS,
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
MAX_ITERS = 20000  # v10's validated setup (v11 used 30000 via resume -- see history)
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


def lr_at(it):
    """LR for iteration `it`: LR until LR_DECAY_START, then cosine from LR down
    to exactly LR_MIN at MAX_ITERS. LR_DECAY_START=None -> constant LR."""
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
#
# IMPORTANT: this VERSION variable (and CKPT_DIR below) is what train_gnot.py's own
# main() uses for a FULL 20k-iteration production run. Bump this to match whichever
# fix combination is confirmed working via the closed-window diagnostic BEFORE
# launching the next full run through this file, so production checkpoints aren't
# mislabeled with stale physics/sampling.
VERSION = "v12_linear_n"
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
    # v8_nondim: divide by S_REF so the CO2 residual is O(1) instead of
    # O(6e-3) -- every term in res_c (dc/dt, conv_c, D*lap(c), S) has units
    # of concentration/second, so this is a pure rescaling of the same
    # equation, not a change to the physics. With this, the trivial C=0
    # solution scores ~0.093 (was 3.08e-6), comparable to NS -- so the
    # network finally has a real incentive to leave it.
    co2_loss = ((res_c / S_REF) ** 2).mean()
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

    # c=0 (clean air in) gets the same co2_weight as the interior residual
    # (fixed at 1.0 in v8 -- see USE_ADAPTIVE_CO2_WEIGHT).
    # v8_nondim: measured in units of C_REF (dimensionless), consistent with
    # the scaled interior residual.
    return (u ** 2).mean() + ((v - target_v) ** 2).mean() + (w ** 2).mean() + co2_weight * ((c / C_REF) ** 2).mean()


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
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True)
    _, _, _, c, _ = model(x, y, z, t, V, N_people)
    dn_walls, _ = _planar_wall_normal_derivative(x, y, z, grad(c, x), grad(c, y), grad(c, z))

    # column side surfaces: outward radial normal from each point's own column
    # axis (normals built from DETACHED coordinates -- they're geometry, not
    # something to differentiate through)
    xc, yc, zc, tc, Vc, Nc = sample_columns_surface(POINTS_COLUMNS_PER, device)
    xc.requires_grad_(True); yc.requires_grad_(True); zc.requires_grad_(True)
    _, _, _, cc, _ = model(xc, yc, zc, tc, Vc, Nc)
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
    _, _, _, cd, _ = model(xd, yd, zd, td, Vd, Nd)
    dn_doors = grad(cd, yd)

    dn = torch.cat([dn_walls, dn_cols, dn_doors], dim=0) / CO2_GRAD_REF
    return co2_weight * (dn ** 2).mean()


def ic_loss(model, device, co2_weight):
    x, y, z, t, V, N_people = sample_ic(POINTS_IC, device)
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True)
    u, v, w, c, p = get_velocity_and_derivs(model, x, y, z, t, V, N_people)
    # v8_nondim: CO2 IC term in units of C_REF, consistent with windows_loss.
    return (u ** 2).mean() + (v ** 2).mean() + (w ** 2).mean() + (p ** 2).mean() + co2_weight * ((c / C_REF) ** 2).mean()


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
    dist2 = (x - SOURCE_X) ** 2 + (y - SOURCE_Y) ** 2 + (z - BREATHING_HEIGHT) ** 2
    S = N_people * EMISSION_PER_PERSON * torch.exp(-dist2 / (SIGMA ** 2))
    return ((S / S_REF) ** 2).mean().item()


def main():
    # Refuse to overwrite an existing result: if this VERSION's checkpoint folder
    # already holds checkpoints, the user forgot to bump VERSION.
    existing = [f for f in os.listdir(CKPT_DIR) if f.endswith(".pth")]
    if existing:
        raise SystemExit(f"{CKPT_DIR} already contains {len(existing)} checkpoint(s) -- set a NEW "
                         f"VERSION in train_gnot.py before training, so existing results are not overwritten.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = GNOTOperator().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GNOT parameters: {n_params:,}")
    floor = trivial_co2_floor(device)
    print(f"[v8_nondim] Trivial-solution CO2(scaled) reference = {floor:.4f}  "
          f"(CO2(scaled) near this = still stuck on C=const; clearly below = learning CO2)")
    print(f"[v8_nondim] adaptive CO2 weighting: {'ON' if USE_ADAPTIVE_CO2_WEIGHT else 'OFF (co2_weight fixed at 1.0)'}; "
          f"guide_w column = norm-balancing weight the Expert's Guide rule would pick (diagnostic)")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    params = list(model.parameters())

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
        model.load_state_dict(ckpt["model_state"])        # in place: optimizer keeps the same tensors
        optimizer.load_state_dict(ckpt["optimizer_state"])
        co2_weight = ckpt.get("co2_weight", 1.0)
        start_iter = ckpt["iter"] + 1
        print(f"[v11] resumed from {resume_path} (iter={ckpt['iter']}, version={ckpt.get('version')}); "
              f"continuing at iter {start_iter}")
    if LR_DECAY_START is None:
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
            print(f"[Iter {it:05d}/{MAX_ITERS}] Total={total_val:.5f} | "
                  f"NS={L_ns.item():.5f} CO2(scaled)={L_co2.item():.5f} CO2_weight={co2_weight:.2f} guide_w={guide_w:.3g} "
                  f"Walls={L_walls.item():.5f} Windows={L_windows.item():.5f} Doors={L_doors.item():.5f} "
                  f"IC={L_ic.item():.5f} CO2_BC={L_co2bc.item():.5f} LR={cur_lr:.2e} | {speed:.2f} it/s")

        # v7_higher_co2_weight: save a new best-loss checkpoint any time
        # unweighted_total hits a new low, overwriting the previous best each
        # time (not versioned by iteration -- this is a running "best so far"
        # pointer, not part of the regular iter-numbered checkpoint history).
        # Gated to every LOG_EVERY iterations (not every single iteration) --
        # FIX (found by audit): checking/saving every iteration would trigger
        # torch.save (GPU->CPU copy + disk I/O) very often during the fast
        # early-loss-drop phase, a real throughput hit for a safety net that
        # doesn't need iteration-exact precision.
        if it % LOG_EVERY == 0 and unweighted_total < best_total_val:
            best_total_val = unweighted_total
            best_path = os.path.join(CKPT_DIR, f"gnot_{VERSION}_best.pth")
            torch.save({"iter": it, "version": VERSION, "co2_weight": co2_weight,
                        "unweighted_total": best_total_val, NONDIM_CHECKPOINT_KEY: True, MODEL_FORMAT_KEY: MODEL_FORMAT, "lr": cur_lr,
                        "model_state": model.state_dict()}, best_path)

        if it % CKPT_EVERY == 0 and it > 0:
            ckpt_path = os.path.join(CKPT_DIR, f"gnot_{VERSION}_iter{it}.pth")
            torch.save({"iter": it, "version": VERSION, "co2_weight": co2_weight,
                        NONDIM_CHECKPOINT_KEY: True, MODEL_FORMAT_KEY: MODEL_FORMAT, "lr": cur_lr,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict()}, ckpt_path)
            print(f"  -> saved checkpoint: {ckpt_path}")

    final_path = os.path.join(CKPT_DIR, f"gnot_{VERSION}_final.pth")
    torch.save({"iter": MAX_ITERS, "version": VERSION, "co2_weight": co2_weight,
                NONDIM_CHECKPOINT_KEY: True, MODEL_FORMAT_KEY: MODEL_FORMAT, "lr": cur_lr,
                "model_state": model.state_dict()}, final_path)
    print(f"Training complete. Final checkpoint: {final_path}")
    print(f"Best checkpoint (lowest unweighted_total={best_total_val:.5f}): "
          f"{os.path.join(CKPT_DIR, f'gnot_{VERSION}_best.pth')}")


if __name__ == "__main__":
    main()
