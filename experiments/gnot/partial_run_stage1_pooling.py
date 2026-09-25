"""
Short (5,000-iteration) VALIDATION run for Stage 1 of the self-adaptive
weighting + sampling upgrade (Chen, Howard & Stinis, arXiv:2511.05452, 2025)
-- see point_sampler.py's module comment above sample_interior() for the
full context.

WHY THIS SCRIPT EXISTS: Stage 1 (persistent point pool, refreshed 20% every
100 calls, instead of 100% fresh every call) is a real architectural change
to how training points are sampled, and it needs to be validated in
ISOLATION before Stage 2 (per-point adaptive weighting) is built on top of
it -- so that if something regresses, we know it's the resampling change,
not the weighting math.

IMPORTANT HONEST CAVEAT (added after independent review of this plan): this
script can only tell us "Stage 1 alone didn't break anything" -- it CANNOT
tell us whether the eventual Stage 1 + Stage 2 combination will actually fix
CO2 localization. The paper this is based on (Chen, Howard & Stinis 2025,
arXiv:2511.05452) reports in its own ablations that adaptive SAMPLING ALONE
(without adaptive weighting) gives negligible improvement on 3 of their 4
benchmark problems, including their Navier-Stokes case -- the closest match
to this project's physics. So a clean result here is a regression/plumbing
check, not evidence the overall approach works. That real test only comes
once Stage 2 is built.

What this DOES check:
  - No NaN/divergence over 10,000 iterations (same physics/weighting as the
    live train_gnot.py at the time it is run -- since v8 that means the
    non-dimensionalized model/losses).
  - Closed-window velocity behavior should look similar to the numbers
    already established for v5/v7 at comparable iteration counts (fix #3
    is untouched by Stage 1 -- this checks the persistent pool didn't
    silently break it).
  - NEW (found by independent review): whether persistent pooling introduces
    its OWN low-frequency oscillation in scenario mixture -- since points now
    persist for up to ~100 iterations instead of being re-randomized every
    single iteration, a skewed random draw (e.g. too many closed-window
    points) could linger instead of averaging out immediately, potentially
    confounding the very CO2 oscillation this project is trying to diagnose.
    See the periodic pool-composition logging below.

Run:
    tmux new -s stage1_pooling
    cd experiments/gnot
    python3 partial_run_stage1_pooling.py
"""
import os
import torch

from train_gnot import (
    GNOTOperator, HERE, LR, GRAD_CLIP_MAX_NORM, POINTS_INTERIOR,
    physics_loss, walls_loss, windows_loss, doors_loss, ic_loss,
    compute_param_grads, gradnorm_weight_update,
    CO2_WEIGHT_WARMUP_ITERS, CO2_WEIGHT_UPDATE_EVERY, CO2_WEIGHT_MAX,
    USE_ADAPTIVE_CO2_WEIGHT,
)
from point_sampler import (interior_pool_composition, CLOSED_SCENARIO_FRAC, SOURCE_SAMPLE_FRAC,
                           USE_PERSISTENT_POOL)
from gnot_model import NONDIM_CHECKPOINT_KEY, MODEL_FORMAT_KEY, MODEL_FORMAT

N_ITERS = 10000  # RAISED from 5000 (found by independent review): the cited
# paper's own hyperparameter study noted resampling-related effects can
# appear "especially at the final stage of training," and this project's own
# CO2 oscillation bug is itself a late-training phenomenon -- 5,000
# iterations (50 refresh cycles) was too short a window to say anything
# about late-stage pooling behavior. 10,000 iterations (100 refresh cycles)
# is still half the real 20k-iteration run length, but covers meaningfully
# more of where problems have actually shown up before.
LOG_EVERY = 100
CKPT_EVERY = 1000
# FIX (found by independent review of the Stage 1 plan, not just the code):
# persistent pooling means a skewed random draw of scenario mixture could now
# persist for a long stretch (up to POOL_REFRESH_EVERY-1 iterations) instead
# of being re-randomized away every single iteration like before -- a NEW
# potential oscillation source that could confound the very CO2 oscillation
# this project is trying to diagnose. Log the pool's actual composition
# periodically so this is directly OBSERVED, not just hoped against.
POOL_COMPOSITION_LOG_EVERY = 250
VERSION = "stage1_pooling_validation"  # NOT a real production version tag --
# this checkpoint is for health-check diagnostics only, not for comparison
# against v5/v6/v7's own checkpoint history.


def main():
    # v8_nondim: this script validates the persistent pool, which v8 switches
    # OFF (point_sampler.USE_PERSISTENT_POOL=False) to keep v8 a clean test of
    # non-dimensionalization. Running it with the pool off would "validate"
    # nothing, so refuse instead of producing a misleading result.
    if not USE_PERSISTENT_POOL:
        raise SystemExit("partial_run_stage1_pooling.py: point_sampler.USE_PERSISTENT_POOL is False "
                         "(v8_nondim), so there is no pool to validate. Set it True first "
                         "(planned for Stage 2, after v8's results are in).")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Validating Stage 1 (persistent-pool sampling) health over {N_ITERS} iterations. "
          f"Same physics/weighting as live train_gnot.py (adaptive CO2 weight "
          f"{'ON' if USE_ADAPTIVE_CO2_WEIGHT else 'OFF, fixed 1.0'}, constant LR={LR}).")

    model = GNOTOperator().to(device)
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)
    co2_weight = 1.0

    ckpt_dir = os.path.join(HERE, "checkpoints", VERSION)
    os.makedirs(ckpt_dir, exist_ok=True)

    for it in range(N_ITERS + 1):
        optimizer.zero_grad()
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

        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_MAX_NORM)
        optimizer.step()

        # Follow train_gnot.py's switch (found by audit: this used to rebalance
        # unconditionally, re-introducing the v8 over-weighting problem).
        if USE_ADAPTIVE_CO2_WEIGHT and it >= CO2_WEIGHT_WARMUP_ITERS and it % CO2_WEIGHT_UPDATE_EVERY == 0:
            co2_weight = gradnorm_weight_update(grad_ns, grad_co2, co2_weight)

        if it % LOG_EVERY == 0:
            print(f"it={it:05d} NS={L_ns.item():.5f} CO2={L_co2.item():.6f} "
                  f"Windows={L_windows.item():.5f} co2_w={co2_weight:.2f}")

        if it % POOL_COMPOSITION_LOG_EVERY == 0:
            comp = interior_pool_composition(POINTS_INTERIOR, device)
            if comp is not None:
                print(f"  [pool composition @ it={it:05d}, pool.calls={comp['calls']}] "
                      f"frac_all_closed={comp['frac_all_closed']:.3f} (target~{CLOSED_SCENARIO_FRAC}) "
                      f"frac_near_source={comp['frac_near_source']:.3f} (target~{SOURCE_SAMPLE_FRAC}-ish) "
                      f"mean_dist_to_source={comp['mean_dist_to_source']:.3f}")

        if it % CKPT_EVERY == 0 and it > 0:
            interim_path = os.path.join(ckpt_dir, f"gnot_{VERSION}_iter{it}.pth")
            torch.save({"iter": it, "version": VERSION, "co2_weight": co2_weight, NONDIM_CHECKPOINT_KEY: True, MODEL_FORMAT_KEY: MODEL_FORMAT,
                        "model_state": model.state_dict()}, interim_path)
            print(f"  -> saved checkpoint: {interim_path}")

    final_path = os.path.join(ckpt_dir, f"gnot_{VERSION}_final.pth")
    torch.save({"iter": N_ITERS, "version": VERSION, "co2_weight": co2_weight, NONDIM_CHECKPOINT_KEY: True, MODEL_FORMAT_KEY: MODEL_FORMAT,
                "model_state": model.state_dict()}, final_path)
    print("Saved:", final_path)


if __name__ == "__main__":
    main()
