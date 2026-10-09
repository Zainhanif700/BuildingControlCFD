"""
Short validation run (5,000 iterations) for the adaptive weighting and sampling test.
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

N_ITERS = 10000
LOG_EVERY = 100
CKPT_EVERY = 1000
POOL_COMPOSITION_LOG_EVERY = 250
VERSION = "stage1_pooling_validation"


def main():
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
