"""
Continues the v5 training for another 10,000 iterations.
"""
import os
import torch

from train_gnot import (
    GNOTOperator, HERE, LR, GRAD_CLIP_MAX_NORM,
    physics_loss, walls_loss, windows_loss, doors_loss, ic_loss,
    compute_param_grads, gradnorm_weight_update,
    CO2_WEIGHT_UPDATE_EVERY,
)

from gnot_model import check_checkpoint_compat

SOURCE_CKPT = os.path.join(HERE, "checkpoints", "v5_closed_window_fix",
                           "gnot_v5_closed_window_fix_partial10000.pth")
N_MORE_ITERS = 10000
LOG_EVERY = 200
CKPT_EVERY = 2000
VERSION = "v5_closed_window_fix"


def main():
    raise SystemExit(
        "resume_v5.py is a superseded pre-v8 experiment script and is disabled. "
        "Use train_gnot.py (v8_nondim) for new runs, or the frozen copy in "
        "milestones/v5_closed_window_fix/ to reproduce the old result."
    )


def _original_main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = GNOTOperator().to(device)
    ckpt = torch.load(SOURCE_CKPT, map_location=device)
    check_checkpoint_compat(ckpt, SOURCE_CKPT)
    model.load_state_dict(ckpt["model_state"])
    co2_weight = ckpt["co2_weight"]
    start_iter = ckpt["iter"]
    print(f"Resumed from {SOURCE_CKPT} (iter={start_iter}, co2_weight={co2_weight:.4f})")

    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)

    ckpt_dir = os.path.join(HERE, "checkpoints", VERSION)
    os.makedirs(ckpt_dir, exist_ok=True)

    for i in range(N_MORE_ITERS + 1):
        it = start_iter + i
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

        if it % CO2_WEIGHT_UPDATE_EVERY == 0:
            co2_weight = gradnorm_weight_update(grad_ns, grad_co2, co2_weight)

        if i % LOG_EVERY == 0:
            print(f"it={it:05d} NS={L_ns.item():.5f} CO2={L_co2.item():.6f} "
                  f"Windows={L_windows.item():.5f} co2_w={co2_weight:.2f}")

        if i % CKPT_EVERY == 0 and i > 0:
            interim_path = os.path.join(ckpt_dir, f"gnot_{VERSION}_iter{it}.pth")
            torch.save({
                "iter": it, "version": VERSION, "co2_weight": co2_weight,
                "model_state": model.state_dict(),
            }, interim_path)
            print(f"  -> saved intermediate checkpoint: {interim_path}")

    final_iter = start_iter + N_MORE_ITERS
    final_path = os.path.join(ckpt_dir, f"gnot_{VERSION}_iter{final_iter}.pth")
    torch.save({
        "iter": final_iter, "version": VERSION, "co2_weight": co2_weight,
        "model_state": model.state_dict(),
    }, final_path)
    print("Saved final:", final_path)


if __name__ == "__main__":
    main()
