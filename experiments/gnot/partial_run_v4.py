"""
Partial (10,000-iteration) training run for v4_source_sampling -- combines
fix #1 (isotropic Fourier features + source_proximity, see gnot_model.py)
and fix #2 (source-concentrated interior sampling, see point_sampler.py).

Kept as a committed script (not a pasted inline python3 -c command) because
repeated heredoc/inline pastes over this SSH+tmux setup were getting
corrupted mid-paste. Run directly:

    tmux new -s partial_v4
    cd experiments/gnot
    python3 partial_run_v4.py

Same 10,000-iteration budget as the fix-#1-only partial run, so the two are
a fair apples-to-apples comparison.
"""
import os
import torch

from train_gnot import (
    GNOTOperator, HERE, LR, GRAD_CLIP_MAX_NORM,
    physics_loss, walls_loss, windows_loss, doors_loss, ic_loss,
    compute_param_grads, gradnorm_weight_update,
    CO2_WEIGHT_WARMUP_ITERS, CO2_WEIGHT_UPDATE_EVERY,
)

N_ITERS = 10000
LOG_EVERY = 200
CKPT_EVERY = 2000  # FIX (found by audit): this script previously only saved a
# checkpoint once, at the very end -- if the process crashed or the server/
# tmux session died before finishing, ALL progress would be lost with nothing
# recoverable. train_gnot.py's own main() already does periodic checkpointing
# (CKPT_EVERY=1000); this script didn't, purely because it started as a quick
# one-off test script. Fixed to match that safer pattern.
VERSION = "v4b_source_sampling_tuned"  # distinct from v4_source_sampling (frac=0.4,
# std=sigma) so the earlier result isn't overwritten -- lets us compare the
# two directly rather than losing the "before" data point.


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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

        if it >= CO2_WEIGHT_WARMUP_ITERS and it % CO2_WEIGHT_UPDATE_EVERY == 0:
            co2_weight = gradnorm_weight_update(grad_ns, grad_co2, co2_weight)

        if it % LOG_EVERY == 0:
            print(f"it={it:05d} NS={L_ns.item():.5f} CO2={L_co2.item():.6f} "
                  f"Windows={L_windows.item():.5f} co2_w={co2_weight:.2f}")

        if it % CKPT_EVERY == 0 and it > 0:
            interim_path = os.path.join(ckpt_dir, f"gnot_{VERSION}_iter{it}.pth")
            torch.save({
                "iter": it, "version": VERSION, "co2_weight": co2_weight,
                "model_state": model.state_dict(),
            }, interim_path)
            print(f"  -> saved intermediate checkpoint: {interim_path}")

    ckpt_path = os.path.join(ckpt_dir, f"gnot_{VERSION}_partial{N_ITERS}.pth")
    torch.save({
        "iter": N_ITERS, "version": VERSION, "co2_weight": co2_weight,
        "model_state": model.state_dict(),
    }, ckpt_path)
    print("Saved:", ckpt_path)


if __name__ == "__main__":
    main()
