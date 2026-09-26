"""
DIAGNOSTIC D2: is the remaining error an OPTIMIZER limit?

Fine-tunes an existing checkpoint with L-BFGS (a quasi-Newton, second-order
method) on ONE FIXED batch of training points, and tracks three things:
  (a) the loss on that fixed batch,
  (b) the loss on a different, held-out batch (fresh points),
  (c) the real error against the finite-difference reference
      (closed windows, 20 people, breathing height, t = 60 s).
Why: PINN losses are ill-conditioned; Adam can stall on them while a
second-order method keeps descending (Rathore et al. 2024, ICML,
arXiv:2402.01868: Adam+L-BFGS lowered errors several-fold vs Adam alone).
L-BFGS needs a deterministic loss, hence the fixed batch.

How to read the result (decides v14):
  (a) AND (b) drop a lot (>= ~3x) and (c) falls -> OPTIMIZER-limited:
      a better optimizer (SOAP, Wang et al. 2025, arXiv:2502.00604) is the fix.
  only (a) drops, (b) and (c) don't -> the network just fits these particular
      points: a SAMPLING / generalization limit, not the optimizer.
  nothing drops -> a representation / formulation floor: stop tuning.

The batch is fixed by re-seeding PyTorch's RNG to the same value before every
evaluation, so the unchanged training samplers and loss functions of
train_gnot.py produce exactly the same points each time. The persistent point
pool must be OFF (it is: point_sampler.USE_PERSISTENT_POOL = False).

Usage:
    python3 lbfgs_probe.py <checkpoint> [--steps 150]
Each outer step runs up to 20 L-BFGS iterations (~20-25 s); 150 steps ~ 1 h.
Saves checkpoints/<version>_lbfgs_probe/ (best and final), not a new version.
"""
import argparse
import os
import time

import numpy as np
import torch

from fd_reference_closed_room import solve, interp, breathing_grid, pinn_on, SX, SY
from gnot_model import GNOTOperator, check_checkpoint_compat, NONDIM_CHECKPOINT_KEY, MODEL_FORMAT_KEY, MODEL_FORMAT
from point_sampler import BREATHING_HEIGHT, USE_PERSISTENT_POOL
from train_gnot import (physics_loss, walls_loss, windows_loss, doors_loss, ic_loss,
                        co2_boundary_loss, HERE)

FIXED_SEED = 1234        # the training batch L-BFGS optimizes on
HELDOUT_SEEDS = [7, 99]  # fresh batches, never optimized on
N_PEOPLE, T_EVAL = 20.0, 60.0


def total_loss(model, device, seed, backward):
    """Exactly the v13 training loss (co2_weight = 1), on the batch defined by `seed`.
    Each term is backward()-ed right away (like train_gnot.main) to keep memory low."""
    torch.manual_seed(seed)
    parts = {}
    L_ns, L_co2 = physics_loss(model, device)
    if backward:
        (L_ns + L_co2).backward()
    parts["ns"], parts["co2"] = L_ns.item(), L_co2.item()
    for name, fn in (("walls", lambda: walls_loss(model, device)),
                     ("windows", lambda: windows_loss(model, device, 1.0)),
                     ("doors", lambda: doors_loss(model, device)),
                     ("ic", lambda: ic_loss(model, device, 1.0)),
                     ("co2_bc", lambda: co2_boundary_loss(model, device, 1.0))):
        L = fn()
        if backward:
            L.backward()
        parts[name] = L.item()
    return sum(parts.values()), parts


def heldout(model, device):
    vals = [total_loss(model, device, s, backward=False) for s in HELDOUT_SEEDS]
    return float(np.mean([v[0] for v in vals])), float(np.mean([v[1]["co2"] for v in vals]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--eval-every", type=int, default=10)
    args = ap.parse_args()
    if USE_PERSISTENT_POOL:
        raise SystemExit("point_sampler.USE_PERSISTENT_POOL must be False for a fixed, seed-defined batch")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    check_checkpoint_compat(ckpt, args.checkpoint)
    model = GNOTOperator().to(device)
    model.load_state_dict(ckpt["model_state"])
    version = ckpt.get("version", "unknown")
    out_dir = os.path.join(HERE, "checkpoints", f"{version}_lbfgs_probe")
    os.makedirs(out_dir, exist_ok=True)

    # finite-difference reference for the real-error check (N=1 solve, scaled)
    xs, ys, xg, yg, inside = breathing_grid()
    m = ~inside
    out1, grid, _ = solve(0.1, "noflux", n_people=1.0)
    ref = N_PEOPLE * interp(out1[T_EVAL], grid, xg, yg, np.full_like(xg, BREATHING_HEIGHT)).astype(float)
    ref_src = N_PEOPLE * float(interp(out1[T_EVAL], grid, SX, SY, BREATHING_HEIGHT))

    def real_error():
        p = pinn_on(model, device, xg, yg, T_EVAL, z=BREATHING_HEIGHT, n_people=N_PEOPLE).astype(float)
        ps = float(pinn_on(model, device, [SX], [SY], T_EVAL, z=BREATHING_HEIGHT, n_people=N_PEOPLE)[0])
        return np.linalg.norm(p[m] - ref[m]) / np.linalg.norm(ref[m]), (ps - ref_src) / ref_src

    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, history_size=100, max_iter=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        val, _ = total_loss(model, device, FIXED_SEED, backward=True)
        return torch.tensor(val, device=device)

    def report(step, t0):
        fixed, parts = total_loss(model, device, FIXED_SEED, backward=False)
        ho, ho_co2 = heldout(model, device)
        l2, src = real_error()
        print(f"step {step:4d} | fixed-batch loss {fixed:.5f} (CO2 {parts['co2']:.5f}) | "
              f"held-out loss {ho:.5f} (CO2 {ho_co2:.5f}) | plane L2 {l2 * 100:5.1f}% | "
              f"source {src * 100:+5.1f}% | {time.time() - t0:6.0f} s", flush=True)
        return fixed, ho, l2

    def save(tag, step):
        torch.save({"iter": ckpt.get("iter", 0), "lbfgs_steps": step, "version": f"{version}_lbfgs_probe",
                    "co2_weight": 1.0, NONDIM_CHECKPOINT_KEY: True, MODEL_FORMAT_KEY: MODEL_FORMAT,
                    "model_state": model.state_dict()}, os.path.join(out_dir, f"{tag}.pth"))

    t0 = time.time()
    print(f"L-BFGS probe on {args.checkpoint} (version={version}); fixed seed {FIXED_SEED}, "
          f"held-out seeds {HELDOUT_SEEDS}; real error = closed windows, N=20, z=1.10 m, t=60 s")
    f0, h0, e0 = report(0, t0)
    best_fixed = f0
    for step in range(1, args.steps + 1):
        loss = opt.step(closure)
        if not torch.isfinite(loss):
            print(f"step {step}: L-BFGS produced a non-finite loss -- stopping (see lbfgs_probe docstring)")
            break
        if loss.item() < best_fixed:
            best_fixed = loss.item()
            save("best_fixed", step)
        if step % args.eval_every == 0 or step == args.steps:
            f, h, e = report(step, t0)
    f, h, e = report(step, t0)
    save("final", step)
    print(f"\nSUMMARY after {step} L-BFGS steps:")
    print(f"  fixed-batch loss : {f0:.5f} -> {f:.5f}  ({f0 / max(f, 1e-30):.2f}x lower)")
    print(f"  held-out loss    : {h0:.5f} -> {h:.5f}  ({h0 / max(h, 1e-30):.2f}x lower)")
    print(f"  plane L2 error   : {e0 * 100:.1f}% -> {e * 100:.1f}%")
    print("  Reading: both losses >= ~3x lower and the plane error down -> optimizer-limited (v14 = SOAP);")
    print("           only the fixed-batch loss down -> sampling/generalization limit;")
    print("           nothing down -> formulation/representation floor.")
    print(f"Checkpoints (best_fixed.pth, final.pth) in {out_dir}")


if __name__ == "__main__":
    main()
