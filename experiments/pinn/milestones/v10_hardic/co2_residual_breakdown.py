# ============================================================================
# MILESTONE SNAPSHOT: v10_hardic (2026-09-26) -- frozen copy, DO NOT EDIT.
# Model/training code from git commit 6529b15 (exactly what the v10 run trained
# with); fd_reference_closed_room.py from a0b2992. See README.md. Run scripts
# from INSIDE this folder so they import this frozen model code.
# ============================================================================

"""
CO2 residual BREAKDOWN (v8 onward): HOW is the network satisfying the CO2
equation, and in WHICH scenarios?

    residual = dc/dt + u.grad(c) - D lap(c) - S

v8 puzzle this answers: the overall CO2(scaled) training loss is ~8x below
the trivial floor, yet probe_co2_time.py shows that with closed windows the
predicted CO2 barely grows over time (~3% of the physical rate). So the
equation is being satisfied SOMEWHERE -- this shows where, and by which term:
  - closed windows: the physics says dc/dt must balance S (no airflow);
  - open windows:   convection u.grad(c) can balance S with tiny gradients.

Evaluated on the SAME spatial point distribution as training
(_generate_interior_batch: 40% uniform + 60% concentrated near the source),
with the same random t and N_people, but the window setting overridden per
scenario. Uses train_gnot.py's own grad/get_velocity_and_derivs and
constants, so the formula is identical to the training loss.

Small batches (default 200 points per scenario) so it can run on the same GPU
while a training run is using most of the memory.

Usage:
    python3 co2_residual_breakdown.py <checkpoint> [n_points]
"""
import sys
import torch

from gnot_model import GNOTOperator, check_checkpoint_compat
from point_sampler import _generate_interior_batch, NUM_WINDOWS, V_MAX, S_REF
from train_gnot import (grad, get_velocity_and_derivs, DIFFUSIVITY, EMISSION_PER_PERSON,
                        SIGMA, SOURCE_X, SOURCE_Y, BREATHING_HEIGHT)


def breakdown(model, device, n, scenario):
    x, y, z, t, V, N_people = _generate_interior_batch(n, device)
    if scenario == "closed":
        V = torch.zeros_like(V)
    elif scenario == "open":
        V = torch.full_like(V, V_MAX)
    # "training mix": keep sample_scenario's own V (30% closed, 20% partial, 50% uniform)
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True); t.requires_grad_(True)

    u, v, w, c, p = get_velocity_and_derivs(model, x, y, z, t, V, N_people)
    dc_dx, dc_dy, dc_dz, dc_dt = grad(c, x), grad(c, y), grad(c, z), grad(c, t)
    lap_c = grad(dc_dx, x) + grad(dc_dy, y) + grad(dc_dz, z)
    conv = u * dc_dx + v * dc_dy + w * dc_dz
    dist2 = (x - SOURCE_X) ** 2 + (y - SOURCE_Y) ** 2 + (z - BREATHING_HEIGHT) ** 2
    S = N_people * EMISSION_PER_PERSON * torch.exp(-dist2 / (SIGMA ** 2))
    diff = DIFFUSIVITY * lap_c
    res = dc_dt + conv - diff - S

    def rms(q):  # RMS in units of S_REF, same scaling as the training loss
        return (q.detach() / S_REF).pow(2).mean().sqrt().item()

    speed = torch.sqrt(u ** 2 + v ** 2 + w ** 2).detach()
    return {
        "loss": (res.detach() / S_REF).pow(2).mean().item(),
        "floor": (S.detach() / S_REF).pow(2).mean().item(),
        "dc/dt": rms(dc_dt), "u.grad(c)": rms(conv), "D lap(c)": rms(diff), "S": rms(S),
        "speed_mean": speed.mean().item(),
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=device)
    check_checkpoint_compat(ckpt, path)
    model = GNOTOperator().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    torch.manual_seed(0)

    print(f"{path} (iter={ckpt.get('iter', '?')}), {n} points per scenario, "
          f"all values RMS in units of S_REF={S_REF:.2e}")
    print(f"{'scenario':14s} {'loss':>7s} {'floor':>7s} {'loss/floor':>10s} | "
          f"{'dc/dt':>7s} {'u.grad(c)':>9s} {'D lap(c)':>8s} {'S':>7s} | {'mean speed':>10s}")
    for scen in ["closed", "open", "training mix"]:
        r = breakdown(model, device, n, scen)
        print(f"{scen:14s} {r['loss']:7.4f} {r['floor']:7.4f} {r['loss'] / r['floor']:10.2f} | "
              f"{r['dc/dt']:7.4f} {r['u.grad(c)']:9.4f} {r['D lap(c)']:8.4f} {r['S']:7.4f} | "
              f"{r['speed_mean']:10.4f}")
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\nHOW TO READ THIS:\n"
          "  loss/floor ~1 -> that scenario is still on the trivial solution (nothing learned).\n"
          "  loss/floor << 1 -> the equation is satisfied there; the term columns show HOW:\n"
          "    closed windows should balance S mainly with dc/dt (CO2 accumulating);\n"
          "    if instead u.grad(c) carries it with closed windows, the network is faking the\n"
          "    source with the leftover spurious velocity.")


if __name__ == "__main__":
    main()
