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
    ROOM_X, ROOM_Y, ROOM_Z, NUM_WINDOWS,
)

# --- physical constants (matching Alexander's config exactly) ---
NU = 0.01
RHO = 1.0
DIFFUSIVITY = 0.005
EMISSION_PER_PERSON = 1.15e-4
SIGMA = 2.5
BREATHING_HEIGHT = 1.10
TAU_RAMP = 2.0
SOURCE_X = (ROOM_X[0] + ROOM_X[1]) / 2
SOURCE_Y = (ROOM_Y[0] + ROOM_Y[1]) / 2

# --- training config ---
# NOTE: these are much smaller than Alexander's own point counts (8000
# interior, etc.) on purpose. His trainer uses a plain MLP; ours uses
# cross-attention, and computing the Laplacian terms needs SECOND-order
# autograd through that attention -- measured to need roughly 4x more GPU
# memory per point than a plain MLP would. Confirmed on the RTX A2000 (12GB):
# 300 interior points (+ proportionally smaller boundary batches) peaked at
# 1.1GB. These defaults scale that up ~5x for a healthy safety margin -- see
# the smoke test in the chat history before changing these further.
POINTS_INTERIOR = 1500
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
# ("loss imbalance" / "spectral bias" -- see literature). Fix: track CO2
# as its own separate, upweighted loss term instead of silently summed in.
CO2_LOSS_WEIGHT = 100.0

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")
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
    """Returns (ns_loss, co2_loss) SEPARATELY -- see CO2_LOSS_WEIGHT note above
    for why these are no longer combined into one number."""
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


def windows_loss(model, device):
    x, y, z, t, V, N_people, window_idx = sample_windows(POINTS_WINDOWS_PER, device)
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True)
    u, v, w, c, p = get_velocity_and_derivs(model, x, y, z, t, V, N_people)

    V_at_point = V.gather(1, window_idx)  # (B,1) -- this point's own window's speed
    target_v = -V_at_point * torch.tanh(3.0 * t / TAU_RAMP)  # inflow into the room (-y direction)

    # c=0 (clean air in) gets the same CO2_LOSS_WEIGHT treatment as the interior
    # residual, for the same reason -- otherwise it's numerically tiny next to
    # the velocity terms and gets neglected during training.
    return (u ** 2).mean() + ((v - target_v) ** 2).mean() + (w ** 2).mean() + CO2_LOSS_WEIGHT * (c ** 2).mean()


def doors_loss(model, device):
    x, y, z, t, V, N_people = sample_doors(POINTS_DOORS, device)
    _, _, _, _, p = model(x, y, z, t, V, N_people)
    return (p ** 2).mean()


def ic_loss(model, device):
    x, y, z, t, V, N_people = sample_ic(POINTS_IC, device)
    x.requires_grad_(True); y.requires_grad_(True); z.requires_grad_(True)
    u, v, w, c, p = get_velocity_and_derivs(model, x, y, z, t, V, N_people)
    return (u ** 2).mean() + (v ** 2).mean() + (w ** 2).mean() + (p ** 2).mean() + CO2_LOSS_WEIGHT * (c ** 2).mean()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = GNOTOperator().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GNOT parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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

        L_ns, L_co2 = physics_loss(model, device)
        L_ns.backward()
        (CO2_LOSS_WEIGHT * L_co2).backward()

        L_walls = walls_loss(model, device)
        L_walls.backward()

        L_windows = windows_loss(model, device)
        L_windows.backward()

        L_doors = doors_loss(model, device)
        L_doors.backward()

        L_ic = ic_loss(model, device)
        L_ic.backward()

        optimizer.step()

        total_val = (L_ns.item() + CO2_LOSS_WEIGHT * L_co2.item() + L_walls.item()
                     + L_windows.item() + L_doors.item() + L_ic.item())

        if it % LOG_EVERY == 0:
            elapsed = time.time() - start
            speed = (it + 1) / elapsed if elapsed > 0 else 0.0
            print(f"[Iter {it:05d}/{MAX_ITERS}] Total={total_val:.5f} | "
                  f"NS={L_ns.item():.5f} CO2(raw)={L_co2.item():.6f} CO2(weighted)={CO2_LOSS_WEIGHT * L_co2.item():.5f} "
                  f"Walls={L_walls.item():.5f} Windows={L_windows.item():.5f} Doors={L_doors.item():.5f} "
                  f"IC={L_ic.item():.5f} | {speed:.2f} it/s")

        if it % CKPT_EVERY == 0 and it > 0:
            ckpt_path = os.path.join(CKPT_DIR, f"gnot_iter{it}.pth")
            torch.save({"iter": it, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict()}, ckpt_path)
            print(f"  -> saved checkpoint: {ckpt_path}")

    final_path = os.path.join(CKPT_DIR, "gnot_final.pth")
    torch.save({"iter": MAX_ITERS, "model_state": model.state_dict()}, final_path)
    print(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
