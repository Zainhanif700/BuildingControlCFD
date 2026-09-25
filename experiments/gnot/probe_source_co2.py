"""
Directly probes the network's predicted CO2 value AT the exact known source
location (not whatever point happens to be the grid's argmax), across one or
more checkpoints -- to distinguish genuine convergence toward the real
source from a coincidental artifact elsewhere in the domain (e.g. a boundary/
Gibbs-ringing effect near a window edge) that happens to be the current
grid-max but isn't the actual physical CO2 source.

Usage:
    python3 probe_source_co2.py checkpoints/v5_closed_window_fix/gnot_v5_closed_window_fix_iter10000.pth checkpoints/v5_closed_window_fix/gnot_v5_closed_window_fix_iter12000.pth checkpoints/v5_closed_window_fix/gnot_v5_closed_window_fix_iter14000.pth
"""
import sys
import torch

from gnot_model import GNOTOperator, check_checkpoint_compat
from point_sampler import NUM_WINDOWS, BREATHING_HEIGHT, ROOM_X, ROOM_Y


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 probe_source_co2.py <checkpoint_path> [<checkpoint_path> ...]")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    source_x = (ROOM_X[0] + ROOM_X[1]) / 2
    source_y = (ROOM_Y[0] + ROOM_Y[1]) / 2

    print(f"Probing CO2 prediction AT the true source location ({source_x:.2f}, {source_y:.2f}, "
          f"{BREATHING_HEIGHT:.2f}), all windows closed, N_people=20, t=60s:")
    # v8_nondim: physical reference value. Closed windows -> no convection;
    # diffusion length sqrt(D*t) = sqrt(0.005*60) ~ 0.55 m << source width
    # sigma = 2.5 m, so near the source C grows almost linearly:
    # C(source, t) ~ N * E * t = 20 * 1.15e-4 * 60 = 0.138.
    from point_sampler import EMISSION_PER_PERSON
    print(f"(physically expected: roughly 20 * {EMISSION_PER_PERSON} * 60 = "
          f"{20 * EMISSION_PER_PERSON * 60:.3f}; every run before v8 gave ~0.00-0.02)\n")

    for ckpt_path in sys.argv[1:]:
        model = GNOTOperator().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        check_checkpoint_compat(ckpt, ckpt_path)  # v8_nondim: refuse pre-v8 checkpoints
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        x = torch.tensor([[source_x]], dtype=torch.float32, device=device)
        y = torch.tensor([[source_y]], dtype=torch.float32, device=device)
        z = torch.tensor([[BREATHING_HEIGHT]], dtype=torch.float32, device=device)
        t = torch.tensor([[60.0]], device=device)
        V = torch.zeros(1, NUM_WINDOWS, device=device)
        N_people = torch.tensor([[20.0]], device=device)

        with torch.no_grad():
            _, _, _, C, _ = model(x, y, z, t, V, N_people)

        print(f"{ckpt_path} (iter={ckpt.get('iter', '?')}): "
              f"C(source) = {C.item():.6f}")


if __name__ == "__main__":
    main()
