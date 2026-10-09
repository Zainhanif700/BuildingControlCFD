"""
Cross-check: does the model reproduce the window inflow? Uses the training sampler and the absolute velocity error.
"""
import sys

import torch

from gnot_model import GNOTOperator, check_checkpoint_compat
from point_sampler import sample_windows, TAU_RAMP, NUM_WINDOWS
from train_gnot import get_velocity_and_derivs

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "milestones/v13_fullocc/gnot_v13_fullocc_final.pth"
dev = "cpu"
ckpt = torch.load(ckpt_path, map_location=dev)
check_checkpoint_compat(ckpt, ckpt_path)
model = GNOTOperator().to(dev)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"{ckpt_path} (version={ckpt.get('version')}, iter={ckpt.get('iter')})")

torch.manual_seed(0)
x, y, z, t, V, N, idx = sample_windows(200)


def window_error(V, t):
    xx, yy, zz = (a.clone().requires_grad_(True) for a in (x, y, z))
    u, v, w, _, _ = get_velocity_and_derivs(model, xx, yy, zz, t, V, N)
    Vk = V.gather(1, idx)
    target = -Vk * torch.tanh(3.0 * t / TAU_RAMP)
    e2 = (u ** 2 + (v - target) ** 2 + w ** 2).detach()
    return e2, Vk.detach(), target.detach(), v.detach()


e2, Vk, target, v = window_error(V, t)
print(f"\n1) training mix, training times: velocity part of windows_loss = {e2.mean().item():.4f}  "
      f"(log 'Windows' ~0.03-0.05 incl. CO2 term)")
e2, Vk, target, v = window_error(V, torch.full_like(t, 60.0))
closed = (V.abs().sum(1) == 0)
opn = Vk.squeeze(1) > 0.5
print(f"   points with all windows closed: {closed.float().mean().item() * 100:.0f}%  -> error there {e2[closed].mean().item():.2e} (exact 0 expected)")
rel = (e2.squeeze(1)[opn].sqrt() / target.abs().squeeze(1)[opn].clamp_min(1e-6))
print(f"   training-mix V at t=60s, own window open (V_k>0.5): rms error / V_k ={rel.pow(2).mean().sqrt().item() * 100:.1f}%, "
      f"median {rel.median().item() * 100:.1f}%;  mean v/target = {(v[opn.unsqueeze(1)] / target[opn.unsqueeze(1)]).mean().item():.2f}")

for name, Vfix in (("all 3 m/s", [3.0] * 8), ("all 1 m/s", [1.0] * 8),
                   ("W1 3 m/s only", [3.0] + [0.0] * 7), ("W8 3 m/s only", [0.0] * 7 + [3.0])):
    Vt = torch.tensor(Vfix).view(1, NUM_WINDOWS).expand(len(x), -1)
    t60 = torch.full_like(t, 60.0)
    e2, Vk, target, v = window_error(Vt, t60)
    m = Vk.squeeze(1) > 0
    r = e2.squeeze(1)[m].sqrt() / target.abs().squeeze(1)[m]
    print(f"2) {name:14s} t=60s: rms error / V_k = {r.pow(2).mean().sqrt().item() * 100:5.1f}%, "
          f"mean v/target = {(v.squeeze(1)[m] / target.squeeze(1)[m]).mean().item():.2f}")

Vr = torch.rand(len(x), NUM_WINDOWS) * 5.0
e2, Vk, target, v = window_error(Vr, torch.full_like(t, 60.0))
m = Vk.squeeze(1) > 0.5
r = e2.squeeze(1)[m].sqrt() / target.abs().squeeze(1)[m]
print(f"3) random V per point, t=60s: rms error / V_k = {r.pow(2).mean().sqrt().item() * 100:5.1f}%, "
      f"mean v/target = {(v.squeeze(1)[m] / target.squeeze(1)[m]).mean().item():.2f}")
