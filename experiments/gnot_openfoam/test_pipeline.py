"""
Pipeline test for Track 2 (run before any real training; ~1-2 min on GPU, a few min on CPU):
  1. SupervisedGNOT: u = 0 exactly with all windows closed, C(t=0) = 0, C exactly linear in N,
     outputs finite; gradients reach the backbone.
  2. Metrics: rel() / metrics() give 0 for a perfect prediction and the right value for a known error.
  3. Training loop on a SYNTHETIC dataset (known smooth fields, saved like extract_case.py does):
     train_supervised.main() runs end to end, writes checkpoints, and the loss falls clearly.
Usage (from experiments/gnot_openfoam):  python3 test_pipeline.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

import common
from model import SupervisedGNOT
import train_supervised as ts

FAILED = []


def check(name, cond, msg=""):
    print(f"{'PASS' if cond else 'FAIL'}: {name} {msg}")
    if not cond:
        FAILED.append(name)


def synthetic_dataset(path, n=3000, seed=0):
    from point_sampler import ROOM_X, ROOM_Y, ROOM_Z
    rng = np.random.default_rng(seed)
    P = np.stack([rng.uniform(*ROOM_X, n), rng.uniform(*ROOM_Y, n), rng.uniform(*ROOM_Z, n)], 1).astype(np.float32)
    t = np.array(common.T_DATA, np.float32)
    ramp = np.tanh(3 * t / 2.0)[:, None]
    U = np.stack([0.2 * np.sin(P[:, 1] / 3.0)[None] * ramp, -0.3 * np.cos(P[:, 0] / 5.0)[None] * ramp,
                  0.05 * np.sin(P[:, 2])[None] * ramp], -1).astype(np.float32)
    C = (1e-3 * (t[:, None] / 120.0) * np.exp(-((P[:, 0] - 7.8) ** 2 + (P[:, 1] - 4.6) ** 2) / 6.0)[None]).astype(np.float32)
    plane = np.abs(P[:, 2] - 1.1) < 0.3
    np.savez_compressed(path, P=P, t=t, U=U, C=C, V=np.array([1.0] + [0.0] * 7, np.float32),
                        N_ref=np.float32(common.N_REF), nu=np.float32(0.01), dx=np.float32(0.1),
                        plane=plane, z_plane=np.float32(1.1))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    m = SupervisedGNOT().to(dev)
    n = 64
    x = torch.rand(n, 1, device=dev) * 15 + 0.2
    y = torch.rand(n, 1, device=dev) * 9 + 0.1
    z = torch.rand(n, 1, device=dev) * 3
    t = torch.rand(n, 1, device=dev) * 120
    Vo = torch.tensor([[1.0, 0, 2.0, 0, 0, 0, 0, 3.0]], device=dev).expand(n, -1)
    Vc = torch.zeros(n, 8, device=dev)
    N10, N20 = torch.full((n, 1), 10.0, device=dev), torch.full((n, 1), 20.0, device=dev)
    with torch.no_grad():
        uc = m(x, y, z, t, Vc, N20)
        uo = m(x, y, z, t, Vo, N20)
        u0 = m(x, y, z, torch.zeros_like(t), Vo, N20)
        u10 = m(x, y, z, t, Vo, N10)
    check("1a u = 0 with all windows closed", max(a.abs().max().item() for a in uc[:3]) == 0.0)
    check("1b u != 0 with windows open", max(a.abs().max().item() for a in uo[:3]) > 0)
    check("1c C(t=0) = 0", u0[3].abs().max().item() == 0.0)
    lin = ((uo[3] - 2 * u10[3]).abs().max() / uo[3].abs().max().clamp_min(1e-30)).item()
    check("1d C linear in N", lin < 1e-5, f"(rel. deviation {lin:.1e})")
    check("1e velocity independent of N", max((a - b).abs().max().item() for a, b in zip(uo[:3], u10[:3])) <= 1e-6)
    check("1f finite", all(torch.isfinite(a).all().item() for a in uo))
    loss = sum(a.pow(2).mean() for a in m(x, y, z, t, Vo, N20))
    loss.backward()
    g = [p.grad for p in m.core.blocks.parameters() if p.grad is not None]
    check("1g gradients reach the attention blocks", len(g) > 0 and all(torch.isfinite(q).all().item() for q in g))

    a = torch.tensor([1.0, 2.0, 2.0])
    check("2a rel(a, a) = 0", ts.rel(a, a) == 0.0)
    check("2b rel known value", abs(ts.rel(a * 1.1, a) - 0.1) < 1e-6)

    tmp = tempfile.mkdtemp()
    try:
        data = os.path.join(tmp, "synth.npz")
        synthetic_dataset(data)
        ds = ts.load(data, dev)

        class Oracle(torch.nn.Module):   # returns the dataset values -> metrics must be 0
            def forward(self, x, y, z, t, V, N):
                ti = int(np.argmin(np.abs(ds["t"] - float(t[0]))))
                i0 = oracle_offset[0]
                oracle_offset[0] += x.shape[0]
                Ur, Cr = ds["U_d"][ti, i0:i0 + x.shape[0]], ds["C_d"][ti, i0:i0 + x.shape[0]]
                return Ur[:, 0:1], Ur[:, 1:2], Ur[:, 2:3], Cr.unsqueeze(1)
        oracle_offset = [0]
        mm = ts.metrics(Oracle(), ds, len(ds["t"]) - 1, dev)
        check("2c metrics of a perfect prediction are 0", all(abs(v) < 1e-7 for k, v in mm.items() if k != "mass")
              and abs(mm["mass"] - 1) < 1e-6, str(mm))

        # 3. end-to-end training on the synthetic data (small, fast)
        common.CKPT_DIR = os.path.join(tmp, "ckpt")
        ts.CKPT_DIR = common.CKPT_DIR
        import io
        import contextlib
        buf = io.StringIO()
        argv = sys.argv
        sys.argv = ["train_supervised.py", "--data", data, "--tag", "t", "--iters", "400", "--batch", "512",
                    "--eval-every", "400", "--device", dev]
        try:
            with contextlib.redirect_stdout(buf):
                ts.main()
        finally:
            sys.argv = argv
        log = buf.getvalue()
        losses = [float(l.split("loss ")[1].split()[0]) for l in log.splitlines() if l.startswith("[")]
        check("3a training runs end to end", "done" in log and len(losses) >= 2)
        check("3b loss falls clearly", len(losses) >= 2 and losses[-1] < 0.3 * losses[0],
              f"({losses[0]:.3e} -> {losses[-1]:.3e})")
        check("3c checkpoint written", os.path.isfile(os.path.join(common.CKPT_DIR, "t", "sup_t_iter400.pth")))
        check("3d held-out times reported", "held-out time" in log)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nRESULT:", "ALL PASSED" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
