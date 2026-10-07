"""
Checks of forecast.py / train_forecast.py on small SYNTHETIC datasets (no real data needed, ~1-2 min CPU):
 1. frame selection: laminar-style t_c (0..120 every 10 s, then every 30 s) and RANS-style t_c give the
    same 61 frames on the 30-s grid
 2. samples: history = 12 frames ending at t0, future = the next 6; scaling by N / N_ref is exact
 3. model: output shapes, finite NLL; persistence error of a constant field = 0; oracle l2 = 0
 4. a short training lowers the NLL clearly (the model can learn)
 5. train_forecast.py end to end (2 members, few iterations) on a synthetic split file
Usage (training env, from experiments/gnot_openfoam_rans):  python3 test_forecast.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

import common  # noqa: F401
import forecast as F

HERE = os.path.dirname(os.path.abspath(__file__))


def synth(path, V, laminar=False, N_ref=20.0, seed=0):
    xs, ys = np.meshgrid(np.arange(0.25, 15.5, 0.5), np.arange(0.25, 9.1, 0.5), indexing="ij")
    pl = np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, 1.15)]
    off = pl[:200].copy()
    off[:, 2] = 2.05
    P = np.r_[pl, off].astype(np.float32)
    plane = np.r_[np.ones(len(pl), bool), np.zeros(len(off), bool)]
    t = np.r_[np.arange(0, 121, 10), np.arange(150, 1801, 30)] if laminar else np.arange(0, 1801, 30)
    vs = float(np.mean(V))
    shape = 1 + 0.3 * np.sin(P[:, 0] / 2 + vs) * np.cos(P[:, 1] / 3)
    level = lambda tt: (N_ref / 20) * 800 * (1 - np.exp(-tt / (600 + 300 * vs)))
    C = np.stack([level(tt) * shape for tt in t]).astype(np.float32)
    np.savez_compressed(path, P=P, plane=plane, t_c=t.astype(np.float32), C=C, V=np.array(V, np.float32),
                        N_ref=np.float32(N_ref))
    return C, t


def main():
    torch.manual_seed(0)
    tmp = tempfile.mkdtemp()
    ok = True

    # 1. frame selection
    C_l, t_l = synth(os.path.join(tmp, "L.npz"), [0.5] * 8, laminar=True)
    C_r, t_r = synth(os.path.join(tmp, "R.npz"), [0.5] * 8)
    a, b = F.load_plane(os.path.join(tmp, "L.npz")), F.load_plane(os.path.join(tmp, "R.npz"))
    e1 = np.abs(a["frames"] - b["frames"]).max()
    c2 = F.load_plane(os.path.join(tmp, "R.npz"), z=1.9)          # nearest stored height: 2.05 m, 200 points
    good = (a["frames"].shape == (61, b["xy"].shape[0]) and e1 == 0 and abs(a["z"] - 1.15) < 1e-6
            and c2["frames"].shape == (61, 200) and abs(c2["z"] - 2.05) < 1e-6)
    print(f"1 frames: laminar {a['frames'].shape} RANS {b['frames'].shape}, max diff {e1:g}  -> {'OK' if good else 'FAIL'}")
    ok &= good

    # 2. samples
    d = F.PlaneData([os.path.join(tmp, "R.npz")], torch.device("cpu"))
    h, f = d.sample(0, 20, 20.0)
    h2, f2 = d.sample(0, 20, 60.0)
    fr = d.frames[0]
    good = (h.shape == (d.xyz.shape[0], 12) and f.shape == (d.xyz.shape[0], 6) and torch.equal(h[:, -1], fr[20])
            and torch.equal(f[:, 0], fr[21]) and torch.equal(h[:, 0], fr[9]) and torch.allclose(h2, 3 * h) and torch.allclose(f2, 3 * f)
            and d.t0s[0][0] == 11 and d.t0s[0][-1] == 54)
    print(f"2 samples: shapes {tuple(h.shape)} {tuple(f.shape)}, t0 {d.t0s[0][0]}..{d.t0s[0][-1]} ({len(d.t0s[0])} per case), "
          f"N-scaling exact  -> {'OK' if good else 'FAIL'}")
    ok &= good

    # 3. model
    m = F.ForecastGNOT(d=64, layers=2, c_scale=500.0)
    gen = torch.Generator().manual_seed(0)
    qx, qh, hx, hh, V, N, tg = d.batch(3, 128, gen)
    mean, lv = m(qx, qh, hx, hh, V, N)
    L = F.nll(mean, lv, tg, 500.0)
    const = torch.ones(10, 6)
    good = (mean.shape == tg.shape == lv.shape == (3, 128, 6) and torch.isfinite(L) and F.l2_rel(tg, tg) == 0
            and F.l2_rel(const, const) == 0 and hh.shape == (3, min(F.N_HIST_TOKENS, d.xyz.shape[0]), 12))
    print(f"3 model: mean {tuple(mean.shape)}, NLL {float(L):.3f}, oracle l2 0  -> {'OK' if good else 'FAIL'}")
    ok &= good

    # 4. short training
    paths = []
    for i, v in enumerate([0.2, 0.6, 1.0, 1.5]):
        p = os.path.join(tmp, f"S{i:02d}.npz")
        synth(p, [v] * 8)
        paths.append(p)
    d4 = F.PlaneData(paths, torch.device("cpu"))
    m = F.ForecastGNOT(d=64, layers=2, c_scale=500.0, d_scale=100.0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(1)
    hist = []
    for it in range(300):
        qx, qh, hx, hh, V, N, tg = d4.batch(4, 256, gen)
        mean, lv = m(qx, qh, hx, hh, V, N)
        loss = F.nll(mean, lv, tg, m.d_scale)
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(float(loss))
    m.eval()
    e, base, n = F.evaluate([m], d4, every=8)
    eo, _, _ = F.evaluate([m], d4, every=8, offset=400.0)
    good = np.mean(hist[-30:]) < np.mean(hist[:30]) - 0.5 and np.isfinite(e) and 0 < eo < e
    print(f"4 training: NLL {np.mean(hist[:30]):+.3f} -> {np.mean(hist[-30:]):+.3f}; l2 {100 * e:.2f}% "
          f"(persistence {100 * base:.2f}%, {n} samples); with +400 offset {100 * eo:.2f}%  ->{'OK' if good else 'FAIL'}")
    ok &= good

    # 5. end to end
    scen = os.path.join(tmp, "scen.txt")
    with open(scen, "w") as fh:
        fh.write("# name split V\n")
        for i in range(3):
            fh.write(f"S{i:02d} train {','.join(['1'] * 8)}\n")
        fh.write(f"S03 test {','.join(['1'] * 8)}\n")
    tag = "_test_forecast_tmp"
    out = os.path.join(HERE, "checkpoints", tag)
    if os.path.isdir(out):
        for f_ in os.listdir(out):
            os.remove(os.path.join(out, f_))
    r = subprocess.run([sys.executable, os.path.join(HERE, "train_forecast.py"), "--data-dir", tmp, "--scenarios", scen,
                        "--tag", tag, "--members", "2", "--iters", "60", "--batch", "2", "--queries", "128", "--d", "32",
                        "--layers", "1", "--log-every", "30", "--eval-every", "60", "--device", "cpu"],
                       capture_output=True, text=True)
    good = r.returncode == 0 and "FINAL" in r.stdout and os.path.isfile(os.path.join(out, "member1.pth"))
    print(f"5 train_forecast.py end to end  -> {'OK' if good else 'FAIL'}")
    if not good:
        print(r.stdout[-2000:], r.stderr[-3000:])
    else:
        print("   " + "\n   ".join(l for l in r.stdout.splitlines() if "ensemble" in l or "FINAL" in l))
    for f_ in os.listdir(out):
        os.remove(os.path.join(out, f_))
    os.rmdir(out)
    ok &= good

    # 6. transition data (make_transitions.py format): sample = (N_A/N_ref) H + (N_B/N_ref) S, end to end
    tdir = os.path.join(tmp, "tr")
    os.makedirs(tdir)
    for i, v in enumerate([0.2, 0.6, 1.0, 1.5]):
        d = np.load(os.path.join(tmp, f"S{i:02d}.npz"))
        P = d["P"].copy()
        P[200:, 2] = 1.55                      # two planes: 1.15 and 1.55 m
        Hs = (d["C"][-1][None, :] * np.exp(-d["t_c"][:, None] / (300 + 200 * v))).astype(np.float32)
        np.savez_compressed(os.path.join(tdir, f"S{i:02d}.npz"), P=P, V=d["V"], N_ref=d["N_ref"], t_c=d["t_c"],
                            H=Hs, S=d["C"], from_case=np.array("Sxx"))
    dt_ = F.PlaneData([os.path.join(tdir, "S01.npz")], torch.device("cpu"), z=1.6, transitions=True)
    h1, f1 = dt_.sample(0, 20, 30.0, 50.0)
    exp_f = dt_.frames[0] * 1.5 + dt_.H[0] * 2.5
    good = (dt_.xyz.shape[0] == len(P) - 200 and abs(float(dt_.xyz[0, 2]) - 1.55) < 1e-6
            and torch.allclose(h1[:, -1], exp_f[20]) and torch.allclose(f1[:, 0], exp_f[21]))
    out = os.path.join(HERE, "checkpoints", tag)
    r = subprocess.run([sys.executable, os.path.join(HERE, "train_forecast.py"), "--data-dir", tdir, "--scenarios", scen,
                        "--transitions", "--z", "1.6", "--c-offset", "400",
                        "--tag", tag, "--members", "1", "--iters", "60", "--batch", "2", "--queries", "128", "--d", "32",
                        "--layers", "1", "--log-every", "30", "--eval-every", "60", "--device", "cpu"],
                       capture_output=True, text=True)
    good = good and r.returncode == 0 and "FINAL" in r.stdout
    print(f"6 transition data: N_A/N_B superposition exact, train_forecast --transitions end to end  -> {'OK' if good else 'FAIL'}")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
    if os.path.isdir(out):
        for f_ in os.listdir(out):
            os.remove(os.path.join(out, f_))
        os.rmdir(out)
    ok &= good

    print("\nALL OK" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
