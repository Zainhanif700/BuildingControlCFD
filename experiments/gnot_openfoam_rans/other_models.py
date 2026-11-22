"""
Other models for the CO2 forecast, on the SAME data, split, samples and error as the GNOT model (train_forecast.py):
  linear : one linear layer per point (the simplest learned baseline)
  mlp    : the same point-wise, with hidden layers (no information from the neighbours)
  fno    : Fourier neural operator (Li et al. 2021) on the 1.6 m plane as an image; points that are not in the plane
           (columns etc.) are masked: they are zero in the input, a mask channel tells where they are, and they do
           not count in the loss or in the error
All three get the same information as GNOT: the 12 CO2 maps (last value and change to it), the 8 window speeds and
whether they are open, the 2 doors, the number of people, the time since the last window change (tau), and the
position. The scalar inputs are given as constant channels. The output is the change of the CO2 for the next 6 maps.
Ensemble of 5 networks (mean), as GNOT. Loss: mean squared error of the change (GNOT: Gaussian NLL, which also
predicts a variance; here no variance is needed for the comparison of the mean).

Rule fixed BEFORE the run: the result is reported whatever it is, with the same two error definitions as the thesis
(paper metric with +400 ppm first, strict second).

Usage (lab PC, env cfd), same data and settings as the final GNOT model v4_tr_tau09_long:
  python3 other_models.py --model fno --data-dir data_transitions_v4 --transitions --scenarios scenarios_v4.txt --tag fno1 --iters 40000
  python3 other_models.py --info --data-dir data_transitions_v4 --transitions --scenarios scenarios_v4.txt    (only the grid check)
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

import common
import forecast as F
from forecast import PlaneData, l2_rel, tau_of, H_IN, T_OUT, N_MAX_FC
from point_sampler import V_MAX
from train_forecast import split, c_scale_of, d_scale_of

HERE = os.path.dirname(os.path.abspath(__file__))
N_CONST = 8 + 8 + 2 + 1 + 1           # speeds, open flags, doors, people, tau
C_IN = H_IN + 2 + 1 + N_CONST         # maps, position, mask, constants


class Grid:
    """The plane points as an image (rows = y, columns = x). Needs points on a regular grid."""
    def __init__(self, xy, dev):
        xs, ys = np.unique(np.round(xy[:, 0], 5)), np.unique(np.round(xy[:, 1], 5))
        dx = float(min(np.diff(xs).min(), np.diff(ys).min()))
        ix, iy = np.rint((xy[:, 0] - xs[0]) / dx), np.rint((xy[:, 1] - ys[0]) / dx)
        err = max(np.abs(xs[0] + ix * dx - xy[:, 0]).max(), np.abs(ys[0] + iy * dx - xy[:, 1]).max())
        assert err < 1e-3, f"the plane points are not on a regular grid (max deviation {err:.4f} m): FNO needs one"
        self.nx, self.ny, self.dx = int(ix.max()) + 1, int(iy.max()) + 1, dx
        lin = (iy * self.nx + ix).astype(np.int64)
        assert len(np.unique(lin)) == len(lin), "two plane points in the same grid cell"
        self.n_points = len(lin)
        self.lin = torch.tensor(lin, device=dev)
        m = torch.zeros(self.ny * self.nx, device=dev)
        m[self.lin] = 1.0
        self.mask = m.view(1, self.ny, self.nx)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, self.ny, device=dev), torch.linspace(0, 1, self.nx, device=dev),
                                indexing="ij")
        self.pos = torch.stack([xx, yy], 0)

    def to_grid(self, x):                      # (Np, C) -> (C, H, W)
        out = x.new_zeros(x.shape[1], self.ny * self.nx)
        out[:, self.lin] = x.T
        return out.view(x.shape[1], self.ny, self.nx)

    def from_grid(self, g):                    # (C, H, W) -> (Np, C)
        return g.reshape(g.shape[0], -1)[:, self.lin].T

    def info(self):
        n = self.ny * self.nx
        return (f"grid {self.ny} x {self.nx} cells of {100 * self.dx:.1f} cm, {self.n_points} plane points, "
                f"{n - self.n_points} empty cells ({100 * (n - self.n_points) / n:.1f} % of the image)")


def build_input(grid, hist, V, N, tau, c_scale, d_scale):
    """hist (Np, 12), V (10,) windows + doors, N people (float), tau (float) -> (C_IN, H, W)."""
    last = hist[:, -1:]
    pt = torch.cat([last / c_scale, (hist[:, :-1] - last) / d_scale], 1)
    Vw = V[:8]
    g = torch.cat([Vw / V_MAX, (Vw != 0).float(), V[8:10], torch.tensor([N / N_MAX_FC, tau], device=V.device)])
    gc = g.view(-1, 1, 1).expand(-1, grid.ny, grid.nx)
    return torch.cat([grid.to_grid(pt), grid.pos, grid.mask, gc], 0)


class PointNet(nn.Module):
    """linear (hidden = 0) or MLP with 1x1 convolutions: every point on its own."""
    def __init__(self, hidden=256, layers=3):
        super().__init__()
        dims = [C_IN] + [hidden] * layers if layers > 0 else [C_IN]
        mods = []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Conv2d(a, b, 1), nn.GELU()]
        mods.append(nn.Conv2d(dims[-1], T_OUT, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class SpectralConv2d(nn.Module):
    def __init__(self, cin, cout, m1, m2):
        super().__init__()
        self.cout, self.m1, self.m2 = cout, m1, m2
        s = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(s * torch.rand(cin, cout, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(s * torch.rand(cin, cout, m1, m2, dtype=torch.cfloat))

    def forward(self, x):
        B, _, H, W = x.shape
        xf = torch.fft.rfft2(x)
        out = torch.zeros(B, self.cout, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :self.m1, :self.m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :self.m1, :self.m2], self.w1)
        out[:, :, -self.m1:, :self.m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -self.m1:, :self.m2], self.w2)
        return torch.fft.irfft2(out, s=(H, W))


class FNO2d(nn.Module):
    def __init__(self, ny, nx, width=32, modes=16, layers=4, pad=8):
        super().__init__()
        self.pad = pad
        m1 = min(modes, (ny + 2 * pad) // 2)
        m2 = min(modes, (nx + 2 * pad) // 2 + 1)
        self.lift = nn.Conv2d(C_IN, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, m1, m2) for _ in range(layers)])
        self.skip = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, T_OUT, 1))

    def forward(self, x):
        x = self.lift(x)
        p = self.pad
        if p:
            x = Fn.pad(x, (p, p, p, p))
        for i, (s, w) in enumerate(zip(self.spec, self.skip)):
            x = s(x) + w(x)
            if i < len(self.spec) - 1:
                x = Fn.gelu(x)
        if p:
            x = x[:, :, p:-p, p:-p]
        return self.proj(x)


def make_model(args, grid):
    if args.model == "linear":
        return PointNet(layers=0)
    if args.model == "mlp":
        return PointNet(hidden=args.width, layers=3)
    return FNO2d(grid.ny, grid.nx, width=args.width, modes=args.modes)


def draw(data, gen, pad_prob, n_lo=10.0, n_hi=80.0):
    """One training sample, drawn as in PlaneData.batch: (k, hist, fut, N, tau)."""
    k = int(torch.randint(len(data.cases), (1,), generator=gen))
    t0 = data.t0s[k][int(torch.randint(len(data.t0s[k]), (1,), generator=gen))]
    N = float(n_lo + (n_hi - n_lo) * torch.rand(1, generator=gen))
    NA = float(n_lo + (n_hi - n_lo) * torch.rand(1, generator=gen)) if data.H is not None else 0.0
    if pad_prob > 0 and data.H is not None and float(torch.rand(1, generator=gen)) < pad_prob:
        t0 = int(torch.randint(H_IN - 1, (1,), generator=gen))
        hist, fut = data.sample_padded(k, t0, N, NA)
    else:
        hist, fut = data.sample(k, t0, N, NA)
    return k, hist, fut, N, tau_of(t0, data.H is not None)


@torch.no_grad()
def predict(models, grid, data, k, hist, N, tau, cs, ds):
    x = build_input(grid, hist, data.V[k], N, tau, cs, ds).unsqueeze(0)
    out = torch.stack([m(x)[0] for m in models]).mean(0)
    return hist[:, -1:] + ds * grid.from_grid(out)


def evaluate(models, grid, data, cs, ds, N=None, every=1, offset=0.0):
    """Mean l2 error of the ensemble and of persistence; the samples and people numbers are those of forecast.evaluate."""
    errs, base = [], []
    rng = np.random.default_rng(1234)
    for k in range(len(data.cases)):
        for t0 in data.t0s[k][::every]:
            if data.H is not None:
                nb, na = rng.uniform(10.0, 80.0, 2)
                n = float(nb) if N is None else N
                na = float(na)
            else:
                n, na = (data.N_ref[k] if N is None else N), 0.0
            hist, fut = data.sample(k, t0, n, na)
            if torch.linalg.norm(fut) == 0:
                continue
            pred = predict(models, grid, data, k, hist, n, tau_of(t0, data.H is not None), cs, ds)
            errs.append(l2_rel(pred + offset, fut + offset))
            base.append(l2_rel(hist[:, -1:].expand_as(fut) + offset, fut + offset))
    return float(np.mean(errs)), float(np.mean(base)), len(errs)


def history_change(models, grid, te, cs, ds, offset, draws=3, t0_max=16):
    """As eval_history_change.py: window change inside the 6-min history (t0 0..10) and normal samples (11..16)."""
    rng = np.random.default_rng(1234)
    rows = {}
    for k in range(len(te.cases)):
        for t0 in range(0, t0_max + 1):
            for _ in range(draws):
                n_now, n_before = (float(v) for v in rng.uniform(10.0, 80.0, 2))
                hist, fut = te.sample_padded(k, t0, n_now, n_before)
                if torch.linalg.norm(fut) == 0:
                    continue
                pred = predict(models, grid, te, k, hist, n_now, tau_of(t0, True), cs, ds)
                base = hist[:, -1:].expand_as(fut)
                r = rows.setdefault(t0, {"m": [], "p": [], "mo": [], "po": []})
                r["m"].append(l2_rel(pred, fut)); r["p"].append(l2_rel(base, fut))
                r["mo"].append(l2_rel(pred + offset, fut + offset)); r["po"].append(l2_rel(base + offset, fut + offset))
    avg = lambda keys, key: 100 * np.mean([v for t in keys for v in rows[t][key]])
    chg = [t for t in rows if t < H_IN - 1]
    nor = [t for t in rows if t >= H_IN - 1]
    for name, keys in (("change inside the history (t0 0..10)", chg), ("normal samples (t0 11..%d)" % t0_max, nor)):
        print(f"  {name}: paper: model {avg(keys, 'mo'):.3f}%  persistence {avg(keys, 'po'):.3f}%   |  "
              f"strict: model {avg(keys, 'm'):.2f}%  persistence {avg(keys, 'p'):.2f}%")


def train_member(k, tr, te, grid, args, dev, cs, ds):
    torch.manual_seed(1000 + k)
    gen = torch.Generator().manual_seed(2000 + k)
    model = make_model(args, grid).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.05)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"member {k}: {args.model}, {n_par / 1e3:.0f}k parameters", flush=True)
    t_start, run = time.time(), []
    for it in range(1, args.iters + 1):
        xs, ys = [], []
        for _ in range(args.batch):
            kk, hist, fut, N, tau = draw(tr, gen, args.pad_prob)
            xs.append(build_input(grid, hist, tr.V[kk], N, tau, cs, ds))
            ys.append(grid.to_grid((fut - hist[:, -1:]) / ds))
        out = model(torch.stack(xs))
        loss = (((out - torch.stack(ys)) ** 2) * grid.mask).sum() / (grid.mask.sum() * out.shape[0] * T_OUT)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run.append(float(loss))
        if it % args.log_every == 0 or it == args.iters:
            msg = f"  m{k} it {it:6d}  MSE {np.mean(run):.4f}  {time.time() - t_start:.0f} s"
            run = []
            if it % args.eval_every == 0 or it == args.iters:
                model.eval()
                e_tr, _, _ = evaluate([model], grid, tr, cs, ds, every=4)
                e_te, b_te, _ = evaluate([model], grid, te, cs, ds, every=4)
                model.train()
                msg += f"  | l2 train {100 * e_tr:.2f}%  test {100 * e_te:.2f}%  (persistence {100 * b_te:.2f}%)"
            print(msg, flush=True)
    path = os.path.join(args.out, f"member{k}.pth")
    torch.save({"state": model.state_dict(), "model": args.model, "width": args.width, "modes": args.modes,
                "c_scale": cs, "d_scale": ds, "pad_prob": args.pad_prob}, path)
    print(f"  saved {path}", flush=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["linear", "mlp", "fno"], default="fno")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--scenarios", default=os.path.join(HERE, "scenarios.txt"))
    ap.add_argument("--tag", default="other")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32, help="fno: channels; mlp: hidden size (use 256)")
    ap.add_argument("--modes", type=int, default=16, help="fno: Fourier modes kept in each direction")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--transitions", action="store_true")
    ap.add_argument("--z", type=float, default=1.6)
    ap.add_argument("--c-offset", type=float, default=400.0, help="outdoor level for the paper metric")
    ap.add_argument("--pad-prob", type=float, default=0.9, help="as the final GNOT model: 90 %% of the samples with the change inside the history")
    ap.add_argument("--info", action="store_true", help="only load the data and print the grid (does the plane have holes?)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    trp, tep = split(args.data_dir, args.scenarios, False)
    tr = PlaneData(trp, dev, seed=0, z=args.z, transitions=args.transitions)
    te = PlaneData(tep, dev, seed=0, z=args.z, transitions=args.transitions)
    grid = Grid(tr.xyz[:, :2].cpu().numpy(), dev)
    print(f"data {args.data_dir}: {len(trp)} train / {len(tep)} test cases; {grid.info()}", flush=True)
    if args.info:
        return
    if args.model == "mlp" and args.width == 32:
        args.width = 256
    if args.model == "fno" and args.lr > 5e-4:
        args.lr = 5e-4
    args.out = os.path.join(HERE, "checkpoints", args.tag)
    os.makedirs(args.out, exist_ok=True)
    if any(f.endswith(".pth") for f in os.listdir(args.out)):
        raise SystemExit(f"{args.out} already has checkpoints -- use a new --tag")
    cs, ds = c_scale_of(tr), d_scale_of(tr)
    print(f"model {args.model}, lr {args.lr:g}, {args.iters} steps, batch {args.batch}, pad_prob {args.pad_prob}, device {dev}", flush=True)

    models = [train_member(k, tr, te, grid, args, dev, cs, ds) for k in range(args.members)]

    print("\nFINAL (every sample)   -> to compare with the GNOT table in the thesis")
    for name, d in (("train", tr), ("test", te)):
        eE, b, n = evaluate(models, grid, d, cs, ds)
        eO, bO, _ = evaluate(models, grid, d, cs, ds, offset=args.c_offset)
        print(f"  {name:5s}: PAPER METRIC ensemble {100 * eO:.2f}%  persistence {100 * bO:.2f}%   |   "
              f"STRICT ensemble {100 * eE:.2f}%  persistence {100 * b:.2f}%   [{n} samples]")
    for N in (10.0, 80.0):
        eE, b, _ = evaluate(models, grid, te, cs, ds, N=N)
        print(f"  test at N = {N:g} (strict): ensemble {100 * eE:.2f}%  persistence {100 * b:.2f}%")
    if args.transitions:
        print("window change inside the history (test scenarios):")
        history_change(models, grid, te, cs, ds, args.c_offset)


if __name__ == "__main__":
    main()
