"""
The CO2 forecast model (GNOT-style, as in Bian & Shi 2025): last 6 min of CO2 on a plane -> next 3 min, plus data loading and the l2 error.
"""
import math

import numpy as np
import torch
import torch.nn as nn

import common
from gnot_model import FourierFeatures, WINDOW_CENTERS, DOOR_CENTERS, ROOM_CENTER
from point_sampler import ROOM_X, ROOM_Y, ROOM_Z, V_MAX

H_IN, T_OUT, DT = 12, 6, 30.0
N_MAX_FC = 80.0
N_HIST_TOKENS = 900


def load_plane(path, z=None):
    """Read the CO2 on one plane every 30 s from a dataset file."""
    d = np.load(path)
    t = d["t_c"].astype(float)
    grid = np.arange(0.0, t[-1] + 1e-6, DT)
    idx = [int(np.argmin(np.abs(t - g))) for g in grid]
    assert all(abs(t[i] - g) < 1e-3 for i, g in zip(idx, grid)), f"{path}: CO2 not saved every {DT:g} s"
    if z is None:
        pl = d["plane"].astype(bool)
    else:
        zs = np.unique(d["P"][:, 2])
        pl = np.abs(d["P"][:, 2] - zs[np.argmin(np.abs(zs - z))]) < 1e-5
    P = d["P"][pl]
    return {"xy": P[:, :2].astype(np.float32), "z": float(P[0, 2]), "frames": d["C"][idx][:, pl].astype(np.float32),
            "V": d["V"].astype(np.float32), "N_ref": float(d["N_ref"]), "name": str(path).split("/")[-1]}


def load_transition(path, z=1.6):
    """Read a transition file: the flushed CO2 of the previous setting (H) and the new setting's CO2."""
    d = np.load(path)
    zs = np.unique(d["P"][:, 2])
    pl = np.abs(d["P"][:, 2] - zs[np.argmin(np.abs(zs - (1.6 if z is None else z)))]) < 1e-5
    P = d["P"][pl]
    assert np.allclose(d["t_c"], np.arange(0.0, d["t_c"][-1] + 1e-6, DT)), f"{path}: not every {DT:g} s"
    return {"xy": P[:, :2].astype(np.float32), "z": float(P[0, 2]), "frames": d["S"][:, pl].astype(np.float32),
            "H": d["H"][:, pl].astype(np.float32), "V": d["V"].astype(np.float32), "N_ref": float(d["N_ref"]),
            "name": str(path).split("/")[-1], "from": str(d["from_case"])}


class ForecastGNOT(nn.Module):
    def __init__(self, d=128, heads=4, layers=3, c_scale=1.0, d_scale=None):
        super().__init__()
        self.c_scale = float(c_scale)
        self.d_scale = float(d_scale if d_scale is not None else c_scale)
        self.ff = FourierFeatures(room_length=ROOM_X[1] - ROOM_X[0], sigma=2.5, n_octaves=7)
        nf = 2 * self.ff.n_freq
        self.q_in = nn.Sequential(nn.Linear(nf + H_IN, d), nn.GELU(), nn.Linear(d, d))
        self.h_in = nn.Sequential(nn.Linear(nf + H_IN, d), nn.GELU(), nn.Linear(d, d))
        self.c_in = nn.Sequential(nn.Linear(3 + 1, d), nn.GELU(), nn.Linear(d, d))
        self.type_emb = nn.Embedding(4, d)
        self.blocks = nn.ModuleList([nn.ModuleDict({
            "att": nn.MultiheadAttention(d, heads, batch_first=True), "n1": nn.LayerNorm(d),
            "ff": nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)), "n2": nn.LayerNorm(d)})
            for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2 * T_OUT))
        lo = torch.tensor([ROOM_X[0], ROOM_Y[0], ROOM_Z[0]])
        ext = torch.tensor([ROOM_X[1] - ROOM_X[0], ROOM_Y[1] - ROOM_Y[0], ROOM_Z[1] - ROOM_Z[0]])
        assert len(WINDOW_CENTERS) == 8 and len(DOOR_CENTERS) == 2
        pos = torch.tensor(list(WINDOW_CENTERS) + list(DOOR_CENTERS) + [ROOM_CENTER], dtype=torch.float32)
        self.register_buffer("tok_pos", (pos - lo) / ext)

    def _point_feats(self, xyz, hist):
        """Point features: position encoding, current CO2 level and recent trend."""
        B, M, _ = xyz.shape
        f = self.ff(xyz.reshape(-1, 3)).reshape(B, M, -1)
        last = hist[..., -1:]
        return torch.cat([f, last / self.c_scale, (hist[..., :-1] - last) / self.d_scale], -1)

    def forward(self, q_xyz, q_hist, h_xyz, h_hist, V, N):
        """Mean and log-variance of the next 6 frames at the query points."""
        s = self.d_scale
        q = self.q_in(self._point_feats(q_xyz, q_hist))
        hk = self.h_in(self._point_feats(h_xyz, h_hist)) + self.type_emb.weight[3]
        B = V.shape[0]
        val = torch.cat([V / V_MAX, torch.zeros(B, 2, device=V.device), N / N_MAX_FC], 1).unsqueeze(-1)
        ck = self.c_in(torch.cat([self.tok_pos.unsqueeze(0).expand(B, -1, -1), val], -1))
        types = torch.tensor([0] * 8 + [1] * 2 + [2], device=V.device)
        ck = ck + self.type_emb(types).unsqueeze(0)
        ctx = torch.cat([ck, hk], 1)
        for b in self.blocks:
            a, _ = b["att"](q, ctx, ctx)
            q = b["n1"](q + a)
            q = b["n2"](q + b["ff"](q))
        out = self.head(q)
        mean = q_hist[..., -1:] + s * out[..., :T_OUT]
        logvar = out[..., T_OUT:].clamp(-12.0, 6.0)
        return mean, logvar


def nll(mean, logvar, target, scale):
    """Gaussian negative log-likelihood (the paper's loss)."""
    r = (target - mean) / scale
    return 0.5 * (logvar + math.log(2 * math.pi) + r ** 2 * torch.exp(-logvar)).mean()


def l2_rel(pred, target):
    """Relative l2 error of one sample (the paper's eq. 12)."""
    return float(torch.linalg.norm(pred - target) / torch.linalg.norm(target).clamp_min(1e-30))


class PlaneData:
    """All cases of one split on the GPU; gives history/future samples for any number of people."""
    def __init__(self, paths, device, seed=0, z=None, transitions=False):
        self.cases = [load_transition(p, z) if transitions else load_plane(p, z) for p in paths]
        xy0 = self.cases[0]["xy"]
        for c in self.cases:
            assert c["xy"].shape == xy0.shape and np.allclose(c["xy"], xy0), "all cases must share the grid"
        self.dev = device
        self.xyz = torch.tensor(np.c_[xy0, np.full(len(xy0), self.cases[0]["z"], np.float32)], device=device)
        self.frames = [torch.tensor(c["frames"], device=device) for c in self.cases]
        self.V = [torch.tensor(c["V"], device=device) for c in self.cases]
        self.N_ref = [c["N_ref"] for c in self.cases]
        self.H = [torch.tensor(c["H"], device=device) for c in self.cases] if transitions else None
        rng = np.random.default_rng(seed)
        self.hist_idx = torch.tensor(np.sort(rng.choice(len(xy0), min(N_HIST_TOKENS, len(xy0)), replace=False)), device=device)
        self.t0s = [list(range(H_IN - 1, f.shape[0] - T_OUT)) for f in self.frames]

    def sample(self, k, t0, N, NA=0.0):
        """History and future frames for N people now and NA people before."""
        f = self.frames[k] * (N / self.N_ref[k])
        if self.H is not None:
            f = f + self.H[k] * (NA / self.N_ref[k])
        hist = f[t0 - H_IN + 1:t0 + 1].T
        fut = f[t0 + 1:t0 + 1 + T_OUT].T
        return hist, fut

    def batch(self, B, Q, gen, n_lo=10.0, n_hi=80.0):
        qh, qx, hh, hx, V, Ns, tg = [], [], [], [], [], [], []
        Np = self.xyz.shape[0]
        for _ in range(B):
            k = int(torch.randint(len(self.cases), (1,), generator=gen))
            t0 = self.t0s[k][int(torch.randint(len(self.t0s[k]), (1,), generator=gen))]
            N = float(n_lo + (n_hi - n_lo) * torch.rand(1, generator=gen))
            NA = float(n_lo + (n_hi - n_lo) * torch.rand(1, generator=gen)) if self.H is not None else 0.0
            hist, fut = self.sample(k, t0, N, NA)
            qi = torch.randint(Np, (Q,), generator=gen).to(self.dev)
            qh.append(hist[qi]); qx.append(self.xyz[qi]); tg.append(fut[qi])
            hh.append(hist[self.hist_idx]); hx.append(self.xyz[self.hist_idx])
            V.append(self.V[k]); Ns.append(N)
        st = lambda a: torch.stack(a)
        return st(qx), st(qh), st(hx), st(hh), st(V), torch.tensor(Ns, device=self.dev).unsqueeze(1), st(tg)


@torch.no_grad()
def predict_full(model, data, k, t0, N, NA=0.0, chunk=4096):
    """Prediction on the whole plane for one sample."""
    hist, fut = data.sample(k, t0, N, NA)
    hh, hx = hist[data.hist_idx].unsqueeze(0), data.xyz[data.hist_idx].unsqueeze(0)
    V = data.V[k].unsqueeze(0)
    Nt = torch.tensor([[N]], device=data.dev)
    out = []
    for i in range(0, hist.shape[0], chunk):
        m, _ = model(data.xyz[i:i + chunk].unsqueeze(0), hist[i:i + chunk].unsqueeze(0), hx, hh, V, Nt)
        out.append(m[0])
    return torch.cat(out), fut, hist


def evaluate(models, data, N=None, every=1, offset=0.0):
    """Mean l2 error of the ensemble over all samples, and of 'CO2 stays the same'."""
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
            preds = [predict_full(m, data, k, t0, n, na) for m in models]
            mean = torch.stack([p[0] for p in preds]).mean(0)
            fut, hist = preds[0][1], preds[0][2]
            if torch.linalg.norm(fut) == 0:
                continue
            errs.append(l2_rel(mean + offset, fut + offset))
            base.append(l2_rel(hist[:, -1:].expand_as(fut) + offset, fut + offset))
    return float(np.mean(errs)), float(np.mean(base)), len(errs)
