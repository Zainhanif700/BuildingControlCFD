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
TAU_CAP = 12      # time since the last window change in maps (30 s each); from 6 min on the history lies after the change


N_PHYS = 6                    # number of physics features (physics_of)
WIN_AREA = 0.61 * 3.15        # m^2, a window of the model (full height of the room)
ROOM_VOLUME = 448.0           # m^3
PPM_FLOW_PER_PERSON = 3.96    # ppm m^3/s of CO2 per person (6 L/min at 40,000 ppm)


def physics_of(V, N):
    """What a well-mixed room would do for this window setting (B,1 tensors): fresh-air flow Q through the open inlet
    windows, the CO2 above outdoor level the room would approach (C_eq = people * 3.96 / Q) and its time constant
    (room volume / Q). Returns the features (B, N_PHYS) and C_eq (B, 1)."""
    Vw = V[:, :8]
    Q = (Vw.clamp(min=0.0) * WIN_AREA).sum(1, keepdim=True)
    Qc = Q.clamp(min=0.05)
    ceq = (N * PPM_FLOW_PER_PERSON / Qc).clamp(max=2000.0)
    troom = (ROOM_VOLUME / Qc).clamp(max=3600.0)
    n_win = (Vw > 0).float().sum(1, keepdim=True) / 8.0
    n_door = V[:, 8:10].sum(1, keepdim=True) / 2.0
    feats = torch.cat([Q / 10.0, torch.log(Qc) / 3.0, ceq / 1000.0, troom / 3600.0, n_win, n_door], 1)
    return feats, ceq


def tau_of(t0, transitions):
    """Input 'time since the last window change', scaled to 0..1. Transition runs start with the change at map 0, so
    the time is t0 maps; direct runs have no recent change (1)."""
    return min(float(t0), float(TAU_CAP)) / TAU_CAP if transitions else 1.0


def doors_of(d):
    """Door states of a data file (1 open, 0 closed); files from before the door input have both doors open."""
    return d["doors"].astype(np.float32) if "doors" in d.files else np.ones(2, np.float32)


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
            "V": d["V"].astype(np.float32), "N_ref": float(d["N_ref"]), "name": str(path).split("/")[-1],
            "doors": doors_of(d)}


def load_transition(path, z=1.6):
    """Read a transition file: the flushed CO2 of the previous setting (H) and the new setting's CO2."""
    d = np.load(path)
    zs = np.unique(d["P"][:, 2])
    pl = np.abs(d["P"][:, 2] - zs[np.argmin(np.abs(zs - (1.6 if z is None else z)))]) < 1e-5
    P = d["P"][pl]
    t = d["t_c"].astype(float)
    assert np.allclose(t, np.arange(0.0, t[-1] + 1e-6, DT)), f"{path}: not every {DT:g} s"
    return {"xy": P[:, :2].astype(np.float32), "z": float(P[0, 2]), "frames": d["S"][:, pl].astype(np.float32),
            "H": d["H"][:, pl].astype(np.float32), "V": d["V"].astype(np.float32), "N_ref": float(d["N_ref"]),
            "doors": doors_of(d),
            "name": str(path).split("/")[-1], "from": str(d["from_case"])}


class ForecastGNOT(nn.Module):
    def __init__(self, d=128, heads=4, layers=3, c_scale=1.0, d_scale=None, open_flags=False, use_tau=False,
                 film=False, physics=False):
        """film: the window setting, the people (and tau, physics) modulate every block (scale and shift, FiLM).
        physics: the model also gets physics_of(V, N) and, at every point, the distance of the CO2 to C_eq."""
        super().__init__()
        self.open_flags = bool(open_flags)
        self.use_tau = bool(use_tau)
        self.film = bool(film)
        self.physics = bool(physics)
        self.n_layers, self.width = layers, d
        self.c_scale = float(c_scale)
        self.d_scale = float(d_scale if d_scale is not None else c_scale)
        self.ff = FourierFeatures(room_length=ROOM_X[1] - ROOM_X[0], sigma=2.5, n_octaves=7)
        nf = 2 * self.ff.n_freq
        npt = 1 if self.physics else 0
        self.q_in = nn.Sequential(nn.Linear(nf + H_IN + npt, d), nn.GELU(), nn.Linear(d, d))
        self.h_in = nn.Sequential(nn.Linear(nf + H_IN + npt, d), nn.GELU(), nn.Linear(d, d))
        self.c_in = nn.Sequential(nn.Linear(3 + 1 + int(self.open_flags) + int(self.use_tau)
                                            + (N_PHYS if self.physics else 0), d), nn.GELU(), nn.Linear(d, d))
        if self.film:
            g_dim = 8 + 8 + 2 + 1 + int(self.use_tau) + (N_PHYS if self.physics else 0)
            self.film_net = nn.Sequential(nn.Linear(g_dim, d), nn.GELU(), nn.Linear(d, 2 * d * layers))
            nn.init.zeros_(self.film_net[-1].weight)          # starts as the plain model
            nn.init.zeros_(self.film_net[-1].bias)
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

    def _point_feats(self, xyz, hist, ceq=None):
        """Point features: position encoding, current CO2 level and recent trend (with physics: distance to C_eq)."""
        B, M, _ = xyz.shape
        f = self.ff(xyz.reshape(-1, 3)).reshape(B, M, -1)
        last = hist[..., -1:]
        parts = [f, last / self.c_scale, (hist[..., :-1] - last) / self.d_scale]
        if self.physics:
            parts.append((ceq.reshape(B, 1, 1) - last) / self.c_scale)
        return torch.cat(parts, -1)

    def forward(self, q_xyz, q_hist, h_xyz, h_hist, V, N, tau=None):
        """Mean and log-variance of the next 6 frames at the query points.
        V: 8 window values (> 0 inlet speed, 0 closed, < 0 outlet window), followed by the 2 door states.
        tau (only with use_tau): time since the last window change, scaled to 0..1, shape (B, 1)."""
        s = self.d_scale
        B = V.shape[0]
        pf, ceq = physics_of(V, N) if self.physics else (None, None)
        q = self.q_in(self._point_feats(q_xyz, q_hist, ceq))
        hk = self.h_in(self._point_feats(h_xyz, h_hist, ceq)) + self.type_emb.weight[3]
        Vw = V[:, :8]
        if self.open_flags:
            Dr = V[:, 8:10]
            speed = torch.cat([Vw.clamp(min=0.0) / V_MAX, torch.zeros_like(Dr), N / N_MAX_FC], 1)
            opened = torch.cat([(Vw != 0).float(), Dr, torch.zeros_like(N)], 1)
            val = torch.stack([speed, opened], -1)
        else:
            val = torch.cat([Vw / V_MAX, torch.zeros(B, 2, device=V.device), N / N_MAX_FC], 1).unsqueeze(-1)
        cin = torch.cat([self.tok_pos.unsqueeze(0).expand(B, -1, -1), val], -1)
        if self.use_tau:
            assert tau is not None, "this model needs the time since the last window change (tau)"
            cin = torch.cat([cin, tau.reshape(B, 1, 1).expand(-1, cin.shape[1], -1).to(cin.dtype)], -1)
        if self.physics:
            cin = torch.cat([cin, pf.unsqueeze(1).expand(-1, cin.shape[1], -1).to(cin.dtype)], -1)
        ck = self.c_in(cin)
        if self.film:
            g = [Vw / V_MAX, (Vw != 0).float(), V[:, 8:10], N / N_MAX_FC]
            if self.use_tau:
                g.append(tau.reshape(B, 1).to(Vw.dtype))
            if self.physics:
                g.append(pf)
            gb = self.film_net(torch.cat(g, 1)).reshape(B, self.n_layers, 2, self.width)
        types = torch.tensor([0] * 8 + [1] * 2 + [2], device=V.device)
        ck = ck + self.type_emb(types).unsqueeze(0)
        ctx = torch.cat([ck, hk], 1)
        for i, b in enumerate(self.blocks):
            a, _ = b["att"](q, ctx, ctx)
            q = b["n1"](q + a)
            q = b["n2"](q + b["ff"](q))
            if self.film:
                q = q * (1.0 + gb[:, i, 0].unsqueeze(1)) + gb[:, i, 1].unsqueeze(1)
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
        self.V = [torch.tensor(np.concatenate([c["V"], c["doors"]]), device=device) for c in self.cases]
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

    def sample_padded(self, k, t0, N, NA):
        """Transition runs: history with the window change INSIDE it (t0 = 0 .. H_IN - 2). The maps before the change
        are the map at the change, repeated (the old CO2 is assumed constant before the change)."""
        f = self.frames[k] * (N / self.N_ref[k]) + self.H[k] * (NA / self.N_ref[k])
        hist = torch.stack([f[max(i, 0)] for i in range(t0 - H_IN + 1, t0 + 1)], 0).T
        fut = f[t0 + 1:t0 + 1 + T_OUT].T
        return hist, fut

    def batch(self, B, Q, gen, n_lo=10.0, n_hi=80.0, pad_prob=0.0, with_tau=False):
        """pad_prob > 0 (transition data only): this share of the samples has the window change inside the history.
        with_tau: also return the time since the last window change (B, 1) as an eighth value."""
        qh, qx, hh, hx, V, Ns, tg, taus = [], [], [], [], [], [], [], []
        Np = self.xyz.shape[0]
        for _ in range(B):
            k = int(torch.randint(len(self.cases), (1,), generator=gen))
            t0 = self.t0s[k][int(torch.randint(len(self.t0s[k]), (1,), generator=gen))]
            N = float(n_lo + (n_hi - n_lo) * torch.rand(1, generator=gen))
            NA = float(n_lo + (n_hi - n_lo) * torch.rand(1, generator=gen)) if self.H is not None else 0.0
            if pad_prob > 0 and self.H is not None and float(torch.rand(1, generator=gen)) < pad_prob:
                t0 = int(torch.randint(H_IN - 1, (1,), generator=gen))
                hist, fut = self.sample_padded(k, t0, N, NA)
            else:
                hist, fut = self.sample(k, t0, N, NA)
            taus.append(tau_of(t0, self.H is not None))
            qi = torch.randint(Np, (Q,), generator=gen).to(self.dev)
            qh.append(hist[qi]); qx.append(self.xyz[qi]); tg.append(fut[qi])
            hh.append(hist[self.hist_idx]); hx.append(self.xyz[self.hist_idx])
            V.append(self.V[k]); Ns.append(N)
        st = lambda a: torch.stack(a)
        out = (st(qx), st(qh), st(hx), st(hh), st(V), torch.tensor(Ns, device=self.dev).unsqueeze(1), st(tg))
        if with_tau:
            out = out + (torch.tensor(taus, device=self.dev, dtype=torch.float32).unsqueeze(1),)
        return out


@torch.no_grad()
def predict_full(model, data, k, t0, N, NA=0.0, chunk=4096):
    """Prediction on the whole plane for one sample."""
    hist, fut = data.sample(k, t0, N, NA)
    hh, hx = hist[data.hist_idx].unsqueeze(0), data.xyz[data.hist_idx].unsqueeze(0)
    V = data.V[k].unsqueeze(0)
    Nt = torch.tensor([[N]], device=data.dev)
    kw = {"tau": torch.tensor([[tau_of(t0, data.H is not None)]], device=data.dev)} if getattr(model, "use_tau", False) else {}
    out = []
    for i in range(0, hist.shape[0], chunk):
        m, _ = model(data.xyz[i:i + chunk].unsqueeze(0), hist[i:i + chunk].unsqueeze(0), hx, hh, V, Nt, **kw)
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
