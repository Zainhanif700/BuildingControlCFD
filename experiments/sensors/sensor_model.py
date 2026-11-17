"""
Sensor version of the forecast model.
The normal model knows the CO2 history at 900 points of the plane and at every point where it predicts.
Here the model only knows the CO2 history at the sensor stations and has to predict the whole plane from it.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gnot_openfoam_rans"))
import common  # noqa: F401  (adds the geometry folders to the path)
from forecast import ForecastGNOT, PlaneData, T_OUT, N_MAX_FC, H_IN
from point_sampler import V_MAX

import stations as st


WIN_AREA = 0.61 * 3.15                       # m2, one window opening over the full height (the model speed is over this area)
ROOM_VOLUME = 15.53 * 9.16 * 3.15            # m3
CO2_PER_PERSON = common.PPM_M3S_PER_PERSON   # ppm * m3/s of CO2 added by one person (6 L/min exhaled at 40,000 ppm)


def physics_features(V, N):
    """Well-mixed room balance as six numbers per sample. V: (B, 10) window speeds (> 0 inlet, < 0 outlet, 0 closed)
    and door states; N: (B, 1) people. Returns (B, 6):
      log10 of the fresh-air flow Q [m3/s] (inlet windows only, at least 0.1), log10 of the room time constant
      V_room / Q [s] / 1000, the CO2 level the room approaches N * g / Q [ppm] / 500 (capped at 1000 ppm),
      number of inlet windows / 8, of outlet windows / 8, of open doors / 2."""
    Vw, Dr = V[:, :8], V[:, 8:10]
    q = (Vw.clamp(min=0.0) * WIN_AREA).sum(1, keepdim=True).clamp(min=0.1)
    ceq = (N * CO2_PER_PERSON / q).clamp(max=1000.0)
    tau = ROOM_VOLUME / q
    return torch.cat([torch.log10(q), torch.log10(tau / 1000.0), ceq / 500.0,
                      (Vw > 0).float().sum(1, keepdim=True) / 8.0, (Vw < 0).float().sum(1, keepdim=True) / 8.0,
                      Dr.sum(1, keepdim=True) / 2.0], 1)


class SensorForecastGNOT(ForecastGNOT):
    """anchor=False: a point where we predict only has its position, the forecast is the mean of the newest
    station values plus a change.
    anchor=True: the station histories are interpolated to the point (nearer stations count more), and the forecast
    is this interpolated value plus a change, like in the normal model."""
    def __init__(self, d=128, heads=4, layers=3, c_scale=1.0, d_scale=None, anchor=False, blind=False,
                 hist_len=H_IN, people_blind=False, physics=False):
        super().__init__(d=d, heads=heads, layers=layers, c_scale=c_scale, d_scale=d_scale, open_flags=True)
        self.anchor = bool(anchor)
        self.physics = bool(physics)     # one more token with the well-mixed room balance (see physics_features)
        if self.physics:
            self.phys_in = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        # test only: the model is not told the number of people (the real system does not know it either)
        self.people_blind = bool(people_blind)
        self.blind = bool(blind)         # test only: the station CO2 is set to zero, only windows, doors and people count
        self.hist_len = int(hist_len)    # number of 30 s steps of station history (12 = 6 min)
        nf = 2 * self.ff.n_freq
        if self.hist_len != H_IN:        # the parent model is built for 12 steps
            self.h_in = nn.Sequential(nn.Linear(nf + self.hist_len, d), nn.GELU(), nn.Linear(d, d))
        if self.anchor:
            if self.hist_len != H_IN:
                self.q_in = nn.Sequential(nn.Linear(nf + self.hist_len, d), nn.GELU(), nn.Linear(d, d))
        else:
            self.q_in = nn.Sequential(nn.Linear(nf, d), nn.GELU(), nn.Linear(d, d))

    @staticmethod
    def interpolate(q_xyz, h_xyz, h_hist, eps=0.25):
        """History at the query points from the station histories: weights 1 / (distance^2 + eps)."""
        w = 1.0 / (torch.cdist(q_xyz, h_xyz) ** 2 + eps)          # (B, M, stations)
        w = w / w.sum(-1, keepdim=True)
        return torch.bmm(w, h_hist)                               # (B, M, history steps)

    def forward(self, q_xyz, q_hist, h_xyz, h_hist, V, N):
        """q_hist is not used. It is only in the list so that the old training and test code still works."""
        B, M, _ = q_xyz.shape
        if self.blind:
            h_hist = torch.zeros_like(h_hist)
        if self.people_blind:
            N = torch.full_like(N, 0.5 * N_MAX_FC)
        if self.anchor:
            q_hist = self.interpolate(q_xyz, h_xyz, h_hist)
            q = self.q_in(self._point_feats(q_xyz, q_hist))
            base = q_hist[..., -1:]
        else:
            q = self.q_in(self.ff(q_xyz.reshape(-1, 3)).reshape(B, M, -1))
            # start from the mean of the newest station values and predict the difference to it
            base = h_hist[..., -1].mean(1).reshape(B, 1, 1)
        hk = self.h_in(self._point_feats(h_xyz, h_hist)) + self.type_emb.weight[3]

        Vw, Dr = V[:, :8], V[:, 8:10]
        speed = torch.cat([Vw.clamp(min=0.0) / V_MAX, torch.zeros_like(Dr), N / N_MAX_FC], 1)
        opened = torch.cat([(Vw != 0).float(), Dr, torch.zeros_like(N)], 1)
        ck = self.c_in(torch.cat([self.tok_pos.unsqueeze(0).expand(B, -1, -1), torch.stack([speed, opened], -1)], -1))
        types = torch.tensor([0] * 8 + [1] * 2 + [2], device=V.device)
        ctx = torch.cat([ck + self.type_emb(types).unsqueeze(0), hk], 1)
        if self.physics:
            ptok = self.phys_in(physics_features(V, N)) + self.type_emb.weight[2]
            ctx = torch.cat([ctx, ptok.unsqueeze(1)], 1)

        for b in self.blocks:
            a, _ = b["att"](q, ctx, ctx)
            q = b["n1"](q + a)
            q = b["n2"](q + b["ff"](q))
        out = self.head(q)
        mean = base + self.c_scale * out[..., :T_OUT]
        logvar = out[..., T_OUT:].clamp(-12.0, 6.0)
        return mean, logvar


class SensorPlaneData(PlaneData):
    """Same data as PlaneData, but the history comes only from the plane points next to the stations."""
    def __init__(self, paths, device, z=st.Z_STATION, transitions=False, noise=0.0, stations=None, hist_len=H_IN,
                 t0_min=0, t0_max=None):
        super().__init__(paths, device, seed=0, z=z, transitions=transitions)
        stations = stations or st.STATIONS
        xy = self.xyz[:, :2].cpu().numpy()
        cells = [int(np.argmin((xy[:, 0] - s.x) ** 2 + (xy[:, 1] - s.y) ** 2)) for s in stations]
        self.hist_idx = torch.tensor(cells, device=device)
        self.noise = float(noise)
        # a longer history needs later start times; t0_min makes runs with different histories use the same samples
        self.hist_len = int(hist_len)
        last = lambda f: f.shape[0] - T_OUT if t0_max is None else min(f.shape[0] - T_OUT, int(t0_max) + 1)
        self.t0s = [list(range(max(self.hist_len - 1, int(t0_min)), last(f))) for f in self.frames]

    def sample(self, k, t0, N, NA=0.0):
        """History (hist_len steps) and future frames for N people now and NA people before."""
        f = self.frames[k] * (N / self.N_ref[k])
        if self.H is not None:
            f = f + self.H[k] * (NA / self.N_ref[k])
        return f[t0 - self.hist_len + 1:t0 + 1].T, f[t0 + 1:t0 + 1 + T_OUT].T

    def batch(self, B, Q, gen, n_lo=10.0, n_hi=80.0):
        qx, qh, hx, hh, V, N, tg = super().batch(B, Q, gen, n_lo, n_hi)
        if self.noise > 0:
            hh = hh + self.noise * torch.randn(hh.shape, generator=gen).to(hh.device)
        return qx, qh, hx, hh, V, N, tg
