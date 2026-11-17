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
from forecast import ForecastGNOT, PlaneData, T_OUT, N_MAX_FC
from point_sampler import V_MAX

import stations as st


class SensorForecastGNOT(ForecastGNOT):
    """anchor=False: a point where we predict only has its position, the forecast is the mean of the newest
    station values plus a change.
    anchor=True: the station histories are interpolated to the point (nearer stations count more), and the forecast
    is this interpolated value plus a change, like in the normal model."""
    def __init__(self, d=128, heads=4, layers=3, c_scale=1.0, d_scale=None, anchor=False):
        super().__init__(d=d, heads=heads, layers=layers, c_scale=c_scale, d_scale=d_scale, open_flags=True)
        self.anchor = bool(anchor)
        if not self.anchor:
            nf = 2 * self.ff.n_freq
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
    def __init__(self, paths, device, z=st.Z_STATION, transitions=False, noise=0.0, stations=None):
        super().__init__(paths, device, seed=0, z=z, transitions=transitions)
        stations = stations or st.STATIONS
        xy = self.xyz[:, :2].cpu().numpy()
        cells = [int(np.argmin((xy[:, 0] - s.x) ** 2 + (xy[:, 1] - s.y) ** 2)) for s in stations]
        self.hist_idx = torch.tensor(cells, device=device)
        self.noise = float(noise)

    def batch(self, B, Q, gen, n_lo=10.0, n_hi=80.0):
        qx, qh, hx, hh, V, N, tg = super().batch(B, Q, gen, n_lo, n_hi)
        if self.noise > 0:
            hh = hh + self.noise * torch.randn(hh.shape, generator=gen).to(hh.device)
        return qx, qh, hx, hh, V, N, tg
