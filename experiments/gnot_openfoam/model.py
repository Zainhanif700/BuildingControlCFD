"""
Supervised GNOT for the laminar data: the same backbone as the physics-only model, trained on the fields directly.
"""
import torch
import torch.nn as nn

import common
from gnot_model import GNOTOperator, door_jet_speed
from point_sampler import C_REF, T_MAX, N_PEOPLE_MAX


class SupervisedGNOT(nn.Module):
    def __init__(self, t_horizon=T_MAX):
        """t_horizon: longest time in the data [s]."""
        super().__init__()
        self.core = GNOTOperator()
        self.t_horizon = float(t_horizon)

    def forward(self, x, y, z, t, V, N_people):
        """Velocity and CO2 at the query points; inputs (B, 1) except V (B, 8)."""
        c = self.core
        context = c._build_context(V, torch.full_like(N_people, N_PEOPLE_MAX))
        q = c.query_encoder(x, y, z, t * (T_MAX / self.t_horizon)).unsqueeze(1)
        for block in c.blocks:
            q = block(q, context)
        out = c.out_head(q).squeeze(1)
        s = door_jet_speed(V)
        u, v, w = s * out[:, 0:1], s * out[:, 1:2], s * out[:, 2:3]
        C = C_REF * (t / T_MAX) * (N_people / N_PEOPLE_MAX) * out[:, 3:4]
        return u, v, w, C
