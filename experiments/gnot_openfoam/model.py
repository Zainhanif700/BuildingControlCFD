"""
Supervised GNOT: the SAME backbone as Track 1 (token encoder for windows/doors/occupancy, query
encoder with Fourier features, cross-attention blocks -- gnot_model.GNOTOperator, imported), but the
outputs are the fields themselves instead of a vector potential, because the training targets are
OpenFOAM data (no PDE residuals, no curl, no second derivatives):

    u = s(V) * raw_uvw,               s(V) = door jet speed (gnot_model.door_jet_speed): u = 0 exactly
                                      with all windows closed, raw output O(1)
    C = C_REF (t/T_MAX) (N/N_MAX) C_hat   exact C(t=0) = 0 and exact linearity in N (as in Track 1;
                                      physical t, also for 30-min horizons)

The exact constraints above hold in the data by construction (room at rest and CO2-free at t = 0;
no buoyancy -> C proportional to N), so building them in is not an assumption about the flow.
"""
import torch
import torch.nn as nn

import common  # noqa: F401  (sys.path)
from gnot_model import GNOTOperator, door_jet_speed
from point_sampler import C_REF, T_MAX, N_PEOPLE_MAX


class SupervisedGNOT(nn.Module):
    def __init__(self, t_horizon=T_MAX):
        """t_horizon: longest time in the data [s]. The Track-1 query encoder normalises t by T_MAX
        (120 s); for longer horizons its time input is rescaled by T_MAX / t_horizon so it stays in
        [0, 1] (its 'ramp' feature tanh(3 t / TAU_RAMP) then acts on a correspondingly longer time
        scale -- a plain input feature, no physics attached in the supervised model)."""
        super().__init__()
        self.core = GNOTOperator()           # only token/query encoders, blocks and out_head are used
        self.t_horizon = float(t_horizon)

    def forward(self, x, y, z, t, V, N_people):
        """Inputs (B,1) except V (B,8); returns u, v, w, C (B,1) in physical units."""
        c = self.core
        context = c._build_context(V, torch.full_like(N_people, N_PEOPLE_MAX))   # N only via the factor
        q = c.query_encoder(x, y, z, t * (T_MAX / self.t_horizon)).unsqueeze(1)
        for block in c.blocks:
            q = block(q, context)
        out = c.out_head(q).squeeze(1)       # (B, 5): 0-2 velocity, 3 C_hat, 4 unused (pressure slot)
        s = door_jet_speed(V)
        u, v, w = s * out[:, 0:1], s * out[:, 1:2], s * out[:, 2:3]
        C = C_REF * (t / T_MAX) * (N_people / N_PEOPLE_MAX) * out[:, 3:4]
        return u, v, w, C
