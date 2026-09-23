"""
GNOT-style operator for the real room (physics-only training, no data).

HOW THIS DIFFERS FROM PINTO (kept as separate, independent tracks per your
request -- this is not shared code with experiments/pinto/):

  PINTO (experiments/pinto/pinto_channel_flow_test.py): the network is only
  ever given a handful of SENSOR POINTS of the same type (velocity readings
  at fixed locations) and has to infer everything else via cross-attention
  over those.

  GNOT (this file): the network is given several DIFFERENT TYPES of
  boundary information as separate tokens -- 8 window tokens (each with its
  own position AND its own velocity V_k), 2 door tokens (position only, fixed
  outlet rule), and 1 occupancy token (room-wide N_people, no position). This
  matches GNOT's own idea of handling *heterogeneous* input tokens, rather
  than PINTO's uniform sensor-point idea.

Query point (x, y, z, t) attends over all these context tokens via
cross-attention, then decodes to (u, v, w, p, c) using the same
vector-potential / curl trick used in pino_parametric_3d_test.py (guarantees
divergence-free velocity everywhere, for free, by construction).

No simulation data anywhere here -- trained purely against the Navier-Stokes
+ advection-diffusion residuals, exactly like the parametric experiments.
"""
import torch
import torch.nn as nn

from point_sampler import NUM_WINDOWS, WINDOWS, DOORS, ROOM_X, ROOM_Y, ROOM_Z

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2


def _window_centers():
    """(x_center, y_center, z_center) for each of the 8 real windows."""
    centers = []
    for xlo, xhi, zlo, zhi in WINDOWS:
        centers.append(((xlo + xhi) / 2, ROOM_Y[1], (zlo + zhi) / 2))
    return centers


def _door_centers():
    centers = []
    for xlo, xhi, zlo, zhi in DOORS:
        centers.append(((xlo + xhi) / 2, ROOM_Y[0], (zlo + zhi) / 2))
    return centers


WINDOW_CENTERS = _window_centers()
DOOR_CENTERS = _door_centers()
ROOM_CENTER = ((ROOM_X[0] + ROOM_X[1]) / 2, (ROOM_Y[0] + ROOM_Y[1]) / 2, (ROOM_Z[0] + ROOM_Z[1]) / 2)


class TokenEncoder(nn.Module):
    """Encodes one context token: its fixed real-world position (3,) + its
    scenario value (1, e.g. V_k or N_people) + a learned type embedding
    (window / door / occupancy) -> a d_model vector."""

    def __init__(self, d_model=D_MODEL, n_types=3):
        super().__init__()
        self.type_embed = nn.Embedding(n_types, d_model)
        self.proj = nn.Sequential(
            nn.Linear(3 + 1, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, pos, value, type_id):
        # pos: (B, K, 3), value: (B, K, 1), type_id: (B, K) long
        x = torch.cat([pos, value], dim=-1)
        return self.proj(x) + self.type_embed(type_id)


class QueryEncoder(nn.Module):
    """Encodes the query point (x, y, z, t) -> d_model vector."""

    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )

    def forward(self, x, y, z, t):
        return self.proj(torch.cat([x, y, z, t], dim=-1))


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, context):
        # query: (B, 1, D), context: (B, K, D)
        attn_out, _ = self.attn(query, context, context)
        query = self.norm1(query + attn_out)
        query = self.norm2(query + self.ff(query))
        return query


class GNOTOperator(nn.Module):
    """Full operator: (x,y,z,t,V1..V8,N_people) -> (u,v,w,p,c) for the real room.

    Divergence-free velocity is enforced by construction via the
    vector-potential/curl trick (same as pino_parametric_3d_test.py):
        u = dA3/dy - dA2/dz
        v = dA1/dz - dA3/dx
        w = dA2/dx - dA1/dy
    """

    def __init__(self, d_model=D_MODEL, n_layers=N_LAYERS):
        super().__init__()
        self.token_encoder = TokenEncoder(d_model, n_types=3)  # 0=window,1=door,2=occupancy
        self.query_encoder = QueryEncoder(d_model)
        self.blocks = nn.ModuleList([CrossAttnBlock(d_model) for _ in range(n_layers)])
        self.out_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 5)  # A1,A2,A3,C,p
        )

        # fixed real-world token positions, registered as buffers (not trained)
        self.register_buffer("window_pos", torch.tensor(WINDOW_CENTERS, dtype=torch.float32))  # (8,3)
        self.register_buffer("door_pos", torch.tensor(DOOR_CENTERS, dtype=torch.float32))       # (2,3)
        self.register_buffer("room_center", torch.tensor([ROOM_CENTER], dtype=torch.float32))   # (1,3)

    def _build_context(self, V, N_people):
        """V: (B, 8), N_people: (B, 1) -> context tokens (B, 11, D)"""
        B = V.shape[0]
        device = V.device

        win_pos = self.window_pos.unsqueeze(0).expand(B, -1, -1)          # (B,8,3)
        win_val = V.unsqueeze(-1)                                         # (B,8,1)
        win_type = torch.zeros(B, NUM_WINDOWS, dtype=torch.long, device=device)

        door_pos = self.door_pos.unsqueeze(0).expand(B, -1, -1)           # (B,2,3)
        door_val = torch.zeros(B, door_pos.shape[1], 1, device=device)    # doors have no "value"
        door_type = torch.ones(B, door_pos.shape[1], dtype=torch.long, device=device)

        occ_pos = self.room_center.unsqueeze(0).expand(B, -1, -1)  # (B,1,3)
        occ_val = N_people.view(B, 1, 1)
        occ_type = torch.full((B, 1), 2, dtype=torch.long, device=device)

        pos = torch.cat([win_pos, door_pos, occ_pos], dim=1)
        val = torch.cat([win_val, door_val, occ_val], dim=1)
        type_id = torch.cat([win_type, door_type, occ_type], dim=1)
        return self.token_encoder(pos, val, type_id)  # (B, 11, D)

    def forward(self, x, y, z, t, V, N_people):
        """All inputs shape (B,1) except V which is (B,8). Returns u,v,w,p,c each (B,1)."""
        context = self._build_context(V, N_people)          # (B, 11, D)
        q = self.query_encoder(x, y, z, t).unsqueeze(1)      # (B, 1, D)

        for block in self.blocks:
            q = block(q, context)

        out = self.out_head(q).squeeze(1)  # (B, 5)
        A1, A2, A3, C, p = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4], out[:, 4:5]
        return A1, A2, A3, C, p

    def velocity_from_potential(self, A1, A2, A3, x, y, z):
        """Curl trick: differentiate the vector potential to get a
        divergence-free (u, v, w). Requires x,y,z to have requires_grad=True."""
        ones = torch.ones_like(A1)
        dA3_dy = torch.autograd.grad(A3, y, grad_outputs=ones, create_graph=True)[0]
        dA2_dz = torch.autograd.grad(A2, z, grad_outputs=ones, create_graph=True)[0]
        dA1_dz = torch.autograd.grad(A1, z, grad_outputs=ones, create_graph=True)[0]
        dA3_dx = torch.autograd.grad(A3, x, grad_outputs=ones, create_graph=True)[0]
        dA2_dx = torch.autograd.grad(A2, x, grad_outputs=ones, create_graph=True)[0]
        dA1_dy = torch.autograd.grad(A1, y, grad_outputs=ones, create_graph=True)[0]

        u = dA3_dy - dA2_dz
        v = dA1_dz - dA3_dx
        w = dA2_dx - dA1_dy
        return u, v, w


if __name__ == "__main__":
    # quick shape self-test (run on the server where torch is installed)
    torch.manual_seed(0)
    model = GNOTOperator()
    B = 16
    x = torch.rand(B, 1, requires_grad=True)
    y = torch.rand(B, 1, requires_grad=True)
    z = torch.rand(B, 1, requires_grad=True)
    t = torch.rand(B, 1)
    V = torch.rand(B, NUM_WINDOWS) * 5.0
    N_people = torch.rand(B, 1) * 50.0

    A1, A2, A3, C, p = model(x, y, z, t, V, N_people)
    print(f"A1 {A1.shape}, A2 {A2.shape}, A3 {A3.shape}, C {C.shape}, p {p.shape}")

    u, v, w = model.velocity_from_potential(A1, A2, A3, x, y, z)
    print(f"u {u.shape}, v {v.shape}, w {w.shape}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {n_params:,}")
    print("\nGNOT forward + curl-trick self-test passed.")
