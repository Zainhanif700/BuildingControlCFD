"""
The physics-only GNOT model: window/door/occupancy tokens, query encoder with Fourier features, cross-attention, velocity and CO2 outputs.
"""
import torch
import torch.nn as nn

from point_sampler import (
    NUM_WINDOWS, WINDOWS, DOORS, ROOM_X, ROOM_Y, ROOM_Z, CO2_SOURCE_SIGMA, BREATHING_HEIGHT,
    T_MAX, V_MAX, N_PEOPLE_MAX, C_REF, TAU_RAMP,
)
from throughflow import through_flow_potential, solid_distance_phi, alpha_potential, co2_window_factor

D_MODEL = 128

_WINDOW_AREAS = [(x1 - x0) * (z1 - z0) for x0, x1, z0, z1 in WINDOWS]
_A_DOORS = sum((x1 - x0) * (z1 - z0) for x0, x1, z0, z1 in DOORS)
P_SCALE_FLOOR = 0.5


def door_jet_speed(V):
    """Mean speed of the air leaving through the doors, (B, 1)."""
    areas = torch.tensor(_WINDOW_AREAS, device=V.device, dtype=V.dtype).view(1, -1)
    return (V * areas).sum(dim=1, keepdim=True) / _A_DOORS
N_HEADS = 4
N_LAYERS = 2

NONDIM_CHECKPOINT_KEY = "nondim"

MODEL_FORMAT_KEY = "model_format"
MODEL_FORMAT = "v24_fixedalpha"
LEARN_ALPHA = False
USE_CO2_WINDOW_FACTOR = False


def check_checkpoint_compat(ckpt, path=""):
    """Stops with a clear error if a checkpoint was trained with an older model version."""
    if ckpt.get(NONDIM_CHECKPOINT_KEY, False) and ckpt.get(MODEL_FORMAT_KEY) == MODEL_FORMAT:
        return
    raise RuntimeError(
        f"Checkpoint {path!r} (version={ckpt.get('version', '?')}, "
        f"model_format={ckpt.get(MODEL_FORMAT_KEY, 'none')}) was trained with an older model "
        f"(live format is {MODEL_FORMAT!r}); loading it here would give WRONG predictions. "
        f"Run the frozen scripts inside milestones/<that version>/ instead "
        f"(e.g. milestones/v8_nondim/, milestones/v9_zeroflow_bc/, milestones/v10_hardic/, "
        f"milestones/v13_fullocc/). From v16 on, every version is a git tag named like its VERSION: "
        f"git worktree add ../<version> <version> (e.g. git worktree add ../v19 v19_throughflow), "
        f"then run the scripts from that folder."
    )


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
    """Encodes one context token (window, door or occupancy)."""

    def __init__(self, d_model=D_MODEL, n_types=3):
        super().__init__()
        self.type_embed = nn.Embedding(n_types, d_model)
        self.proj = nn.Sequential(
            nn.Linear(3 + 1, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, pos, value, type_id):
        x = torch.cat([pos, value], dim=-1)
        return self.proj(x) + self.type_embed(type_id)


class FourierFeatures(nn.Module):
    """Fixed random Fourier features of (x, y, z) over several length scales."""

    def __init__(self, room_length, sigma, n_octaves=7, directions_per_octave=3, seed=0):
        super().__init__()
        f_min = 1.0 / (2.0 * room_length)
        f_max = 1.0 / (sigma / 4.0)
        n_octaves = max(2, n_octaves)
        log_f = torch.linspace(torch.log2(torch.tensor(f_min)), torch.log2(torch.tensor(f_max)), n_octaves)
        freqs = 2.0 ** log_f

        gen = torch.Generator().manual_seed(seed)
        directions = torch.randn(n_octaves, directions_per_octave, 3, generator=gen)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        freq_vectors = directions * freqs.view(n_octaves, 1, 1)
        self.register_buffer("freq_vectors", freq_vectors.reshape(-1, 3))
        self.n_octaves = n_octaves
        self.directions_per_octave = directions_per_octave
        self.n_freq = n_octaves * directions_per_octave

    def forward(self, coords):
        proj = 2 * torch.pi * (coords @ self.freq_vectors.T)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class QueryEncoder(nn.Module):
    """Encodes the query point (x, y, z, t) -> d_model vector."""

    _SOURCE_X = (ROOM_X[0] + ROOM_X[1]) / 2
    _SOURCE_Y = (ROOM_Y[0] + ROOM_Y[1]) / 2
    _SOURCE_Z = BREATHING_HEIGHT

    def __init__(self, d_model=D_MODEL, room_length=ROOM_X[1] - ROOM_X[0],
                 sigma=CO2_SOURCE_SIGMA, n_octaves=7):
        super().__init__()
        self.fourier = FourierFeatures(room_length=room_length, sigma=sigma, n_octaves=n_octaves)
        self._sigma2 = sigma ** 2
        fourier_dim = 2 * self.fourier.n_freq
        self.proj = nn.Sequential(
            nn.Linear(fourier_dim + 3, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )

    def forward(self, x, y, z, t):
        coords = torch.cat([x, y, z], dim=-1)
        feats = self.fourier(coords)
        dist_sq = (x - self._SOURCE_X) ** 2 + (y - self._SOURCE_Y) ** 2 + (z - self._SOURCE_Z) ** 2
        source_proximity = torch.exp(-dist_sq / self._sigma2)
        t_hat = t / T_MAX
        ramp = torch.tanh(3.0 * t / TAU_RAMP)
        return self.proj(torch.cat([feats, t_hat, ramp, source_proximity], dim=-1))


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
        attn_out, _ = self.attn(query, context, context)
        query = self.norm1(query + attn_out)
        query = self.norm2(query + self.ff(query))
        return query


class GNOTOperator(nn.Module):
    """Full operator: (x,y,z,t,V1..V8,N_people) -> (u,v,w,p,c) for the real room."""

    def __init__(self, d_model=D_MODEL, n_layers=N_LAYERS):
        super().__init__()
        self.token_encoder = TokenEncoder(d_model, n_types=3)
        self.query_encoder = QueryEncoder(d_model)
        self.blocks = nn.ModuleList([CrossAttnBlock(d_model) for _ in range(n_layers)])
        self.out_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 5)
        )
        self.alpha_head = nn.Sequential(nn.Linear(NUM_WINDOWS + 2, 32), nn.Tanh(), nn.Linear(32, 1))
        nn.init.zeros_(self.alpha_head[2].weight)
        nn.init.zeros_(self.alpha_head[2].bias)

        self.register_buffer("window_pos", torch.tensor(WINDOW_CENTERS, dtype=torch.float32))
        self.register_buffer("door_pos", torch.tensor(DOOR_CENTERS, dtype=torch.float32))
        self.register_buffer("room_center", torch.tensor([ROOM_CENTER], dtype=torch.float32))

    def _build_context(self, V, N_people):
        """Context tokens from the window speeds and the number of people."""
        B = V.shape[0]
        device = V.device

        win_pos = self.window_pos.unsqueeze(0).expand(B, -1, -1)
        win_val = V.unsqueeze(-1)
        win_type = torch.zeros(B, NUM_WINDOWS, dtype=torch.long, device=device)

        door_pos = self.door_pos.unsqueeze(0).expand(B, -1, -1)
        door_val = torch.zeros(B, door_pos.shape[1], 1, device=device)
        door_type = torch.ones(B, door_pos.shape[1], dtype=torch.long, device=device)

        occ_pos = self.room_center.unsqueeze(0).expand(B, -1, -1)
        occ_val = N_people.view(B, 1, 1)
        occ_type = torch.full((B, 1), 2, dtype=torch.long, device=device)

        pos = torch.cat([win_pos, door_pos, occ_pos], dim=1)
        val = torch.cat([win_val, door_val, occ_val], dim=1)
        type_id = torch.cat([win_type, door_type, occ_type], dim=1)

        lo = torch.tensor([ROOM_X[0], ROOM_Y[0], ROOM_Z[0]], device=device, dtype=pos.dtype)
        ext = torch.tensor([ROOM_X[1] - ROOM_X[0], ROOM_Y[1] - ROOM_Y[0], ROOM_Z[1] - ROOM_Z[0]],
                           device=device, dtype=pos.dtype)
        pos_hat = (pos - lo) / ext
        val_scale = torch.cat([
            torch.full((NUM_WINDOWS,), 1.0 / V_MAX, device=device, dtype=val.dtype),
            torch.ones(door_pos.shape[1], device=device, dtype=val.dtype),
            torch.full((1,), 1.0 / N_PEOPLE_MAX, device=device, dtype=val.dtype),
        ]).view(1, -1, 1)
        val_hat = val * val_scale
        return self.token_encoder(pos_hat, val_hat, type_id)

    def forward(self, x, y, z, t, V, N_people):
        """Model output (u, v, w, p, c); inputs (B, 1) except V (B, 8)."""
        context = self._build_context(V, torch.full_like(N_people, N_PEOPLE_MAX))
        q = self.query_encoder(x, y, z, t).unsqueeze(1)

        for block in self.blocks:
            q = block(q, context)

        out = self.out_head(q).squeeze(1)
        A1, A2, A3, C_hat, p = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4], out[:, 4:5]

        s = door_jet_speed(V)
        phi = solid_distance_phi(x, y, z, V) * torch.tanh(3.0 * t / TAU_RAMP)
        chi, psi = through_flow_potential(x, y, z, t, V, self.door_split(t, V))
        A1, A2, A3 = chi + s * phi * A1, s * phi * A2, psi + s * phi * A3
        p = torch.clamp(s, min=P_SCALE_FLOOR) ** 2 * p
        C = C_REF * (t / T_MAX) * (N_people / N_PEOPLE_MAX) * C_hat
        if USE_CO2_WINDOW_FACTOR:
            C = C * co2_window_factor(x, y, V)
        return A1, A2, A3, C, p

    def door_split(self, t, V):
        """Share of the air leaving through door 1."""
        feats = torch.cat([V / V_MAX, t / T_MAX, torch.tanh(3.0 * t / TAU_RAMP)], dim=-1)
        a0 = alpha_potential(V).clamp(1e-4, 1 - 1e-4)
        if not LEARN_ALPHA:
            return a0
        return torch.sigmoid(torch.log(a0 / (1.0 - a0)) + self.alpha_head(feats))

    def velocity_from_potential(self, A1, A2, A3, x, y, z):
        """Curl trick: differentiate the vector potential to get a divergence-free (u, v, w)."""
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
