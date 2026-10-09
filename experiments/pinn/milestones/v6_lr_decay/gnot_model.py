# ============================================================================
# MILESTONE SNAPSHOT: v6_lr_decay (2026-09-25)
# Frozen copy -- DO NOT EDIT. See milestones/v6_lr_decay/README.md for what
# this milestone documents (cosine LR decay tested, closed-window velocity
# noise already OK before this change, CO2 magnitude suppressed not fixed).
# Ongoing development continues in the live files under experiments/gnot/.
# ============================================================================

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

from point_sampler import (
    NUM_WINDOWS, WINDOWS, DOORS, ROOM_X, ROOM_Y, ROOM_Z, CO2_SOURCE_SIGMA, BREATHING_HEIGHT,
)

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


class FourierFeatures(nn.Module):
    """Multi-octave, ISOTROPIC random Fourier feature positional encoding for
    (x,y,z), fixed (non-trainable), with the frequency BAND chosen from the
    ACTUAL physical length scales in this problem -- not an arbitrary default.

    FIX for a real problem found during verification: plain (x,y,z) fed
    straight into a Tanh MLP has a well-documented bias toward learning only
    smooth, low-frequency spatial patterns ("spectral bias" -- Rahaman et al.
    2019, arXiv:1806.08734). Our CO2 source is a SHARP, SMALL Gaussian bump
    (sigma=2.5m in a room ~15.5m long) -- exactly the kind of localized
    feature plain coordinate inputs struggle to represent.

    HISTORY: a first version used a single random Gaussian scale (scale=1.0,
    no justification) -- fixed by using a log-spaced BAND of octaves from the
    room's own scale down to a quarter of the CO2 source's width:
        f_min = 1 / (2 * ROOM_LENGTH)   -- resolves whole-room-scale variation
        f_max = 1 / (SIGMA / 4)         -- resolves a quarter of the CO2 bump's width
    A second version (kept that frequency band, but applied it SEPARABLY --
    each axis got its own independent set of sin/cos features at each octave,
    simply concatenated). That version trained but produced a CO2 "band"
    artifact: elevated across the room's full width instead of a compact
    bump. Investigated via literature research (see gnot_model.py's
    QueryEncoder for the full citation trail) -- the separable encoding used
    here was actually a simplification of Tancik et al. 2020's own method:
    their random Fourier features are drawn from an ISOTROPIC distribution
    over the full coordinate vector (not independently per axis), and the
    same isotropic convention is used in the closest literature we could find
    solving PDEs with interior point sources (Song, Wang & Alkhalifah 2022,
    GJI 232(3):1503, Fourier-feature PINN for seismic point-source
    wavefields). A separable per-axis encoding is not inherently equipped to
    represent a jointly-radial function of combined 3D distance (a sum of
    independent per-axis sinusoids doesn't easily approximate an isotropic
    bump); isotropic random Fourier features -- one random 3D direction
    vector per frequency, rather than 3 independent per-axis frequencies --
    fixes this directly since each single feature already responds to
    distance from any direction, not just along one axis.

    THIS VERSION: for each of n_octaves frequency magnitudes (same band as
    before), draws `directions_per_octave` random unit vectors in R^3 (fixed
    at construction via a seeded generator, non-trainable buffer) and scales
    each by that octave's frequency magnitude. `directions_per_octave=3` by
    default purely to keep the total feature count identical to the old
    separable version (3 axes x n_octaves -> now 3 directions x n_octaves),
    not because 3 is architecturally special once directions are isotropic.
    """

    def __init__(self, room_length, sigma, n_octaves=7, directions_per_octave=3, seed=0):
        super().__init__()
        f_min = 1.0 / (2.0 * room_length)
        f_max = 1.0 / (sigma / 4.0)
        # log-spaced frequency magnitudes from f_min to f_max, one octave per step
        n_octaves = max(2, n_octaves)
        log_f = torch.linspace(torch.log2(torch.tensor(f_min)), torch.log2(torch.tensor(f_max)), n_octaves)
        freqs = 2.0 ** log_f  # (n_octaves,)

        # Fixed seed (not a training hyperparameter) so the encoding is
        # reproducible across runs/checkpoints -- these directions must stay
        # IDENTICAL between training and any later inference/visualization,
        # since they're baked into what the downstream proj layer learned.
        gen = torch.Generator().manual_seed(seed)
        directions = torch.randn(n_octaves, directions_per_octave, 3, generator=gen)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        freq_vectors = directions * freqs.view(n_octaves, 1, 1)  # (n_octaves, K, 3)
        self.register_buffer("freq_vectors", freq_vectors.reshape(-1, 3))  # (n_octaves*K, 3)
        self.n_octaves = n_octaves
        self.directions_per_octave = directions_per_octave
        self.n_freq = n_octaves * directions_per_octave

    def forward(self, coords):
        # coords: (N, 3) -> isotropic random projection -> (N, n_freq) -> sin/cos
        proj = 2 * torch.pi * (coords @ self.freq_vectors.T)  # (N, n_freq)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, 2*n_freq)


class QueryEncoder(nn.Module):
    """Encodes the query point (x, y, z, t) -> d_model vector."""

    # Fixed, known CO2 source location (room center at breathing height) --
    # plain Python floats (not nn.Parameter/buffer): they broadcast fine
    # against any-device tensors in the subtraction below with no device
    # bookkeeping needed, and are never meant to be trainable.
    _SOURCE_X = (ROOM_X[0] + ROOM_X[1]) / 2
    _SOURCE_Y = (ROOM_Y[0] + ROOM_Y[1]) / 2
    _SOURCE_Z = BREATHING_HEIGHT

    def __init__(self, d_model=D_MODEL, room_length=ROOM_X[1] - ROOM_X[0],
                 sigma=CO2_SOURCE_SIGMA, n_octaves=7):
        super().__init__()
        self.fourier = FourierFeatures(room_length=room_length, sigma=sigma, n_octaves=n_octaves)
        self._sigma2 = sigma ** 2
        # Derived from the actual FourierFeatures instance (2 * n_freq), not
        # recomputed independently -- a hardcoded formula here previously
        # assumed 3 (one per axis); now that FourierFeatures uses isotropic
        # random directions instead, deriving it directly avoids a second
        # place this can silently drift out of sync.
        fourier_dim = 2 * self.fourier.n_freq
        # ADDITIONAL, COMPLEMENTARY fix on top of the isotropic Fourier feature
        # switch above (see FourierFeatures' own docstring for the primary
        # fix and its literature trail -- Rahaman et al. 2019, arXiv:
        # 1806.08734; Tancik et al. 2020, arXiv:2006.10739; Song, Wang &
        # Alkhalifah 2022, GJI 232(3):1503). The isotropic encoding fixes the
        # ENCODING's ability to represent a radial function; this feature
        # additionally gives the network a SHORTCUT to the exact known
        # location, since the source position doesn't need to be learned at
        # all -- it's a fixed constant. We checked directly and found no PINN
        # paper doing exactly this (an explicit distance/proximity-to-a-known-
        # INTERIOR-point input for source localization); the closest adjacent
        # precedent is Sukumar & Srivastava 2021 (arXiv:2104.08426), which
        # feeds distance functions from known points/boundaries as explicit
        # PINN inputs for exact BOUNDARY-condition enforcement, not an
        # interior source term. Treat this feature as a pragmatic addition
        # without direct literature precedent, not an established technique
        # -- disclose it as such in the thesis.
        #
        # This feature is exp(-dist_squared/sigma^2) -- the actual Gaussian
        # kernel value, NOT raw sqrt(distance). Two reasons: (1) physics_loss()
        # needs SECOND-order derivatives through this entire encoder (for the
        # Laplacian terms), and sqrt(dist^2) has a genuine curvature
        # singularity exactly at the source point (a cone tip) -- risky,
        # especially once source-concentrated sampling is added next, which
        # deliberately puts points near that exact singularity. The Gaussian
        # form is smooth (verified symbolically: all derivatives are
        # polynomial-in-1/sigma^2 times the same exponential, which is entire/
        # analytic everywhere) with no singularity at all. (2) it's also
        # better SCALED than raw (squared) distance: bounded in (0,1]
        # regardless of room size, vs. raw squared distance which could range
        # past 300 in this room -- and it directly matches the exact
        # functional form the source term already has.
        #
        # KNOWN LIMITATION (disclose in thesis, not fatal): this feature feeds
        # into the SHARED trunk that also produces velocity/pressure
        # (A1,A2,A3,p), via the same out_head. Nothing architecturally stops
        # the network from letting source_proximity leak into the velocity
        # prediction too, even though NS physics has no dependence on the CO2
        # source location -- a well-trained network should learn near-zero
        # weight from this feature into u,v,w,p, but that's not enforced. This
        # is a real, likely-low-risk tradeoff of the shared-trunk design.
        self.proj = nn.Sequential(
            nn.Linear(fourier_dim + 2, d_model),  # fourier(x,y,z) + raw t + source proximity
            nn.Tanh(),
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )

    def forward(self, x, y, z, t):
        coords = torch.cat([x, y, z], dim=-1)
        feats = self.fourier(coords)
        dist_sq = (x - self._SOURCE_X) ** 2 + (y - self._SOURCE_Y) ** 2 + (z - self._SOURCE_Z) ** 2
        source_proximity = torch.exp(-dist_sq / self._sigma2)
        return self.proj(torch.cat([feats, t, source_proximity], dim=-1))


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
