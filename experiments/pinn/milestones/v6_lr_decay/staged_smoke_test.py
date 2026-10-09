# ============================================================================
# MILESTONE SNAPSHOT: v6_lr_decay (2026-09-25)
# Frozen copy -- DO NOT EDIT. See milestones/v6_lr_decay/README.md for what
# this milestone documents (cosine LR decay tested, closed-window velocity
# noise already OK before this change, CO2 magnitude suppressed not fixed).
# Ongoing development continues in the live files under experiments/gnot/.
# ============================================================================

"""
Staged smoke test for GNOT -- runs each layer of the pipeline in isolation,
from lowest-level to full training step, printing PASS/FAIL after each stage
and STOPPING at the first failure.

WHY THIS EXISTS: our last few bugs (backward-graph reuse, GPU OOM, two
separate CO2-weighting collapses) were only caught by running the FULL
30-iteration integrated smoke test and reading through the printed losses.
That works, but when something breaks you only know "somewhere in the whole
pipeline something is wrong" -- you still have to manually narrow it down.
This script narrows it down FOR you: each stage tests one specific piece
(the Fourier encoding, the query encoder, the full model forward, the
curl-trick divergence-free property, each individual loss term, then finally
one full combined training step), in the same order data actually flows
through the model. If stage 3 fails, you know immediately the bug is in the
model forward pass, not e.g. in a loss term you haven't reached yet.

Run on the SERVER (needs torch + CUDA):
    cd experiments/gnot
    python3 staged_smoke_test.py

Each stage uses a TINY point count (16-64 points) purely for speed -- this is
about catching CRASHES / NaNs / shape bugs, not about training quality or
memory-ceiling behavior (the separate memory-sweep smoke test already covers
that, at the full POINTS_INTERIOR=1000 scale).
"""
import sys
import torch

# FIX (found by audit): PyTorch defaults to allow_tf32=True for float32 matmul
# on Ampere GPUs (e.g. the RTX A2000 this project trains on), which truncates
# matmul precision to ~10 mantissa bits. Chained through every Linear/attention
# matmul plus TWO rounds of second-order autograd (stage 4's divergence check),
# this can push a numerically-fine implementation's residual above a naive
# 1e-3 threshold -- a FALSE failure that looks like a broken curl trick but
# isn't. Disabling TF32 here trades a little speed for exact float32 semantics,
# which is what this script's tight numerical assertions actually assume.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

FAILED = False


def stage(name):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            global FAILED
            if FAILED:
                return None
            print(f"\n=== STAGE: {name} ===")
            try:
                result = fn(*args, **kwargs)
                print(f"PASS: {name}")
                return result
            except Exception as e:
                FAILED = True
                print(f"FAIL: {name}")
                print(f"  {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                return None
        return wrapper
    return decorator


def assert_finite(t, label):
    if torch.isnan(t).any():
        raise AssertionError(f"{label} contains NaN")
    if torch.isinf(t).any():
        raise AssertionError(f"{label} contains Inf")


@stage("0. sample_interior -- source-concentrated sampling: counts, bounds, no column overlap")
def test_sample_interior(device):
    from point_sampler import (
        sample_interior, SOURCE_X, SOURCE_Y, BREATHING_HEIGHT, CO2_SOURCE_SIGMA,
        SOURCE_SAMPLE_FRAC, ROOM_X, ROOM_Y, ROOM_Z, COLUMNS,
    )
    n = 1000
    x, y, z, t, V, N_people = sample_interior(n, device)
    for name, tensor in [("x", x), ("y", y), ("z", z), ("t", t), ("V", V), ("N_people", N_people)]:
        assert_finite(tensor, name)
    assert x.shape[0] == n, f"expected {n} points, got {x.shape[0]}"

    # every point must be within room bounds (the Gaussian branch clamps, but
    # verify no bug slipped a point outside)
    assert (x >= ROOM_X[0]).all() and (x <= ROOM_X[1]).all(), "x out of room bounds"
    assert (y >= ROOM_Y[0]).all() and (y <= ROOM_Y[1]).all(), "y out of room bounds"
    assert (z >= ROOM_Z[0]).all() and (z <= ROOM_Z[1]).all(), "z out of room bounds"

    # no point should land inside a column (rejection sampling should have caught all of them)
    xn = x.detach().cpu().numpy().ravel()
    yn = y.detach().cpu().numpy().ravel()
    for cx, cy, r, _, _ in COLUMNS:
        inside = (xn - cx) ** 2 + (yn - cy) ** 2 <= r ** 2
        assert not inside.any(), f"{inside.sum()} points landed inside column at ({cx},{cy})"

    # sanity-check the mixture actually concentrates points near the source:
    # with SOURCE_SAMPLE_FRAC of points drawn from a Gaussian near the
    # source, the fraction of ALL n points within 1-sigma-ish of the source
    # should now be noticeably higher than pure uniform sampling would give
    # (a rough, not exact, check -- this isn't testing an exact probability,
    # just that the mixture is doing SOMETHING, not silently falling back to
    # pure uniform sampling due to a bug).
    dist = torch.sqrt((x - SOURCE_X) ** 2 + (y - SOURCE_Y) ** 2 + (z - BREATHING_HEIGHT) ** 2)
    frac_near = (dist < CO2_SOURCE_SIGMA).float().mean().item()
    print(f"  n_source={int(round(n * SOURCE_SAMPLE_FRAC))}, n_uniform={n - int(round(n * SOURCE_SAMPLE_FRAC))}, "
          f"fraction of all {n} points within ~1 sigma of source: {frac_near:.3f}")
    # rough uniform-sampling baseline: a sphere of radius sigma over the room's
    # volume (ignoring z-clipping/column effects, just an order-of-magnitude check)
    room_volume = (ROOM_X[1] - ROOM_X[0]) * (ROOM_Y[1] - ROOM_Y[0]) * (ROOM_Z[1] - ROOM_Z[0])
    sphere_volume = (4.0 / 3.0) * torch.pi * CO2_SOURCE_SIGMA ** 3
    uniform_baseline = min(1.0, sphere_volume / room_volume)
    assert frac_near > uniform_baseline, (
        f"fraction near source ({frac_near:.3f}) is not higher than the pure-uniform baseline "
        f"({uniform_baseline:.3f}) -- source-concentrated sampling may not be working"
    )

    # sanity-check fix #3 (closed/partial-closed window-scenario oversampling):
    # verify a meaningful fraction of V rows are EXACTLY all-zero (the all-closed
    # scenario), not just occasionally-small from ordinary uniform sampling. The
    # old pure-uniform scheme would give ~1e-16 probability of landing on exact
    # all-zero, so seeing a near-CLOSED_SCENARIO_FRAC fraction here directly
    # confirms the fix is wired up, not silently bypassed.
    from point_sampler import CLOSED_SCENARIO_FRAC
    all_zero_frac = (V == 0).all(dim=1).float().mean().item()
    print(f"  fraction of {n} points with ALL windows exactly V=0: {all_zero_frac:.3f} "
          f"(expected close to CLOSED_SCENARIO_FRAC={CLOSED_SCENARIO_FRAC})")
    assert all_zero_frac > CLOSED_SCENARIO_FRAC * 0.5, (
        f"only {all_zero_frac:.3f} of points have all-zero V, expected close to "
        f"{CLOSED_SCENARIO_FRAC} -- closed-scenario oversampling may not be working"
    )


@stage("1. FourierFeatures -- shape + finiteness + 1st/2nd derivative w.r.t. raw coords")
def test_fourier_features(device):
    from gnot_model import FourierFeatures
    ff = FourierFeatures(room_length=15.53, sigma=2.5, n_octaves=7).to(device)
    n = 16
    coords = torch.randn(n, 3, device=device, requires_grad=True)
    out = ff(coords)
    expected_dim = 2 * ff.n_freq
    assert out.shape == (n, expected_dim), f"expected shape ({n},{expected_dim}), got {tuple(out.shape)}"
    assert_finite(out, "FourierFeatures output")

    # first + second derivative w.r.t. coords must exist and be finite
    loss = out.sum()
    grad1 = torch.autograd.grad(loss, coords, create_graph=True)[0]
    assert_finite(grad1, "1st derivative of FourierFeatures output")
    grad2 = torch.autograd.grad(grad1.sum(), coords, retain_graph=True)[0]
    assert_finite(grad2, "2nd derivative of FourierFeatures output")
    print(f"  n_freq={ff.n_freq}, output dim={out.shape[1]}, "
          f"grad1 max abs={grad1.abs().max().item():.4f}, grad2 max abs={grad2.abs().max().item():.4f}")
    return ff


@stage("2. QueryEncoder -- shape + source_proximity range + 1st/2nd derivatives")
def test_query_encoder(device):
    from gnot_model import QueryEncoder, D_MODEL
    qe = QueryEncoder().to(device)
    n = 16
    x = torch.randn(n, 1, device=device, requires_grad=True)
    y = torch.randn(n, 1, device=device, requires_grad=True)
    z = torch.randn(n, 1, device=device, requires_grad=True)
    t = torch.rand(n, 1, device=device, requires_grad=True) * 120.0
    out = qe(x, y, z, t)
    # FIX (found by audit): import D_MODEL instead of hardcoding 128, so this
    # doesn't false-fail if D_MODEL is ever changed in gnot_model.py.
    assert out.shape == (n, D_MODEL), f"expected ({n},{D_MODEL}), got {tuple(out.shape)}"
    assert_finite(out, "QueryEncoder output")

    # sanity-check source_proximity directly (recompute the same way QueryEncoder does)
    dist_sq = (x - qe._SOURCE_X) ** 2 + (y - qe._SOURCE_Y) ** 2 + (z - qe._SOURCE_Z) ** 2
    prox = torch.exp(-dist_sq / qe._sigma2)
    assert (prox > 0).all() and (prox <= 1.0).all(), "source_proximity out of expected (0,1] range"

    loss = out.sum()
    grad1 = torch.autograd.grad(loss, [x, y, z], create_graph=True)
    for g, name in zip(grad1, ["x", "y", "z"]):
        assert_finite(g, f"1st derivative w.r.t. {name}")
    grad2x = torch.autograd.grad(grad1[0].sum(), x, retain_graph=True)[0]
    assert_finite(grad2x, "2nd derivative w.r.t. x (this is exactly what physics_loss needs for the Laplacian)")
    print(f"  source_proximity range: [{prox.min().item():.4f}, {prox.max().item():.4f}]")
    return qe


@stage("3. GNOTOperator forward -- full model, tiny batch, all 5 outputs finite")
def test_model_forward(device):
    from gnot_model import GNOTOperator
    from point_sampler import NUM_WINDOWS
    model = GNOTOperator().to(device)
    n = 16
    x = torch.randn(n, 1, device=device, requires_grad=True)
    y = torch.randn(n, 1, device=device, requires_grad=True)
    z = torch.randn(n, 1, device=device, requires_grad=True)
    t = torch.rand(n, 1, device=device, requires_grad=True) * 120.0
    V = torch.rand(n, NUM_WINDOWS, device=device) * 5.0
    N_people = torch.rand(n, 1, device=device) * 50.0

    A1, A2, A3, C, p = model(x, y, z, t, V, N_people)
    for name, tensor in [("A1", A1), ("A2", A2), ("A3", A3), ("C", C), ("p", p)]:
        assert tensor.shape == (n, 1), f"{name} expected shape ({n},1), got {tuple(tensor.shape)}"
        assert_finite(tensor, name)
    print(f"  n_params={sum(p_.numel() for p_ in model.parameters()):,}")
    return model, (x, y, z, t, V, N_people, A1, A2, A3, C, p)


@stage("4. Curl trick -- velocity_from_potential produces a divergence-free field")
def test_curl_trick(device, model_and_inputs):
    model, (x, y, z, t, V, N_people, A1, A2, A3, C, p) = model_and_inputs
    u, v, w = model.velocity_from_potential(A1, A2, A3, x, y, z)
    for name, tensor in [("u", u), ("v", v), ("w", w)]:
        assert tensor.shape == (x.shape[0], 1), f"{name} wrong shape: {tuple(tensor.shape)}"
        assert_finite(tensor, name)

    du_dx = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
    dv_dy = torch.autograd.grad(v.sum(), y, create_graph=True, retain_graph=True)[0]
    dw_dz = torch.autograd.grad(w.sum(), z, create_graph=True, retain_graph=True)[0]
    divergence = du_dx + dv_dy + dw_dz
    max_div = divergence.abs().max().item()
    # should be ~0 up to float32 numerical error (curl of any vector potential
    # is exactly divergence-free analytically -- this checks the IMPLEMENTATION
    # matches the math, not just that it runs)
    assert max_div < 1e-3, f"divergence not near-zero: max|div|={max_div:.6f} (curl trick may be broken)"
    print(f"  max|div(u,v,w)| = {max_div:.2e} (should be ~1e-6 to 1e-4, float32 noise)")


@stage("5a. physics_loss (interior NS + CO2 residuals) -- finite, grads exist")
def test_physics_loss(device):
    from train_gnot import physics_loss
    from gnot_model import GNOTOperator
    model = GNOTOperator().to(device)
    ns_loss, co2_loss = physics_loss(model, device)
    assert_finite(ns_loss, "ns_loss")
    assert_finite(co2_loss, "co2_loss")
    params = list(model.parameters())
    grad_ns = torch.autograd.grad(ns_loss, params, retain_graph=True, allow_unused=True)
    grad_co2 = torch.autograd.grad(co2_loss, params, allow_unused=True)
    n_ns = sum(1 for g in grad_ns if g is not None)
    n_co2 = sum(1 for g in grad_co2 if g is not None)
    assert n_ns > 0, "ns_loss produced no gradients w.r.t. any parameter"
    assert n_co2 > 0, "co2_loss produced no gradients w.r.t. any parameter"
    print(f"  ns_loss={ns_loss.item():.6f}, co2_loss={co2_loss.item():.6f}, "
          f"params with grad: ns={n_ns}/{len(params)}, co2={n_co2}/{len(params)}")


@stage("5b. walls_loss / windows_loss / doors_loss / ic_loss -- each finite, no crash")
def test_boundary_losses(device):
    from train_gnot import walls_loss, windows_loss, doors_loss, ic_loss
    from gnot_model import GNOTOperator
    model = GNOTOperator().to(device)
    co2_weight = 1.0
    L_walls = walls_loss(model, device)
    assert_finite(L_walls, "walls_loss")
    L_windows = windows_loss(model, device, co2_weight)
    assert_finite(L_windows, "windows_loss")
    L_doors = doors_loss(model, device)
    assert_finite(L_doors, "doors_loss")
    L_ic = ic_loss(model, device, co2_weight)
    assert_finite(L_ic, "ic_loss")
    print(f"  walls={L_walls.item():.5f} windows={L_windows.item():.5f} "
          f"doors={L_doors.item():.5f} ic={L_ic.item():.5f}")


@stage("6. One full combined training step -- weights actually change, no crash")
def test_one_training_step(device):
    from train_gnot import (
        physics_loss, walls_loss, windows_loss, doors_loss, ic_loss,
        compute_param_grads, GRAD_CLIP_MAX_NORM, LR,
    )
    from gnot_model import GNOTOperator
    model = GNOTOperator().to(device)
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)
    co2_weight = 1.0

    # snapshot weights before the step
    before = [p.detach().clone() for p in params]

    optimizer.zero_grad()
    L_ns, L_co2 = physics_loss(model, device)
    grad_ns = compute_param_grads(L_ns, params, retain_graph=True)
    grad_co2 = compute_param_grads(L_co2, params, retain_graph=False)
    for p, g_ns, g_co2 in zip(params, grad_ns, grad_co2):
        total_grad = None
        if g_ns is not None:
            total_grad = g_ns
        if g_co2 is not None:
            weighted = co2_weight * g_co2
            total_grad = weighted if total_grad is None else total_grad + weighted
        if total_grad is None:
            continue
        p.grad = total_grad.clone() if p.grad is None else p.grad + total_grad

    walls_loss(model, device).backward()
    windows_loss(model, device, co2_weight).backward()
    doors_loss(model, device).backward()
    ic_loss(model, device, co2_weight).backward()

    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_MAX_NORM)
    optimizer.step()

    n_changed = sum(1 for b, p in zip(before, params) if not torch.equal(b, p.detach()))
    assert n_changed > 0, "optimizer.step() ran but NO parameters changed -- gradients may all be zero/None"
    for p in params:
        assert_finite(p.detach(), "a model parameter after optimizer.step()")
    print(f"  {n_changed}/{len(params)} parameter tensors changed after one step (expected: all of them)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: running on CPU -- fine for catching shape/NaN bugs, "
              "but won't tell you anything about GPU memory behavior.")

    test_sample_interior(device)
    test_fourier_features(device)
    test_query_encoder(device)
    model_and_inputs = test_model_forward(device)
    if model_and_inputs is not None:
        test_curl_trick(device, model_and_inputs)
    test_physics_loss(device)
    test_boundary_losses(device)
    test_one_training_step(device)

    print("\n" + "=" * 60)
    if FAILED:
        print("RESULT: at least one stage FAILED -- see above for exactly which one.")
        sys.exit(1)
    else:
        print("RESULT: ALL STAGES PASSED. Safe to proceed to the multi-iteration smoke test.")


if __name__ == "__main__":
    main()
