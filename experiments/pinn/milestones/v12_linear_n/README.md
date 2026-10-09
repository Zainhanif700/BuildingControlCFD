# Milestone: v12_linear_n (2026-09-26)

Frozen copy of the code the `v12_linear_n` run trained with (git commit `0340b21`).
**Do not edit.** Run scripts from inside this folder, e.g.
`python3 validate_closed_room.py gnot_v12_linear_n_final.pth`.

## What v12 changed (vs v10_hardic)

**Exact proportionality of CO2 to occupancy:** C = C_REF*(t/T_MAX)*(N/N_MAX)*C_hat,
with the network body no longer seeing N (the occupancy token carries a constant).
This is exact physics for this model: the CO2 equation is linear, the source is
proportional to N, the IC/BCs are homogeneous, and the flow does not depend on N
(no buoyancy). A first attempt also fed N into the network, so C was N*C_hat(N),
which is not linear; the smoke test caught it before any training.

## Result: a mixed trade-off vs v10 (closed windows, `validate_closed_room.py`, 27 cases)

| final checkpoints | v10 | v12 |
|---|---|---|
| error at source (mean) | ~3-9% (worse at low N) | **1.9%** |
| plane L2, 5 people | 47-54% | **19-26%** |
| plane L2, 20 people | **14-17%** | 19-26% |
| plane L2, 50 people | **11-14%** | 19-26% |

- **Better:** v12 is the most accurate at the source, robust at every occupancy,
  and exactly consistent by construction (an empty room gives zero CO2; the flow
  does not depend on N). At iter 10000 it was far ahead of v10 (mean 24% vs 58%;
  20 people 24% vs 38%).
- **Worse:** plane error at 20-50 people. v12 improved only from 24% to 22%
  between iter 10000 and 20000, while v10 kept improving (20 people: 38% to 15%).
  guide_w = ||grad L_NS|| / ||grad L_CO2|| rose to 18-23 late in v12, against
  1-12 in v10.

## Diagnosed cause (research agent + algebra, checked against the code)

With C exactly proportional to N, every sample's CO2 residual is
r = (N/N_MAX) * r_hat, where r_hat does not depend on N. The squared CO2 loss
therefore weights each sample by (N/N_MAX)^2. With N ~ U[0, 50]:
- the mean weight is 1/3;
- the effective sample size is (1/3)^2 / (1/5) = 0.56 of nominal.

So **CO2 got about 3x less gradient weight than intended**, which is consistent
with guide_w ~20. v12 introduced this weighting itself; v10 did not have it.

A second, separate limitation holds in both versions: **the loss does not "see"
relative error in low-concentration regions.** With closed windows the error obeys
e_t - D lap(e) = r, e(0) = 0, and the diffusion length is only ~0.8 m in 120 s, so
the error accumulates locally from the residual. A tail residual of 1% of S_REF
gives ~20% error where the concentration is small, yet costs almost nothing in the
loss.

## Next steps (ranked by the research; each a separate, attributable run)

1. **v13:** evaluate all CO2 losses at N = N_MAX. This is exact because of the
   linearity, and gives every sample full weight. Single change.
2. Grad-norm loss balancing per Wang, Sankaran, Wang & Perdikaris 2023
   (arXiv:2308.08468): lambda_i = sum_j ||grad L_j|| / ||grad L_i||, moving average
   alpha = 0.9, updated every f = 1000 steps (jaxpi defaults, verified).
3. Tail-aware CO2 weighting or self-adaptive pointwise weights, aimed at the
   low-concentration error.
4. Lower priority: exponential LR decay (0.9 per 2k steps), residual-based adaptive
   sampling (Wu et al. 2023, k=1, c=1), an L-BFGS/NNCG finishing stage (Rathore et
   al. 2024), causal weighting.
