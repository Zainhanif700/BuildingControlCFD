# Milestone: v13_fullocc (2026-09-27), the current thesis reference model

Frozen copy of the code the `v13_fullocc` run trained with (git commit `a32a282`).
**Do not edit.** Run scripts from inside this folder, e.g.
`python3 validate_closed_room.py gnot_v13_fullocc_final.pth`.

## What v13 changed (vs v12_linear_n)

All CO2 losses (interior equation, windows, initial condition, CO2 wall/door
conditions) are evaluated at full occupancy N = N_MAX (`CO2_LOSS_AT_FULL_OCCUPANCY`).
Since v12 makes CO2 exactly proportional to N and the flow does not depend on N,
the residual at N_MAX is exactly the occupancy-independent residual. This removes
v12's hidden (N/N_MAX)^2 per-sample loss weighting, which had given CO2 only
about 1/3 of its intended weight. The trivial-solution reference is 0.279
(= 3 x 0.093), which independently confirms the 1/3 factor.

## Result (closed windows, `validate_closed_room.py`, 27 cases vs the grid-converged FD reference)

| plane relative L2 | v10 | v12 | **v13** |
|---|---|---|---|
| 5 people | 47-54% | 19-26% | **14-18%** |
| 20 people | 14-17% | 19-26% | **14-18%** |
| 50 people | **11-14%** | 19-26% | 14-18% |
| mean over all 27 cases | ~26% | 21.8% | **15.9%** |
| worst case | 54% | 26% | **17.9%** |
| source error (mean) | 3-9% | **1.9%** | 5.8% (consistently low) |

By height, v13's plane error is 15.4% at z=0.5 m, 14.7% at 1.10 m and 17.5% at 2.0 m,
identical across N by construction. During training, guide_w
(||grad L_NS|| / ||grad L_CO2||) stayed at 2-5, against about 20 late in v12,
which confirms the weighting diagnosis.

**Verdict:** the best all-round model. It is accurate at every occupancy with no
bad case, and has exact physics by construction (zero flow with closed windows,
C = 0 at t = 0, C proportional to occupancy, flow independent of occupancy).
v10 is still about 3 points better at 50 people, and v13 reads 5-7% low at the
source.

## Known remaining limitation

v10, v11 and v13 all settle at about 14-16% plane error. The loss barely penalizes
relative error in low-concentration regions: with closed windows the error obeys
e_t - D lap(e) = r with e(0) = 0, and a tail residual of 1% of S_REF gives about
20% error where the concentration is small, at almost no cost in the loss. Next
candidates: grad-norm loss balancing (Wang et al. 2023, arXiv:2308.08468) and
tail-aware CO2 weighting.

Other limitations, unchanged: only the closed-window case is quantitatively
validated (open windows need CFD reference data); c = 0 is applied at closed
windows (shown to be negligible, see milestones/v10_hardic/README.md).
