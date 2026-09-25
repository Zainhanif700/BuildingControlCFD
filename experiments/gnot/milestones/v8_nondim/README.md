# Milestone: v8_nondim (2026-09-25)

Frozen copy of the code the `v8_nondim` run trained with (git commit `2aa439e`),
plus the two diagnostics written afterwards to analyse it
(`probe_co2_time.py`, `co2_residual_breakdown.py`). **Do not edit.** Run the
scripts from inside this folder so they use this frozen model code.

## What v8 changed

Non-dimensionalization (Wang, Sankaran, Wang & Perdikaris 2023, "An Expert's
Guide to Training Physics-informed Neural Networks", arXiv:2308.08468, step 1):
t, V, N_people and token positions scaled to ~[0,1] inside the model; CO2
output as C_REF * C_hat; CO2 residual divided by S_REF. Plus a second time
input tanh(3t/TAU_RAMP) and the adaptive CO2 weight switched off (fixed 1.0).

Root cause it addressed: in every run v1-v6 the CO2 loss sat at ~3.08e-6 from
iteration ~10 onward -- exactly the loss of a constant (trivial) CO2 field. Raw
t (0-120 s) and N_people (0-50) saturated 75-91% of the first-layer tanh units,
and CO2 residuals were ~1e4x smaller than velocity residuals.

## What is CONFIRMED

- **First run ever to leave the trivial CO2 solution.** CO2(scaled) fell to
  ~0.01 by iter 2000, against a trivial floor of 0.0926.
- Gradient norms balanced without any adaptive weighting (guide_w ~1-3).
- The early "CO2 peak pinned at the door wall" (iter 1000-2000) resolved by
  itself by iter 5000 (peak at (8.35, 1.94)).

## What is NOT solved -- and why (diagnosed, not guessed)

C(source, t=60 s, closed windows, N=20) stayed ~0.012-0.021 vs the physically
expected ~0.138. `probe_co2_time.py` (iter 3000/5000) showed it is NOT an
offset problem (C at t=0 within +-0.03) but a GROWTH problem: CO2 at the
source grows only ~3% of the physical rate -- essentially frozen in time.

`co2_residual_breakdown.py` at iter 5000 showed why, per term of the CO2
equation (RMS in units of S_REF):

| scenario | loss/floor | dc/dt | u.grad(c) | D lap(c) | S | mean speed (m/s) |
|---|---|---|---|---|---|---|
| closed windows | 0.03 | 0.012 | **0.332** | 0.018 | 0.318 | 0.072 |
| all open | 0.11 | 0.022 | 0.216 | 0.009 | 0.290 | 0.238 |
| training mix | 0.05 | 0.036 | 0.267 | 0.013 | 0.269 | 0.157 |

**With every window closed, the network balances the CO2 source with
convection (0.332 ~ 0.318) instead of accumulation (0.012).** Physically, a
closed room has no forcing, so the true velocity is exactly zero and CO2 can
only accumulate. The network invented a slow (~0.07 m/s) spurious flow -- which
costs almost nothing in the Navier-Stokes loss (small viscosity, small
velocity) -- to "carry CO2 away". This loophole only became attractive once
v8's scaling made the CO2 term matter.

## Next step (v9)

Hard-constrain the velocity to be exactly zero when all windows are closed
(multiply the vector potential by s(V) = RMS(V)/V_MAX, which is 0 only when
every window is closed), plus the missing CO2 boundary conditions
(no-flux walls/columns, zero-gradient outflow at doors).

## Checkpoints

On the training server under `checkpoints/v8_nondim/`. To keep one here:
```
cp ~/BuildingControlCFD/experiments/gnot/checkpoints/v8_nondim/gnot_v8_nondim_iter5000.pth \
   ~/BuildingControlCFD/experiments/gnot/milestones/v8_nondim/
```
