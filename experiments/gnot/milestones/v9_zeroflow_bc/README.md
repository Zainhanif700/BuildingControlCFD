# Milestone: v9_zeroflow_bc (2026-09-25)

Frozen copy of the code the `v9_zeroflow_bc` run trained with (git commit
`59ab017`). **Do not edit.** Run the scripts from inside this folder.
`v9_final_diagnostics.log` (written overnight by `run_after_v9.sh`) holds the
diagnostics of the finished run.

## What v9 changed (vs v8_nondim)

1. **Hard zero-flow constraint** (gnot_model.py): the velocity potential is
   multiplied by s(V) = RMS(V)/V_MAX, so the velocity is exactly zero when all
   windows are closed. This closes the loophole diagnosed in v8, where the network
   balanced the CO2 source with a spurious ~0.07 m/s flow instead of accumulation.
2. **The missing CO2 boundary conditions** (train_gnot.py `co2_boundary_loss`):
   no-flux dc/dn = 0 on walls, floor, ceiling and columns; zero-gradient outflow
   at the doors.

## Result: CO2 is physically right in shape and location for the first time

`co2_residual_breakdown.py` at iter 7000, closed windows (RMS in units of S_REF):

| | loss/floor | dc/dt | u.grad(c) | D lap(c) | S |
|---|---|---|---|---|---|
| v8 iter 5000 (loophole) | 0.03 | 0.012 | 0.332 | 0.018 | 0.318 |
| **v9 iter 7000** | **0.05** | **0.233** | **0.000** | 0.074 | 0.318 |

The source is now balanced by **accumulation** (dc/dt), which is the correct
physics for a closed room.

`probe_co2_time.py` at iter 7000 (closed windows, N=20, z=1.10 m):
- **Growth peak** C(60)-C(0) at (7.57, 4.92), against the true source at (7.76, 4.58).
- **Half-max width** 21/40 along y and 11/40 along x; the physical values are 18 and 10.
- **Growth at the source** is 0.099 at t=60 and 0.176 at t=120. Those are about
  81-82% of the physical values (about 0.121 and 0.216 once diffusion spreading is
  included; the probe's printed "expected" of 0.138/0.276 ignores diffusion).

## Remaining issue: constant offset

C at t=0 is about -0.039 everywhere, while the initial condition requires exactly 0.
The soft initial-condition penalty is too cheap: an offset of 0.04 costs about
0.003 in loss. Fix (v10): hard-enforce it with C = C_REF * (t/T_MAX) * C_hat.
