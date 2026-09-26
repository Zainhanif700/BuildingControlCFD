# Milestone: v10_hardic (2026-09-26): first quantitatively validated model

Frozen copy of the code the `v10_hardic` run trained with (model/training code
from git commit `6529b15`; `fd_reference_closed_room.py` from `a0b2992`).
**Do not edit.** Run the scripts from inside this folder, e.g.
`python3 fd_reference_closed_room.py --ckpt gnot_v10_hardic_final.pth`.

## The path to this model (v8 -> v10), each step diagnosed rather than guessed

| version | change | what it fixed / revealed |
|---|---|---|
| v8_nondim | non-dimensionalized inputs, outputs and the CO2 residual (Wang et al. 2023 Expert's Guide, step 1) | First run to leave the trivial CO2 solution (every run v1-v6 sat exactly on it). Revealed a loophole: with windows closed, the CO2 source was balanced by a spurious ~0.07 m/s flow (convection) instead of accumulation. |
| v9_zeroflow_bc | hard zero-flow constraint for closed windows; CO2 no-flux/outflow boundary conditions | Source now balanced by accumulation; shape and location correct. Remaining: a -0.039 offset at t=0 (mostly gone by iter 20000), and one training collapse at iter ~15000 that recovered. |
| **v10_hardic** | hard initial condition C = C_REF*(t/T_MAX)*C_hat (Lagaris et al. 1998) | C(t=0) = 0 exactly; best residuals of any run; **no collapse**. |

## Validation against an independent finite-difference reference

`fd_reference_closed_room.py` solves the closed-window problem exactly as a
diffusion equation. It uses a finite-volume grid of the real room with the 4
columns and no-flux walls, and explicit time stepping. With windows closed the
true velocity is zero, so this is the exact physics. **Grid-converged:** the
dx = 0.1 m (155x92x31) and dx = 0.05 m (311x183x63) grids agree to all 4 printed
decimals at every probe point and time.

CO2 at the source (7.76, 4.58, 1.10 m), all windows closed, N = 20:

| t | FD reference | v10 PINN | error |
|---|---|---|---|
| 10 s | 0.0225 | 0.0213 | -5% |
| 30 s | 0.0644 | 0.0621 | -4% |
| 60 s | 0.1214 | 0.1189 | -2% |
| 90 s | 0.1727 | 0.1701 | -1.5% |
| 120 s | 0.2197 | 0.2160 | -1.7% |

(v9 final gave 0.137 at t = 60 s, 13% high. v10 is the more accurate model.)

**Peak location at t = 60 s:** FD (7.57, 4.69), v10 (7.57, 4.69), identical on the
40x40 grid.

**Half-max width:** FD 18/40 along y and 12/40 along x; v10 18/40 and 10/40.

**Whole breathing-height plane: relative L2 error 14.0% (t=30), 13.9% (t=60),
14.7% (t=120).** Max absolute error at t = 60 s: 0.0115 (field max 0.1205).

**Where the error is:** in the tails, not the peak. The PINN blob is slightly too
narrow in x: 2.5 m east of the source it reads 0.043 vs 0.049 at t = 60 s (-11%).
Far from the source it drifts slightly positive: the far corner reads 0.0055 at
t = 60 s where it should be about 0.

Residual breakdown (final checkpoint, loss/floor; lower is better):
- closed windows: **0.01**, balanced by accumulation (dc/dt 0.262) plus diffusion (0.064) vs S 0.318; u.grad(c) = 0
- open windows: 0.08
- training mix: 0.02

## Known limitations (state these in the thesis)

1. **Only the closed-window case is quantitatively validated.** It is the case with
   an exact simple reference. Open-window cases need real CFD reference data.
2. **c = 0 is enforced at the windows even when they are closed** (windows_loss). A
   closed window is physically a wall, so this is a modelling inconsistency. The FD
   reference shows its effect on these results is negligible (the no-flux and c=0
   variants are identical to 4 decimals; over 120 s CO2 diffuses about 0.8 m, while
   the window wall is 4.6 m from the source). A correct future fix is the Danckwerts
   inflow condition (Danckwerts 1953), which reduces to c = 0 for strong inflow and
   to no-flux for a closed window.
3. **Training stability:** v9 collapsed once (iter ~15000) with a constant learning
   rate and recovered; v10 did not. One clean run is not proof, so a repeat with a
   different seed would confirm it.

## Robustness across occupancy, height and time (`validate_closed_room.py`)

These are 27 closed-window cases, all from one trained model with no retraining,
which demonstrates the operator property. The reference is one FD solve at N = 1
scaled by N: this is exact because the closed-room CO2 problem is linear in the
source, and the source is proportional to N.

| occupancy | error at source (range over 3 heights x 3 times) | plane relative L2 |
|---|---|---|
| 50 people | -0.5% to -7.5% | 11-14% |
| 20 people | -1.3% to -4.9% | 14-17% |
| 5 people | -6.1% to -12.8% | **47-54%** |

Heights 0.5 m, 1.10 m and 2.0 m behave alike. **At low occupancy the relative error
explodes.** The true field is exactly proportional to N, so a correct model's
relative error would not depend on N. This model has an error component that does
NOT scale with N (the far-field drift of about +0.005), which dominates when the
signal is small. Fix (v12): build the exact N-proportionality into the output.

## Follow-up tried: v11_latedecay, a negative result

Resumed this model at iter 20000, with its Adam state, and cosine-decayed the LR from
1e-3 to 1e-5 over 10,000 more iterations (code: git `3cb59d8`). The relative L2 error
against the FD reference at t = 60 s did not improve:

| checkpoint | plane L2 error |
|---|---|
| v10 final (this model) | **13.9%** |
| v11 iter 22000 | 18.1% |
| v11 iter 25000 | 15.8% |
| v11 iter 28000 | 15.2% |
| v11 final (iter 30000) | 15.5% |

The decay settled the error at about 15% rather than lowering it. Checkpoints of one
run differ by a few percent, so this reads as "no measurable change". **The ~14-15%
error in the low-concentration tails is systematic, not optimizer noise.** The network
sits at the minimum of the loss we give it, and a small residual at the sampled points
does not pin down the field where residuals are cheap: the tails, where the source is
tiny and only 40% of points are sampled uniformly. **This v10 model remains the best.**

Possible future refinement for the tails (not attempted): residual-based adaptive
sampling (e.g. Wu et al. 2023, a comparison of non-adaptive and residual-based
adaptive sampling for PINNs).
