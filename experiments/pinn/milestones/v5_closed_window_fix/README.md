# Milestone: v5_closed_window_fix (2026-09-25)

This folder is a frozen snapshot of the code that produced the `v5_closed_window_fix`
checkpoints, taken at the point where fix #3 (spurious closed-window velocity) was
confirmed working and CO2 source-localization was confirmed still unsolved. It exists
so this specific result is preserved and clearly documented before continuing further
work on the CO2 problem in the live `experiments/gnot/` files.

**Do not edit the files in this folder.** They are copies for reference and
reproducibility. Ongoing development continues in `experiments/gnot/` directly.

## What is CONFIRMED FIXED

**The spurious closed-window velocity artifact.** Previously, the model predicted
non-trivial velocity (up to ~0.3-0.6 m/s) even when all 8 windows were closed (V=0)
and there was no forcing at all -- physically, the correct answer for zero forcing,
zero initial condition, and no body-force term is exactly u=v=w=0 everywhere.

Root cause: `sample_scenario()` in `point_sampler.py` used to draw each of the 8
window velocities independently and uniformly. The probability of all 8 landing
near zero simultaneously (the exact scenario being tested) was astronomically small
(~1e-16 for a small tolerance) -- a curse-of-dimensionality problem. The network had
essentially never been trained on anything resembling the all-closed scenario.

Fix: `sample_scenario()` now deliberately injects correlated closed and
partially-closed window configurations into training (`CLOSED_SCENARIO_FRAC=0.3`,
`PARTIAL_CLOSED_SCENARIO_FRAC=0.2`), on top of the remaining 50% fully-uniform
sampling.

**Evidence** (closed-window diagnostic, `speed` = |velocity|, all windows closed):

| checkpoint | speed max (m/s) | speed mean (m/s) |
|---|---|---|
| pre-fix (v4b, before fix #3) | 0.58 | 0.075 |
| iter10000 (fix #3 applied) | 0.026 | 0.0056 |
| iter12000 | 0.096 | 0.0098 |
| iter14000 | 0.055 | 0.0066 |
| iter16000 | 0.097 | 0.0074 |
| iter18000 | 0.045 | 0.0047 |
| iter20000 | 0.065 | 0.0056 |

All roughly an order of magnitude smaller than pre-fix, consistently, across every
checkpoint tested. This is a real, durable fix.

**Open-window regime was checked for regression** (since fix #3 reduces the fraction
of purely-uniform-sampled scenarios from 100% to 50%): at iter20000, open-window
speed reaches up to 5.05 m/s with mean 0.425 -- correctly forced, physically
plausible, no sign of degradation from the closed-scenario oversampling.

## What is NOT YET SOLVED

**CO2 source localization.** The CO2 field still does not localize around the known
source (room center, breathing height). Across three layered fixes attempted
(isotropic random Fourier features + explicit `source_proximity` input feature;
source-concentrated interior sampling at `frac=0.4, std=sigma`; then a tuned,
stronger version at `frac=0.6, std=sigma/2`), the CO2 prediction:

- Never produces a real, spatially compact bump centered on the true source. The
  "peak" found by the diagnostic repeatedly lands near a window-wall edge instead,
  not the room center.
- Its overall magnitude oscillates rather than converging: it rose fairly steadily
  from checkpoint iter10000 through iter18000 (grid-max CO2 climbing from ~0.002 to
  ~0.019), then dropped back down by ~4x at iter20000 (~0.005) -- consistent with
  known late-training instability from using a constant learning rate with no decay
  schedule (the same issue flagged in earlier, pre-fix runs of this project).
- The correct physical scale (estimated via dimensional analysis:
  `(N_people * EMISSION_PER_PERSON) / (DIFFUSIVITY * CO2_SOURCE_SIGMA) ~= 0.18`) is
  still roughly 10-40x larger than anything observed.

**Direct source-location probe** (`probe_source_co2.py`, evaluates C exactly at the
true source coordinates, not just wherever the grid's argmax happens to be):

| checkpoint | C(source) | grid max | source as % of max |
|---|---|---|---|
| iter10000 | -0.0036 | -0.0032 | (both negative) |
| iter12000 | 0.0006 | 0.0022 | 28% |
| iter14000 | 0.0062 | 0.0070 | 88% |
| iter16000 | 0.0080 | 0.0104 | 77% |
| iter18000 | 0.0184 | 0.0188 | 98% (but field nearly flat, ~3% spread) |
| iter20000 | 0.0046 | 0.0048 | 97% (field went flat again, lower magnitude) |

Interpretation: the network appears to be learning the correct overall CO2 *scale*
(a coarse, low-frequency property) before learning the correct spatial *shape* (a
localized bump) -- consistent with the well-documented "spectral bias" of coordinate
networks (Rahaman et al. 2019). The oscillation in overall magnitude, rather than
smooth convergence, points at the missing LR decay schedule as the most likely next
fix, not a fundamentally wrong architecture.

## Recommended next step (not yet attempted)

Add a learning-rate decay schedule to `train_gnot.py`'s optimizer (e.g. cosine decay
or step decay over the training run) and retrain. This directly targets the observed
late-training oscillation, which is currently the main obstacle to the CO2 field
settling into a stable, correctly-shaped solution.

## Reproducing this checkpoint

The checkpoint itself (`gnot_v5_closed_window_fix_iter20000.pth`, ~1.3MB) lives on
the training server, not in this git-tracked folder (binary checkpoints aren't
committed as a matter of course). To copy it into this milestone folder for
long-term reference, run on the server:

```
cp ~/BuildingControlCFD/experiments/gnot/checkpoints/v5_closed_window_fix/gnot_v5_closed_window_fix_iter20000.pth \
   ~/BuildingControlCFD/experiments/gnot/milestones/v5_closed_window_fix/
cd ~/BuildingControlCFD
git add experiments/gnot/milestones/v5_closed_window_fix/gnot_v5_closed_window_fix_iter20000.pth
git commit -m "Add v5_closed_window_fix milestone checkpoint (iter20000)"
git push myfork main
```

To rerun the diagnostics against it:
```
cd experiments/gnot/milestones/v5_closed_window_fix
python3 closed_window_diagnostic.py gnot_v5_closed_window_fix_iter20000.pth closed
python3 closed_window_diagnostic.py gnot_v5_closed_window_fix_iter20000.pth open
python3 probe_source_co2.py gnot_v5_closed_window_fix_iter20000.pth
```
(Note: these scripts import from `gnot_model`/`point_sampler`/`train_gnot` in the
same directory via Python's local-import resolution, so run them from inside this
folder, not from `experiments/gnot/`, to make sure they use the frozen snapshot
code rather than the live, evolving version.)
