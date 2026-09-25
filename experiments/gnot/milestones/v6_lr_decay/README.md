# Milestone: v6_lr_decay (2026-09-25)

This folder is a frozen snapshot of the code used for the `v6_lr_decay`
checkpoints -- a full, fresh 20,000-iteration run testing whether adding a
cosine learning-rate decay schedule on top of everything confirmed in
`v5_closed_window_fix` (isotropic Fourier features + source_proximity,
source-concentrated sampling, closed/partial-closed scenario oversampling)
would fix the CO2 magnitude oscillation observed late in the v5 run.

**Do not edit the files in this folder.** They are copies for reference and
reproducibility. Ongoing development continues in `experiments/gnot/`
directly.

## What changed vs. v5

Only `train_gnot.py`: added `torch.optim.lr_scheduler.CosineAnnealingLR`,
decaying the Adam learning rate from `1e-3` down to `1e-5` over the full
20,000-iteration run (see the `LR_MIN` comment in the file for the literature
context -- this is a standard, generic deep-learning technique, not something
specific to PINNs or to this project's CO2-localization literature search).
`gnot_model.py` and `point_sampler.py` are unchanged from v5 (included here
only so this snapshot is fully self-contained).

## Result: the hypothesis was tested and did NOT hold up as hoped

**Important correction made during review:** the closed-window velocity
numbers below look "tighter" in v6 than in v5, but this is NOT evidence that
LR decay fixed a real problem. The v5 velocity numbers (0.026-0.097 m/s
across checkpoints) were already an order of magnitude below the pre-fix
baseline (0.58 m/s) and were already documented as "a real, durable fix" in
the v5 README -- that checkpoint-to-checkpoint variation was ordinary noise
around an already-fixed, already-low baseline, not a known bug. Framing v6's
slightly tighter range as "fixing velocity oscillation" would be retroactively
treating normal noise as a solved problem. There was nothing broken here for
LR decay to fix.

**Closed-window speed max (m/s), for reference only (not a fix, just noise
in an already-working baseline):**

| checkpoint | v5_closed_window_fix | v6_lr_decay |
|---|---|---|
| iter10000 | 0.026 | 0.045 |
| iter12000 | 0.096 | 0.039 |
| iter14000 | 0.055 | 0.026 |
| iter16000 | 0.097 | 0.030 |
| iter18000 | 0.045 | 0.032 |
| iter20000 | 0.065 | 0.038 |

**The actual target -- CO2 magnitude oscillation -- was NOT fixed.** v5's
CO2 grid-max climbed from ~0.002 (iter10000) to ~0.019 (iter18000) before
crashing back down ~4x at iter20000 (~0.005) -- real, if unstable, growth
toward the expected physical scale (~0.18, from dimensional analysis). In
contrast, v6's CO2 magnitude never grew at all: `C(source)` stayed pinned
in the +-0.001 range for the entire second half of training and its sign
kept flipping, rather than showing any trend.

**C(source) -- CO2 prediction at the true source location:**

| checkpoint | v5_closed_window_fix | v6_lr_decay |
|---|---|---|
| iter10000 | -0.0036 | -0.0011 |
| iter12000 | 0.0006 | 0.0012 |
| iter14000 | 0.0062 | -0.0001 |
| iter16000 | 0.0080 | 0.0001 |
| iter18000 | 0.0184 | 0.0001 |
| iter20000 | 0.0046 | 0.0001 |

v6's peak magnitude (0.0012) is roughly 15x smaller than v5's peak (0.0184).
Localization is also not better in v6: the CO2 peak repeatedly lands at a
domain corner (e.g. (0.10, 0.10)) rather than near the true source
(7.76, 4.58), and iter12000 shows the most extreme "band" artifact seen in
any version so far (40/40 and 37/40 grid rows/columns above half-max --
literally the entire grid in one direction).

## Interpretation

The most likely explanation: velocity and CO2 are two loss terms sharing
ONE optimizer and ONE learning-rate schedule, but they appear to converge on
very different timescales (velocity is comparatively easy and converges
early; CO2 is affected by spectral bias and a highly localized source term,
and needs more training time/larger steps, consistent with the literature
reviewed before this run -- see the research discussion earlier in this
project's history on spectral bias and localized-source PINN failure modes).
Decaying the LR to stabilize the easy term likely choked off the large
updates the hard term still needed, freezing CO2 at a near-zero, noise-level
magnitude instead of letting it grow (even unstably) toward the correct
scale.

This is a genuine, useful, reportable finding for the thesis: a single
global LR schedule is the wrong tool when multiple loss terms in the same
PINN have different convergence timescales. It points toward the
curriculum-learning / per-term-scheduling / causality-weighting family of
techniques discussed in the literature search as more promising next
directions, rather than further tuning of a single shared cosine schedule.

## What is still NOT SOLVED

CO2 source localization -- unchanged from v5. This remains the core open
problem for the thesis.

## Recommended next step (not yet attempted)

Investigate per-loss-term learning rate / schedule separation (e.g. two
optimizer param groups, or two independent schedules, one for the
shared trunk driving velocity/pressure and one for whatever drives the CO2
head), OR revisit the curriculum-learning / wavelet-based approaches
specific to localized-source PINN problems identified in the literature
search, rather than a single shared LR schedule.

## Reproducing this checkpoint

The checkpoints themselves live on the training server at
`checkpoints/v6_lr_decay/` (iter1000 through iter20000, plus `_final.pth`).
To copy the final checkpoint into this milestone folder for long-term
reference, run on the server:

```bash
cp ~/BuildingControlCFD/experiments/gnot/checkpoints/v6_lr_decay/gnot_v6_lr_decay_final.pth \
   ~/BuildingControlCFD/experiments/gnot/milestones/v6_lr_decay/
cd ~/BuildingControlCFD
git add experiments/gnot/milestones/v6_lr_decay/gnot_v6_lr_decay_final.pth
git commit -m "Add v6_lr_decay milestone checkpoint (final, iter20000)"
git push myfork main
```

To rerun the diagnostics against it:
```bash
cd experiments/gnot/milestones/v6_lr_decay
python3 closed_window_diagnostic.py gnot_v6_lr_decay_final.pth closed
python3 closed_window_diagnostic.py gnot_v6_lr_decay_final.pth open
python3 probe_source_co2.py gnot_v6_lr_decay_final.pth
```
(Run from inside this folder, not from `experiments/gnot/`, so these use the
frozen snapshot code rather than the live, evolving version.)
