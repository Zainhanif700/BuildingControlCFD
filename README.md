# CO₂ forecasting in a real classroom with a neural operator

This is the code of my Master's thesis. I take the method of Bian & Shi (2025),
*Data-driven operator learning for energy-efficient building control*
([arXiv:2504.21243](https://arxiv.org/abs/2504.21243)), and apply it to a real classroom at our
university: I simulate the airflow and CO₂ in the room, train a neural operator (GNOT) on the
results, and use it to forecast the CO₂ in the room a few minutes ahead.

The repository started as a fork of the authors' code. Their original code is still here (see
[Original paper code](#original-paper-code) at the end); my own work is in `experiments/`.

## The room

The classroom is 15.53 × 9.16 × 3.15 m with 8 windows on one wall, 2 doors on the opposite wall and
4 free-standing columns. The geometry comes from STL files of the real room
(`experiments/pinn/geometry/`). Fresh air comes in through the open windows and leaves through the
doors. The windows are grouped into three zones (windows 1–3, 4–6 and 7–8), and each zone is either
closed or open with an inflow speed between 0.2 and 2.0 m/s. The people sit in a seating area and
each person breathes out 6 L/min of air at 40,000 ppm CO₂ (the same values as in the paper); the
outdoor air has 400 ppm.

## Project structure

```
BuildingControlCFD/
├── experiments/
│   ├── gnot_openfoam_rans/   Approach 1: training with CFD data (main result)
│   └── pinn/                 Approach 2: training without data (physics-only network)
├── learning/  simulation/  control/  local/  scripts/   original code of the paper
├── images/                   figures of the original README
├── sweep_results/            results of an early parameter sweep
└── requirements.txt
```

**`experiments/gnot_openfoam_rans/`** – the complete pipeline from the OpenFOAM simulation to the
trained forecast model:

| File | What it does |
|---|---|
| `common.py` | all physical settings in one place (air, turbulence, CO₂ per person, seating area) |
| `scenarios.py`, `scenarios.txt` | the 40 window settings and the fixed train/test split |
| `make_rans_case.py`, `run_rans_case.sh` | write and run one OpenFOAM case (k-ω SST) |
| `run_one_rans.sh`, `run_dataset_rans.sh` | one window setting end to end / all settings in parallel |
| `extract_rans.py` | turns an OpenFOAM case into a dataset file and computes the 30-min CO₂ on the GPU |
| `fv_torch.py`, `fv_turb.py`, `fv_cons.py` | my finite-volume CO₂ solver (the dataset uses `fv_cons.py`) |
| `make_transitions.py` | paper-like runs that start from the CO₂ of another window setting |
| `forecast.py`, `train_forecast.py` | the forecast model and its training |
| `make_lc_scenarios.py`, `run_learning_curve.sh` | learning curve (training with fewer cases) |
| `plot_forecast.py` | figures of the predictions on the test cases |
| `test_*.py` | tests of the CO₂ solver and of the forecast code |
| `check_*.py`, `compare_*.py`, `scan_sct.py`, `steadiness_series.py` | the checks I used to validate the data |

**`experiments/pinn/`** – the physics-only approach: model (`gnot_model.py`), training points
(`point_sampler.py`), built-in base flow (`throughflow.py`), training (`train_gnot.py`), the checks
against OpenFOAM (`openfoam/`) and the room geometry (`geometry/`). Some parts of it (geometry, grid,
OpenFOAM case writer) are also used by approach 1. `milestones/` holds frozen snapshots of earlier
versions for traceability and should not be edited.

## Approach 1: training with CFD data

**Simulation.** For every window setting I run OpenFOAM v2412 (`pimpleFoam`, k-ω SST with wall
functions, the same turbulence model as the paper) on a mesh of 438,929 cells of 10 cm. A full
30-minute OpenFOAM run would take about two days per case, so I run 10 minutes: the airflow has
settled after about 2.5 minutes, and from then on I use its time average. The CO₂ for the full
30 minutes is then computed on the GPU with my own finite-volume solver on this flow. The solver
keeps the total amount of CO₂ exactly (divergence-free face velocities, flux-form advection with a
limiter, so the CO₂ never becomes negative). The CO₂ is saved every 30 seconds.

**Checks.** For every case my CO₂ was compared with the CO₂ that OpenFOAM computes itself on the
same run. After two fixes (mass conservation, and a turbulent Schmidt number of 0.3 to make up for
the mixing that the averaged flow leaves out) the difference is about 1–4 % in the paper's metric.
One case was also run for the full 30 minutes in OpenFOAM, and my CO₂ stayed within about 3 % the
whole time. The mesh passes `checkMesh`, y+ is 19–281, and the same case run twice gives identical
results.

**Paper-like data.** In the paper every run starts from the steady state of another setting. I do
the same without new OpenFOAM runs: for each case I start from the CO₂ of another case after
30 minutes and let the new flow carry it away. Because the CO₂ equation is linear, the numbers of
people before and after the change can be chosen freely during training (10–80, as in the paper).

**Forecast model.** As in the paper: the CO₂ map at 1.6 m height over the last 6 minutes (12 maps)
plus the window speeds and the number of people go in, and the CO₂ map for the next 3 minutes
comes out, with an uncertainty. Five networks (about 480k parameters each) are trained with
different seeds and averaged.

**Results.** On the 8 test settings, which the model never sees during training:

| | Model (5 networks) | "CO₂ stays the same" |
|---|---|---|
| Test error, CO₂ above the outdoor level | 0.11 % | 0.39 % |
| Test error, paper's metric (incl. 400 ppm) | 0.01 % | 0.05 % |
| Bian & Shi (2025), test, paper's metric | 10.9 % | – |

The model is about 4 times better than simply assuming that the CO₂ stays the same, and up to
12 times better on the hardest test case. The numbers are much smaller than the paper's, but they
cannot be compared directly: our open windows exchange the room air about 3–120 times per hour,
much more than the ceiling vents in the paper, so the CO₂ of the previous setting is gone within a
few minutes and the CO₂ changes less. The comparison with "CO₂ stays the same" is the fairer
measure.

A learning curve (training with 5, 10, 20 and 31 of the training settings) gives 0.70, 0.39, 0.30
and 0.27 times the "stays the same" error, so the curve levels off and the 39 simulations are
enough for this room.

**Time.** One OpenFOAM case takes about 13 hours on one CPU core (8–20 hours depending on the window
speed), the 30-minute CO₂ 0.5–2.5 hours on the GPU, the transition data about 22 hours for all
cases, and the training of the 5 networks about 1.5 hours.

## Approach 2: training without data (PINN)

Here the network never sees simulation data. It learns only from the equations for airflow and CO₂
at random points in the room, and some rules are built into it so that they always hold exactly
(no air through the walls, the right inflow at each window, start from rest, CO₂ proportional to
the number of people). I compared every version with an OpenFOAM solution of the same equations.
Over 25 versions I fixed several problems, but the velocity error stayed at about 50 %: the network
settles on a too-smooth flow that still fits the equations, a failure that is known from the
literature. One training run takes about 17 hours on the GPU. For our room this approach is not
accurate enough, so the thesis focuses on approach 1.

## How to run approach 1

I use two conda environments: `cfd` (Python with PyTorch and a GPU, see `requirements.txt`) and
`foam` (OpenFOAM v2412 from conda-forge). The scripts call
`~/anaconda3/envs/cfd/bin/python`; for another location set `CFD_PY` or edit the path at the top of
the shell scripts. All commands are run from `experiments/gnot_openfoam_rans/`.

```bash
cd experiments/gnot_openfoam_rans

# 0. tests (a few minutes)
python3 test_fv_cons.py
python3 test_forecast.py

# 1. dataset: OpenFOAM + CO2 for all 39 window settings, 4 in parallel (about 13 h per case)
bash run_dataset_rans.sh 4          # -> data/S*.npz, finished cases are skipped
python3 check_dataset.py            # quick check of all files

# 2. paper-like transition data (two in parallel on one GPU)
python3 make_transitions.py --shard 0/2 &
python3 make_transitions.py --shard 1/2 &
wait                                # -> data_transitions/S*.npz

# 3. training of the forecast model (5 networks, about 1.5 h)
python3 train_forecast.py --data-dir data_transitions --transitions --tag rans_tr \
    --members 5 --iters 20000 --z 1.6 --c-offset 400      # -> checkpoints/rans_tr/, errors at the end

# 4. figures for the test cases
python3 plot_forecast.py --tag rans_tr --data-dir data_transitions --transitions --z 1.6   # -> figures/

# 5. learning curve (optional, about 4.5 h)
bash run_learning_curve.sh
```

`scenarios.txt` is part of the repository, so the train/test split is fixed; `python3 scenarios.py`
writes the same file again. The dataset files keep the flow, so the CO₂ can be recomputed later (for
example with the real seating plan) without new OpenFOAM runs: `python3 recompute_all.py`.

## Next steps

- Prepare the model for the real sensors in the classroom: start from the CO₂ at the 8 sensor
  positions, with the window speeds from the wind sensors and the number of people from the
  distance sensors.
- Test the forecast on real sensor data.
- Use the model to choose window openings that keep the CO₂ low with as little ventilation as
  possible, as in the paper.
- Set the pipeline up for the university's batch system.

## Original paper code

The folders `learning/`, `simulation/`, `control/`, `local/` and `scripts/` are the code of Bian &
Shi (2025). To reproduce their results I added evaluation and plotting scripts
(`learning/evaluate_ensemble.py`, `learning/visualize_prediction.py`, `scripts/run_evaluation.sh`)
and my own retrained 5-model ensemble (`learning/data/checkpoints/ensemble_5/`). They need the authors' dataset
(about 2.7 GB):

```bash
mkdir -p learning/dataset
curl -L -o learning/dataset/train_data_norm.pkl \
  https://huggingface.co/datasets/alwaysbyx/Bear-CFD-dataset/resolve/main/processed_data/train_data_norm.pkl
curl -L -o learning/dataset/test_data_norm.pkl \
  https://huggingface.co/datasets/alwaysbyx/Bear-CFD-dataset/resolve/main/processed_data/test_data_norm.pkl
```

Then `scripts/run_evaluation.sh` evaluates the included 5-model ensemble (the paper reports 5.9 %
train and 10.9 % test error), `learning/train.py --seed <n>` trains a model,
`simulation/transient_simulation.py` runs their simulation and `control/control_optimization.ipynb`
the control strategy. More details are in the authors' repository
([github.com/alwaysbyx/BuildingControlCFD](https://github.com/alwaysbyx/BuildingControlCFD)).

## References

- Y. Bian, Y. Shi (2025). Data-driven operator learning for energy-efficient building control.
  arXiv:2504.21243.
- Z. Hao et al. (2023). GNOT: A general neural operator transformer for operator learning. ICML.

The dataset and code of the original paper are for research use only.

```bibtex
@article{bian2025data,
  title={Data-driven operator learning for energy-efficient building control},
  author={Bian, Yuexin and Shi, Yuanyuan},
  journal={arXiv preprint arXiv:2504.21243},
  year={2025}
}
```
