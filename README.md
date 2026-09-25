# BuildingControlCFD

This repository goes with the paper:
**Data-driven operator learning for energy-efficient building control**
[arXiv:2504.21243](https://arxiv.org/abs/2504.21243) -- Yuexin Bian, Yuanyuan Shi

The code has three parts: CFD airflow simulation, a neural operator model that learns from that simulation data, and a control strategy that uses the trained model.

<p align="center">
  <img src="images/framework.png" alt="Framework" width="600">
</p>
<p align="center"><em>Figure 1. Our Framework</em></p>

<p align="center">
  <img src="images/predict2.png" alt="1" width="600">
</p>
<p align="center"><em>Figure 2. Predicted indoor airflow distribution from the learned operator model.</em></p>

## Step 1: Clone and Install

```bash
Requires Python 3.10.

```bash
git clone https://github.com/Zainhanif700/BuildingControlCFD.git
cd BuildingControlCFD
pip install -r requirements.txt
```

`requirements.txt` has the exact package versions this project was built with, including `torch`/`dgl` for CUDA 11.8. If your machine has a different CUDA version, check https://github.com/HaoZhongkai/GNOT for how to install matching versions instead.

## Step 2: Download the Dataset

This code needs two data files that are too large to store on GitHub (about 2.7GB together). Download them and put them at these exact paths inside the folder you just cloned:

```
learning/dataset/train_data_norm.pkl   (2,100,505,182 bytes)
learning/dataset/test_data_norm.pkl    (607,608,040 bytes)
```

```bash
mkdir -p learning/dataset
curl -L -o learning/dataset/train_data_norm.pkl \
  https://huggingface.co/datasets/alwaysbyx/Bear-CFD-dataset/resolve/main/processed_data/train_data_norm.pkl
curl -L -o learning/dataset/test_data_norm.pkl \
  https://huggingface.co/datasets/alwaysbyx/Bear-CFD-dataset/resolve/main/processed_data/test_data_norm.pkl
```

After downloading, check the file sizes match the numbers above:
```bash
ls -l learning/dataset/
```
If a size doesn't match, the download didn't finish properly -- delete the file and try again.

The trained model checkpoints are already included in this repo (`local/models/`, `learning/data/checkpoints/ensemble_5/`), so you don't need to download those separately.

## Step 3: Check the Results

This step does not train anything -- it just loads our already-trained models and runs them against the test data, so it's quick.

```bash
./run_evaluation.sh        # Linux / macOS
run_evaluation.bat         # Windows
```

By default this checks our own retrained 5-model ensemble (`learning/data/checkpoints/ensemble_5/`) and prints each model's error plus the ensemble's error, on both the training and test sets. The paper reports 5.9% (train) and 10.90% (test) for the ensemble (Table 3), so you can compare directly.

To check the paper authors' own checkpoints instead:
```bash
./run_evaluation.sh ../local/models/*.pt
```

Both commands are just a shortcut for `python learning/evaluate_ensemble.py <ckpt1.pt> ... <ckpt5.pt>`, which you can also run directly with any set of checkpoint files.

## Optional: Train From Scratch

Training builds a new model from the data instead of using the checkpoints already in this repo. The paper reports this takes about 16 GPU-hours in total for all 5 models (on 2x RTX 2080 Ti) -- your hardware may be faster or slower.

This trains one model per run. We used seeds 2023, 2024, 2025, 2026, and 2027 for our 5-model ensemble:

```bash
cd learning
python train.py --seed 2023
python train.py --seed 2024
python train.py --seed 2025
python train.py --seed 2026
python train.py --seed 2027
```

Then put the 5 resulting checkpoint files together in one folder.

**Important:** your new checkpoints are saved as new files in `learning/data/checkpoints/`, separate from the `ensemble_5/` folder already in this repo. The commands above and below default to `ensemble_5/`, which is our result, not yours. To check or visualize your own trained model instead of ours, point the commands at your new files (run from the repo root, then `cd learning` for the last one):

```bash
ls -t learning/data/checkpoints/*.pt | head -5
./run_evaluation.sh data/checkpoints/<your_files>.pt
cd learning
python visualize_prediction.py data/checkpoints/<your_file>.pt --out my_prediction.png
```

## Optional: Visualize a Prediction (Uses Our Model by Default)

This produces one image showing the true CO2 levels next to the model's predicted CO2 levels for one test sample, along with the error between them -- similar to Figure 5 in the paper. By default it uses our included model (`ensemble_5`), not a model you trained yourself -- see the note above if you trained your own and want to see that instead.

```bash
cd learning
python visualize_prediction.py data/checkpoints/ensemble_5/*.pt --out prediction.png
```

## Optional: Run the CFD Simulation

We already provide simulated data for seeds 0 to 300. You only need this step if you want more data:
```bash
python simulation/transient_simulation.py --seed 0
```

## Optional: Control Strategy

Uses the trained model to plan ventilation control, and lets you look at the results:
```bash
control/control_optimization.ipynb
control/visualize.ipynb
```

## Thesis Extension: GNOT for a Real Classroom (`experiments/gnot/`)

This part of the repo is a Master's thesis extension of the paper above, adapting the GNOT-based neural operator to a real classroom instead of the paper's original domain. It is a **physics-only PINN** trained purely against the Navier-Stokes and CO2 advection-diffusion equations at randomly sampled points -- no CFD simulation data is used at any point.

Key differences from `learning/` (the original paper code): the model uses cross-attention over heterogeneous tokens (windows, doors, occupancy), a vector-potential/curl trick to guarantee divergence-free velocity, and trains against the real room's geometry (`experiments/gnot/geometry/*.stl`: 2 doors, 8 windows, 4 columns) instead of the paper's CFD dataset.

### Layout

- `gnot_model.py`, `point_sampler.py`, `train_gnot.py` -- the model, the physics-domain sampler, and the training loop. These three are the source of truth; every script below imports from them.
- `staged_smoke_test.py` -- an 8-stage isolated component test (Fourier encoding -> query encoder -> full forward pass -> divergence-free check -> each loss term -> one combined training step). Run this before any full training run to catch bugs early:
  ```bash
  cd experiments/gnot
  python3 staged_smoke_test.py
  ```
- `closed_window_diagnostic.py <checkpoint> [closed|open]` -- evaluates a trained checkpoint on a 40x40 grid with all windows closed (true solution: zero velocity) or open, to check for known failure modes.
- `probe_source_co2.py <checkpoint>` -- evaluates predicted CO2 directly at the true source location, to distinguish real convergence from a coincidental peak elsewhere in the domain.
- `visualize_gnot.py`, `visualize_gnot_slice.py`, `export_grid_for_viewer.py`, `sensor_mapper.py` -- visualization and export utilities.
- `geometry/` -- the room's STL files and the scripts that derive physical constants (room bounds, window/door positions, column radius) directly from them.
- `checkpoints/` -- trained model weights, organized by version tag.
- `milestones/` -- frozen, documented snapshots of the code + a checkpoint at specific points in development, kept for thesis traceability. Each has its own README stating exactly what was confirmed fixed vs. still open at that point. **Do not edit files inside `milestones/`** -- ongoing work continues in the live files above.

### Status

See `milestones/v5_closed_window_fix/README.md` for the most recent documented checkpoint: the spurious closed-window velocity artifact is confirmed fixed; CO2 source localization is still an open problem, currently suspected to need a learning-rate decay schedule.

## Dataset Details

The full dataset (including raw simulation data, not just the two files from Step 2) is on [Hugging Face](https://huggingface.co/datasets/alwaysbyx/Bear-CFD-dataset).

- Simulation tool: ANSYS FLUENT 2023R2
- Data types: steady-state and transient flow simulations
- Domain: indoor airflow and CO2 concentration in ventilated buildings

## License and Citation

The dataset and code are for research use only.

```bibtex
@article{bian2025data,
  title={Data-driven operator learning for energy-efficient building control},
  author={Bian, Yuexin and Shi, Yuanyuan Shi},
  journal={arXiv preprint arXiv:2504.21243},
  year={2025}
}
```

## Contact

Original authors: [Email](yubian@ucsd.edu)
