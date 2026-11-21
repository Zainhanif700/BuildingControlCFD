# CO₂ forecasting in a real classroom with a neural operator

This is the code of my Master's thesis. I take the method of Bian & Shi (2025),
*Data-driven operator learning for energy-efficient building control*
([arXiv:2504.21243](https://arxiv.org/abs/2504.21243)), and apply it to a real classroom at our
university: I simulate the airflow and CO₂ in the room, train a neural operator (GNOT) on the
results, and use it to forecast the CO₂ in the room a few minutes ahead. A second model sees only the
sensor stations of the room. All results are from simulation; there is no real sensor data yet.

The repository started as a fork of the authors' code. Their original code is still here (see
[Original paper code](#original-paper-code) at the end); my own work is in `experiments/`.

## The room

The classroom (room 1-130) is 15.53 × 9.16 × 3.15 m with 8 windows (W1–W8) on one wall, 2 doors on the
opposite wall and 4 free-standing columns. The geometry comes from STL files of the real room
(`experiments/pinn/geometry/`). Fresh air comes in through open windows and leaves through open doors
or outlet windows. Each person breathes out 6 L/min of air at 40,000 ppm CO₂ (the same values as in the
paper); the outdoor air has 400 ppm. The temperature is not simulated.

The data has 68 window settings (52 for training, 16 for testing):

- 28 settings (N01–N28): only W1, W3 and W6 can open (0.2–2.0 m/s), door 1 open, door 2 closed.
- 40 settings (A01–A40): all 8 windows and both doors can be open or closed; a window can let air in
  or out.

The scenario file is `experiments/gnot_openfoam_rans/scenarios_v4.txt` (one line per setting: name,
train/test, speeds at W1–W8, doors). A new setup only needs a new file in this format.

## Project structure

```
BuildingControlCFD/
├── experiments/
│   ├── gnot_openfoam_rans/   OpenFOAM cases, CO2 solver, forecast model and its training
│   ├── co2_sampling/         the final dataset: OpenFOAM with a save every second, CO2 build, server script
│   ├── sensors/              the model that sees only the sensor stations, warning tests, chain tests
│   └── pinn/                 training without data (physics-only network), not used in the end
├── learning/  simulation/  control/  local/  scripts/   original code of the paper
├── thesis/                   the LaTeX source of the thesis
├── Dockerfile                OpenFOAM v2412 + Python, for running the simulations in a container
└── requirements.txt
```

The data files and the trained networks are large and are not in the repository.

## Method in short

**Simulation.** For every setting I run OpenFOAM v2412 (`pimpleFoam`, k-ω SST with wall functions, the
same turbulence model as the paper) on a mesh of 856,284 cells of 8 cm. The airflow is saved every
second for the first 150 s (the real start-up flow) and every 10 s up to 600 s. A save every 10 s was
not enough: my CO₂ then differed from OpenFOAM's own CO₂ by up to 45 % in fast scenarios, with a save
every second the difference is about 2 %.

**CO₂ data.** The CO₂ for the full 30 minutes is computed on the GPU with my own finite-volume solver
(`fv_cons.py`): the real flow for the first 150 s, then the mean flow of 150–600 s, CO₂ saved every
30 s. After 150 s the mean flow keeps too much CO₂ in the room in fast scenarios; this is a known limit
that is described in the thesis.

**Paper-like data.** In the paper every run starts from the steady state of another setting. I do the
same without new OpenFOAM runs: a run starts from the CO₂ of a partner setting after 30 minutes, the
people are switched off and the new flow carries the CO₂ away. The CO₂ equation is linear, so the
number of people before and after the change can be chosen freely during training (10–80).

**Forecast model.** As in the paper: the CO₂ map at 1.6 m over the last 6 minutes (12 maps), the window
speeds, which windows and doors are open, and the number of people go in. The CO₂ map of the next
3 minutes comes out, with an uncertainty. Five networks are trained with different seeds and averaged.
The final model also gets the time since the last window change, and 90 % of its training samples
have a window change inside the 6-minute history (`--tau --pad-prob 0.9`, see below).

**Sensor model.** A second model (`experiments/sensors/`) gets the CO₂ history only at the 8 stations
(6 windows and both doors) and has to forecast the whole plane. I also tested what one more sensor in
the middle of the room would give, and a warning for 700 and 800 ppm.

## Results (simulation only)

Measured on the 16 test settings, which the model never sees in training. "Strict" counts only the CO₂
above the outdoor level, "paper" is the definition of the paper (with the 400 ppm added).

| | Strict | Paper |
|---|---|---|
| Reproduction of the paper (their code and data) | 29.8 % | 11.46 % (paper: 10.90 %) |
| Final forecast model, transition runs | 0.40 % | 0.14 % |
| "CO₂ stays as it is now" (persistence) | 1.77 % | 0.44 % |
| Sensor model (8 stations), random splits | about 32 % | about 5–6 % |

With a window change inside the 6-minute history the final model has a strict error of 21.3 %
(persistence 40.1 %); in the first minute after the change it is still weak (48 % against 100 %).
The warning of the sensor model finds about 87–91 % of the real cases with a margin of 30–40 ppm at
700 ppm (70–80 ppm at 800 ppm), with about 2–5 % of the other 30-second steps giving a false alarm.
Details, tables and limits are in the thesis (`thesis/Thesis.pdf`).

**Time** (measured on our computers): OpenFOAM for the first 150 s of one setting with a save every
second took about 2 h 15 min on 4 cores; a full 600 s run on one core takes 15–90 h depending on the air
speed. My CO₂ solver needs 1.25–4 h per setting on the GPU. Training the 5 networks with 40,000 steps
took about 3 h 12 min.

## How to run

Python packages are in `requirements.txt` (`pip install -r requirements.txt`, any environment such as a
venv). The scripts use the `python3` of the active environment; another one can be given with
`CFD_PY=/path/to/python`. OpenFOAM v2412 (openfoam.com) is installed separately: source its `etc/bashrc`
before the simulation scripts, or set `FOAM_BASHRC` to it (`experiments/gnot_openfoam_rans/README_batch.md`
has the details, also for a server, SLURM and Docker).

### 0. Tests (a few minutes, CPU)

```bash
cd experiments/gnot_openfoam_rans
python3 test_fv_cons.py        # CO2 solver
python3 test_forecast.py       # forecast code
cd ../sensors && python3 test_sensors.py
```

### 1. OpenFOAM, one setting (the current script)

```bash
cd experiments/co2_sampling
bash run_full_case.sh N01 4     # setting name, number of CPU cores
```

This runs 0–150 s with a save every second and then up to 600 s, writes
`cases/N01ws1_dx0.08/` and a file `done` at the end. It continues where it stopped if it is started
again. The older `run_batch.sh` saves the flow only every 10 s and should not be used for new data.
For several machines at once, `server_run.sh` installs OpenFOAM and Python in the home folder and runs a
list of settings: `bash server_run.sh setup`, then `bash server_run.sh run 4 A01 A02 ...`.

### 2. CO₂ dataset and transition files (GPU)

```bash
cd experiments/co2_sampling
python3 build_v4.py --name A01 --case cases/A01ws1_dx0.08 --out data_v4 \
    --scen ../gnot_openfoam_rans/scenarios_v4.txt --trans-out data_transitions_v4
```

`collect_v4.sh` does this for all finished cases (and copies them from other machines). The 28 N settings
were built earlier with `build_v3.py`, which uses the same method.

### 3. Training of the forecast model

```bash
cd experiments/gnot_openfoam_rans
# final transition model: window change in 90 % of the histories, time since the change as input, 40,000 steps
python3 train_forecast.py --data-dir data_transitions_v4 --transitions --scenarios scenarios_v4.txt \
    --tag v4_tr_tau09_long --z 1.6 --c-offset 400 --open-flags --pad-prob 0.9 --tau --iters 40000
# model for the runs that start with fresh air
python3 train_forecast.py --data-dir data_v4 --scenarios scenarios_v4.txt \
    --tag v4_dir_long --z 1.6 --c-offset 400 --open-flags --iters 40000
```

Each call trains 5 networks (about 38 min each on an RTX A2000) into `checkpoints/<tag>/` and prints the
errors at the end. Without `--tau --pad-prob` it is the model without window changes in the history
(`run_tau_final.sh` runs the two commands above in a row).

### 4. Window change inside the history

```bash
python3 eval_history_change.py --ckpt checkpoints/v4_tr_tau09_long --data-dir data_transitions_v4 \
    --scenarios scenarios_v4.txt                # error right after a window change; --which train for training settings
python3 oracle_mean_bound.py --data-dir data_transitions_v4 --scenarios scenarios_v4.txt   # bound of a mixing-box rule
```

### 5. Sensor model and warning

```bash
cd experiments/sensors
python3 train_sensor.py --data-dir ../gnot_openfoam_rans/data_transitions_v4 --transitions \
    --scenarios ../gnot_openfoam_rans/scenarios_v4.txt --tag my_run --noise 10
python3 diagnose_sensor.py --data-dir ../gnot_openfoam_rans/data_transitions_v4 --transitions \
    --scenarios ../gnot_openfoam_rans/scenarios_v4.txt --tag my_run         # errors, warning with margins
python3 warning_rules.py ...     # found cases and false alarms for other warning rules
python3 lecture_demo.py ...      # synthetic lecture (420 ppm start, students come and go)
python3 diagnose_position.py ... # how well a place in the room tells the seating-area average
```

`experiments/sensors/README.md` lists all files and the `run_*.sh` series of tests, and each script
says in its first lines how it is called.

## Approach that did not work: training without data (PINN)

The network never sees simulation data and learns only from the equations for airflow and CO₂ at
random points of the room. Over 25 versions the velocity error stayed at about 50 %, so this was not used
(`experiments/pinn/`; `milestones/` holds frozen snapshots and should not be edited).

## Next steps

- Test the forecast and the warning on real sensor data (the stations are being installed).
- Windows only tilted in winter (small openings) and the temperature are not covered by the data.
- Use the model to choose window openings that keep the CO₂ low with as little ventilation as
  possible, as in the paper.

## Original paper code

The folders `learning/`, `simulation/`, `control/`, `local/` and `scripts/` are the code of Bian &
Shi (2025). To reproduce their results I added evaluation and plotting scripts
(`learning/evaluate_ensemble.py`, `learning/evaluate_ensemble_strict.py`, `learning/visualize_prediction.py`,
`scripts/run_evaluation.sh`) and my own retrained 5-model ensemble (`learning/data/checkpoints/ensemble_5/`).
They need the authors' dataset (about 2.7 GB):

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
