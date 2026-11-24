<h1 align="center">CO₂ forecasting in a real classroom</h1>

<p align="center"><em>with a neural operator (GNOT), OpenFOAM simulations and sensor stations</em></p>

<p align="center">
  Version <b>v2026-11-24c</b> &nbsp;·&nbsp; updated <b>24 November 2026</b> &nbsp;·&nbsp; <a href="CHANGELOG.md">changelog</a> &nbsp;·&nbsp; <a href="thesis/Thesis.pdf">thesis (PDF)</a>
</p>

<p align="center">
  <a href="#1-start-here">Start here</a> ·
  <a href="#2-what-is-where">What is where</a> ·
  <a href="#3-the-room-and-the-data">Room and data</a> ·
  <a href="#4-method-in-short">Method</a> ·
  <a href="#5-results">Results</a> ·
  <a href="#6-how-to-run-levels-2-and-3">How to run</a> ·
  <a href="#8-original-paper-code-and-references">References</a>
</p>

Code of my Master's thesis. I take the method of Bian & Shi (2025), *Data-driven operator learning for energy-efficient
building control* ([arXiv:2504.21243](https://arxiv.org/abs/2504.21243)), and apply it to a real classroom of our university:
I simulate the airflow and the CO₂ in the room, train a neural operator (GNOT) on the results and forecast the CO₂ of the
next 3 minutes from the last 6 minutes. A second model sees only the sensor stations of the room.

---

## 1. Start here

### Which version do I have?

Versions are named by their date, `vYYYY-MM-DD` (a second version on the same day gets a letter). Every change is listed in
[`CHANGELOG.md`](CHANGELOG.md), and a script that I changed has an "Updated" line with the date in its first lines. If your
copy is older than the top entry, download it again.

| Version | Date | Main change |
|---|---|---|
| **v2026-11-24c** | 24 Nov 2026 | one-line setup comment in each scenario file; `compare_datasets.py` moved to `runs/` |
| v2026-11-24b | 24 Nov 2026 | code cleanup: 18 obsolete scripts removed, experiment runners moved into `runs/` folders |
| v2026-11-24 | 24 Nov 2026 | README redesigned; learning-curve script for the final setup |
| v2026-11-23c | 23 Nov 2026 | simpler READMEs for the batch scripts, the dataset and the sensor code |
| v2026-11-23b | 23 Nov 2026 | README shortened, "Start here" with the three levels |
| v2026-11-23 | 23 Nov 2026 | demo (`run_demo.sh`), `TRACE=1` option, changelog |

### What do you want to run?

Three levels, from small to big:

| Level | What you do | You need | Time |
|---|---|---|---|
| **1. Demo** | The final model forecasts one test setting (nothing is trained); OpenFOAM runs one setting for 20 s | Python packages; OpenFOAM v2412 for the OpenFOAM part | OpenFOAM part about 10 min on 4 cores (measured on our lab PC); the forecast is short |
| **2. Model only** | Train the final model again from the CO₂ data of the 68 settings and test it (no OpenFOAM) | a GPU; the data `data_transitions_v4` (about 1 GB, I send it) | training about 3 h 20 min on our GPU (estimated from file times) |
| **3. Full re-run** | All 68 OpenFOAM simulations, then the CO₂ on the GPU, then the training | many CPU cores and a GPU | several days of OpenFOAM and more days for the CO₂ (rough estimates, not measured on your machines) |

### Level 1 in three commands

The model and one test setting (N14) are in `experiments/gnot_openfoam_rans/demo/`. Once, in
`experiments/gnot_openfoam_rans/`: `bash setup_env.sh` (it should print `python OK`, and `OpenFOAM OK` if OpenFOAM is found).
Then:

```bash
bash run_demo.sh forecast   # forecast with the final model; prints the error against "the CO2 stays as it is now"; saves demo_forecast.png
bash run_demo.sh openfoam   # OpenFOAM, setting N01, 20 s of flow, 4 cores (NP=8 for more)
bash run_demo.sh            # both; they are independent, the 20 s run does not feed the model
```

> [!TIP]
> If a run stops without a message, put `TRACE=1` in front of the command. The log then shows every command with the time
> and the line.

**Level 2:** put `data_transitions_v4` into `experiments/gnot_openfoam_rans/` and run steps 3 and 4 of section 6.
**Level 3:** run steps 1 to 3 of section 6 for all settings of `scenarios_v4.txt`.

---

## 2. What is where

```
experiments/gnot_openfoam_rans/   OpenFOAM cases, CO2 solver, forecast model, its training, the demo
experiments/co2_sampling/         the final dataset: OpenFOAM with a save every second, CO2 build, server script
experiments/sensors/              the sensor model, warning tests, forecast from a sensor file (see its README.md)
experiments/*/runs/               the experiment runners that made the thesis results (training series, mesh study, tests)
experiments/pinn/                 training without data (physics only): tried, not used
experiments/gpu_lbm/              GPU lattice-Boltzmann solver: tried, not used
experiments/openfoam_versions/    comparison of OpenFOAM versions
thesis/                           LaTeX source of the thesis
Dockerfile, requirements.txt      OpenFOAM v2412 + Python in a container; Python packages
learning/ simulation/ control/ local/ scripts/   original code of the paper (see learning/README.md)
```

Data files and trained networks are large and not in the repository (except the small `demo/` folder).

---

## 3. The room and the data

Room 1-130: 15.53 × 9.16 × 3.15 m, 8 windows (W1–W8) on one wall, 2 doors on the opposite wall, 4 columns (geometry from
STL files in `experiments/pinn/geometry/`). Fresh air comes in through open windows and leaves through open doors or outlet
windows. A person breathes out 6 L/min at 40,000 ppm CO₂ (as in the paper); the outdoor air has 400 ppm.

The data has **68 window settings** (52 for training, 16 for testing), in `experiments/gnot_openfoam_rans/scenarios_v4.txt`
(one line per setting: name, train/test, speeds at W1–W8, doors):

| Settings | What is open |
|---|---|
| N01–N28 | only W1, W3 and W6 can open (0.2–2.0 m/s), door 1 open, door 2 closed |
| A01–A40 | all 8 windows and both doors can be open or closed; a window can let air in or out |

---

## 4. Method in short

| Step | What happens |
|---|---|
| **Airflow** | OpenFOAM v2412, `pimpleFoam`, k-ω SST, 856,284 cells of 8 cm. The flow is saved every second for the first 150 s and every 10 s up to 600 s (a save every 10 s was not enough: my CO₂ then differed from OpenFOAM's by up to 45 %). |
| **CO₂** | My own finite-volume solver on the GPU (`fv_cons.py`) computes 30 minutes: the real flow for 150 s, then the mean flow of 150–600 s, saved every 30 s. |
| **Transition runs** | As in the paper, a run starts from the CO₂ of another setting; the people are switched off and the new flow carries the CO₂ away. The CO₂ equation is linear, so the number of people can be chosen freely (10–80). |
| **Forecast model** (final model) | The CO₂ at 1.6 m of the last 6 minutes (12 maps), window speeds, open windows and doors and the number of people go in; the CO₂ of the next 3 minutes comes out, with an uncertainty. 5 networks are averaged. The final model also gets the time since the last window change; 90 % of its training samples have a window change inside the 6 minutes (`--tau --pad-prob 0.9`). |
| **Sensor model** | Gets the CO₂ history only at the 8 stations (6 windows, 2 doors) and forecasts the whole plane. I also tested one more sensor in the middle, and a warning for 700 and 800 ppm. |

---

## 5. Results

Simulation, 16 test settings. "Strict" counts only the CO₂ above the outdoor level; "paper" is the definition of the paper
(400 ppm added).

| | Strict | Paper |
|---|---|---|
| Reproduction of the paper (their code and data) | 29.8 % | 11.46 % (paper: 10.90 %) |
| **Final forecast model** | **0.40 %** | **0.14 %** |
| "CO₂ stays as it is now" (persistence) | 1.77 % | 0.44 % |
| Sensor model (8 stations), random splits | about 32 % | about 5–6 % |

With a window change inside the history the final model has a strict error of 21.3 % (persistence 40.1 %). The warning of the
sensor model finds about 87–91 % of the real cases at 700 ppm with a margin of 30–40 ppm, with about 2–5 % false alarms.
Details and tables: [`thesis/Thesis.pdf`](thesis/Thesis.pdf).

**Time** (measured): OpenFOAM for the first 150 s of one setting about 2 h 15 min on 4 cores; a full 600 s run on one core
15–90 h; the CO₂ solver 1.25–4 h per setting on the GPU; training 5 networks about 3 h 12 min.

---

## 6. How to run (levels 2 and 3)

Packages: `pip install -r requirements.txt` in any environment (the scripts use the active `python3`, or `CFD_PY=/path`).
OpenFOAM v2412 (openfoam.com) is installed separately and its `etc/bashrc` is sourced (or `FOAM_BASHRC` is set); server,
SLURM and Docker notes are in [`experiments/gnot_openfoam_rans/README_batch.md`](experiments/gnot_openfoam_rans/README_batch.md).

```bash
# 0. tests (a few minutes, CPU)
cd experiments/gnot_openfoam_rans && python3 test_fv_cons.py && python3 test_forecast.py

# 1. OpenFOAM, one setting (4 cores): 0-150 s with a save every second, then up to 600 s; continues if started again
cd ../co2_sampling && bash run_full_case.sh N01 4            # writes cases/N01ws1_dx0.08/ and a file "done"

# 2. CO2 dataset and transition files (GPU); collect_v4.sh does this for all finished cases
python3 build_v4.py --name A01 --case cases/A01ws1_dx0.08 --out data_v4 \
    --scen ../gnot_openfoam_rans/scenarios_v4.txt --trans-out data_transitions_v4

# 3. training of the final model (5 networks, about 38 min each on an RTX A2000) -> checkpoints/<tag>/
cd ../gnot_openfoam_rans
python3 train_forecast.py --data-dir data_transitions_v4 --transitions --scenarios scenarios_v4.txt \
    --tag v4_tr_tau09_long --z 1.6 --c-offset 400 --open-flags --pad-prob 0.9 --tau --iters 40000

# 4. test with a window change inside the history (and the bound of a simple mixing-box rule, no model, about 1 minute)
python3 eval_history_change.py --ckpt checkpoints/v4_tr_tau09_long --data-dir data_transitions_v4 --scenarios scenarios_v4.txt
python3 oracle_mean_bound.py --data-dir data_transitions_v4 --scenarios scenarios_v4.txt

# 5. sensor model and warning (all scripts and tests: experiments/sensors/README.md)
cd ../sensors
python3 train_sensor.py --data-dir ../gnot_openfoam_rans/data_transitions_v4 --transitions \
    --scenarios ../gnot_openfoam_rans/scenarios_v4.txt --tag my_run --noise 10
python3 diagnose_sensor.py --data-dir ../gnot_openfoam_rans/data_transitions_v4 --transitions \
    --scenarios ../gnot_openfoam_rans/scenarios_v4.txt --tag my_run
```

The older `run_batch.sh` saves the flow only every 10 s and should not be used for new data. The learning curve of the final
setup (13, 26 and 39 training settings) is `experiments/gnot_openfoam_rans/runs/run_learning_curve_v4.sh`.

---

## 7. Next steps

- Test the forecast and the warning with real sensor data when the stations deliver it.
- Use the model to choose window openings that keep the CO₂ low with as little ventilation as possible, as in the paper.

---

## 8. Original paper code and references

The folders `learning/`, `simulation/`, `control/`, `local/` and `scripts/` are the code of Bian & Shi (2025); how to
reproduce their results is in [`learning/README.md`](learning/README.md). The dataset and code of the original paper are for
research use only.

- Y. Bian, Y. Shi (2025). Data-driven operator learning for energy-efficient building control. arXiv:2504.21243.
- Z. Hao et al. (2023). GNOT: A general neural operator transformer for operator learning. ICML.
