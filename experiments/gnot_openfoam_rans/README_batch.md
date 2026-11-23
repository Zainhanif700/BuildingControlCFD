# Running the OpenFOAM simulations (CPU only)

*Updated 2026-11-23. If a run stops without a message, start it again with `TRACE=1` in front, for example
`TRACE=1 bash run_full_case.sh N01 4`. The log then shows every command with the time and the line, and its first line
shows the bash version and the system. All changes with their dates: `CHANGELOG.md` in the main folder.*

These scripts run the airflow simulations of the classroom (OpenFOAM, 8 cm mesh) for the window settings of a scenario
file. They need only CPU cores. The CO₂ is computed afterwards on a GPU (main README, step 2).

## Which script do I use?

| I want to ... | I use |
|---|---|
| check that everything works (about 10 min) | `bash run_demo.sh openfoam` (same as the quick test below) |
| make data for the thesis, one setting | `cd ../co2_sampling && bash run_full_case.sh N01 4` (setting name, number of cores) |
| run many settings at once for tests | `bash run_batch.sh ...` (see below) |

`run_full_case.sh` saves the flow every second for the first 150 s. `run_batch.sh` saves it only every 10 s. With that,
my CO₂ differed from OpenFOAM's CO₂ by up to 45 % in fast cases, so **do not use `run_batch.sh` for new data**. It is fine
for quick tests.

## The settings (scenario files)

One line per setting: name, train/test, the inflow speed at the windows W1–W8 in m/s (0 = closed) and, if given, door 1
and door 2 (1 = open, 0 = closed). A new setup only needs a new file in the same format.

- `scenarios.txt`: first setup, all 8 windows can open, both doors open (40 settings)
- `scenarios_v2.txt`: second setup, only W1, W3 and W6 can open, door 1 open, door 2 closed (28 settings)
- `scenarios_v4.txt`: final setup, the 28 settings above plus 40 settings (A01–A40) where all 8 windows and both doors can
  be open or closed (68 settings)

## Set up once

Python packages (here in a new environment):

```
git clone https://github.com/Zainhanif700/BuildingControlCFD.git
cd BuildingControlCFD/experiments/gnot_openfoam_rans
python3 -m venv ~/venv-cfd && source ~/venv-cfd/bin/activate
bash setup_env.sh
```

OpenFOAM is not a Python package, so you install it separately: **version v2412 from openfoam.com**
([precompiled versions](https://gitlab.com/openfoam/core/openfoam/-/wikis/precompiled); on a Mac with Homebrew:
`brew install gerlero/openfoam/openfoam@2412`). The versions from openfoam.org (12, 13, 14, ...) are a different program and
do not work with these cases. Before you run a script, load OpenFOAM with `source <OpenFOAM folder>/etc/bashrc`, or set
`FOAM_BASHRC=<that file>` and the scripts do it for you. A conda environment named `foam` in `~/anaconda3` (as on our lab
PC) is found automatically.

`setup_env.sh` should end with `python OK` and `OpenFOAM OK`. The scripts use the `python3` of the active environment;
another one can be given with `CFD_PY=/path/to/python`.

## Quick test (about 10 minutes)

One setting (N01), only 20 s of flow, 4 cores. Measured on our lab PC: about 10 minutes.

```
T_END=20 bash run_batch.sh scenarios_v2.txt 1 4 N01
ls cases/N01_test_dx0.08/done && rm -rf cases/N01_test_dx0.08     # it worked; this removes the test folder (about 160 MB)
```

## Many settings

**One server with many cores.** One setting per core (20 cores = 20 settings at a time). This gives the most output;
several cores per setting only help when there are fewer settings than cores.

```
nohup bash run_batch.sh scenarios_v2.txt 20 1 > batch.log 2>&1 &
```

**Another machine.** `../co2_sampling/server_run.sh` installs OpenFOAM and Python in the home folder and runs a list of
settings: `bash server_run.sh setup`, then `bash server_run.sh run 4 A01 A02 ...` (4 cores per setting). The line in its
first lines removes everything again.

**Docker (for example DGX Spark).** The image has OpenFOAM v2412 and the Python packages. The project folder is mounted, so
all results stay on the computer. We used this image for the 20 s tests of all 28 settings on the DGX Spark. Build once in
the main folder of the project, then run the quick test and the batch:

```
docker build -t classroom-cfd .
docker run --rm --user "$(id -u):$(id -g)" --shm-size=2g -v "$PWD":/work -e T_END=20 classroom-cfd \
    bash run_batch.sh scenarios_v2.txt 1 4 N01
docker run -d --name cfd-batch --user "$(id -u):$(id -g)" --shm-size=4g -v "$PWD":/work classroom-cfd \
    bash run_batch.sh scenarios_v2.txt 20 1
docker logs -f cfd-batch
```

The container only runs OpenFOAM. The CO₂ data and the training run afterwards outside the container, in a Python
environment with the GPU version of PyTorch: `bash extract_cases.sh data_v2`.

**SLURM cluster.** One job per setting (the resources are at the top of `slurm_batch.sh`):

```
mkdir -p logs
sbatch --array=1-$(grep -v '^#' scenarios_v2.txt | awk 'NF' | wc -l)%8 slurm_batch.sh scenarios_v2.txt
```

A finished setting gets a file `done` in `cases/<name>_dx0.08/` and is skipped if you start the batch again. Each setting has
its log in `logs/cases/<name>.log`.

## Run time and disk

On one core, one setting took 15–90 h on our lab PC, depending on the window speeds (faster air needs smaller time steps).
With 4 cores it ran 2.7 times faster. A finished case needs about 2–4 GB.

## Why not on the GPU?

OpenFOAM runs on the CPU and is sped up by splitting the mesh over more cores. There are add-ons that move only the
pressure solver to the GPU (PETSc4FOAM, AmgX, Ginkgo/OGL), but they must be built separately, and the reported gain for the
whole simulation is small. More cores per case is the simpler way. The GPU is used later, for the CO₂ data and the training.
