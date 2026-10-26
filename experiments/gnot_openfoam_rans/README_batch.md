# Running the OpenFOAM simulations on a server

These scripts run the classroom airflow simulations (OpenFOAM, k-omega SST, 8 cm mesh) for every window
setting in a scenario file. This step only needs CPU cores; the CO2 data is computed afterwards on a GPU.

## Scenario files

The code is the same for every setup; only the scenario file changes. Each line is one setting:
name, train/test, the inflow speed at windows W1-W8 in m/s (0 = closed) and, if given, door 1 and door 2
(1 = open, 0 = closed).

- `scenarios.txt`: first setup, all 8 windows can open, both doors open (40 settings)
- `scenarios_v2.txt`: current setup, only W1, W3 and W6 can open, door 1 open, door 2 closed (28 settings)

A new setup only needs a new scenario file in the same format.

## Setup (once, about 15 min, no root rights needed)

```
git clone -b new-setup https://github.com/Zainhanif700/BuildingControlCFD.git
cd BuildingControlCFD/experiments/gnot_openfoam_rans
bash setup_env.sh
```

It should end with `python OK` and `OpenFOAM OK`.

## Quick test (about 15 min)

One setting, only 20 s of flow, on 4 cores:

```
T_END=20 bash run_batch.sh scenarios_v2.txt 1 4 N01
ls cases/N01_test_dx0.08/done && rm -rf cases/N01_test_dx0.08
```

## All settings

Normal server, e.g. 32 cores = 8 settings at a time with 4 cores each:

```
nohup bash run_batch.sh scenarios_v2.txt 8 4 > batch.log 2>&1 &
```

SLURM cluster, one job per setting (the resources are at the top of `slurm_batch.sh`):

```
mkdir -p logs
sbatch --array=1-$(grep -v '^#' scenarios_v2.txt | awk 'NF' | wc -l)%8 slurm_batch.sh scenarios_v2.txt
```

A finished setting gets a `done` file in `cases/<name>_dx0.08/` and is skipped if the batch is started again.
Each setting has its log in `logs/cases/<name>.log`.

## Run time and disk

On one core a setting takes 15-90 h, depending on the window speeds (faster air needs smaller time steps).
More cores per setting make it faster. A finished case needs about 2-4 GB.

## Why not on the GPU

OpenFOAM runs on the CPU and is parallelised by splitting the mesh over cores. There are add-ons that move
only the pressure solver to the GPU (PETSc4FOAM, AmgX, Ginkgo/OGL), but they have to be built separately and
the reported gain for the whole simulation is small. More CPU cores per case is the simpler way to speed it
up. The GPU is used later, for the CO2 data and the training.
