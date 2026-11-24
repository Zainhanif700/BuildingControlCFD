# Changelog

Versions are named by their date, `vYYYY-MM-DD`. The newest entry is on top. A script that was changed has an
"Updated" line with the date in its first lines. Older changes: `git log`.

## v2026-11-24c (24 Nov 2026)
- The three scenario files (`scenarios.txt`, `scenarios_v2.txt`, `scenarios_v4.txt`) now start with one line that says which
  setup they are (first, second, final); the current pipeline uses `scenarios_v4.txt`. The scripts that write them
  (`scenarios.py`, `scenarios_v2.py`, `scenarios_v4.py`) write the same line; the files are otherwise unchanged.
- `compare_datasets.py` moved to `experiments/gnot_openfoam_rans/runs/` (`python3 runs/compare_datasets.py ...`).

## v2026-11-24b (24 Nov 2026)
- Cleanup of `experiments/`: 18 obsolete scripts removed (pilot checks of the first phase, the old dataset batch scripts,
  one-off tests; they referred to pilot cases and folders that no longer exist). The tag `v2026-11-24` has them.
- The experiment runners that made the thesis results moved into `runs/` folders: 10 in `experiments/gnot_openfoam_rans/runs/`
  (for example `run_tau_final.sh`, `run_mesh_study.sh`, `run_learning_curve_v4.sh`) and 15 in `experiments/sensors/runs/`. They
  change into the folder above, so the commands only get the prefix `runs/` (for example `bash runs/run_tau_final.sh`).
- `README.md`: "What is where" now lists `runs/`, `gpu_lbm/` and `openfoam_versions/`.
- `forecast.py`: the sensor model with the physics token runs again; `test_sensors.py` passes (the three tests of the README, step 0, pass).

## v2026-11-24 (24 Nov 2026)
- Added `experiments/gnot_openfoam_rans/run_learning_curve_v4.sh`: the learning curve of the final setup (same training
  as the final model with 13, 26 and 39 of the 52 training settings, same 16 test settings).
- `README.md` redesigned: header with version and date, a version table, numbered sections with dividers, a tip box. The
  content is the same.

## v2026-11-23c (23 Nov 2026)
- `experiments/gnot_openfoam_rans/README_batch.md` rewritten in simpler words, with a table "Which script do I use?".
- New short READMEs for the details that are no longer in the main README: `experiments/co2_sampling/README.md` (how the
  dataset is made, `collect_v4.sh`, `server_run.sh`, `build_v3.py`) and `experiments/pinn/README.md`; the sensor README
  lists the scripts `warning_rules.py`, `lecture_demo.py` and `diagnose_position.py` again.
- `README.md`: `oracle_mean_bound.py` put back in step 4; the section "Limits and next steps" is now only "Next steps".
  No code changed.

## v2026-11-23b (23 Nov 2026)
- `README.md` restructured and shortened (from 261 to 143 lines): "Start here" with the three levels first, then
  what is where, the room and the data, the method, the results, how to run. The instructions for reproducing the
  paper moved to `learning/README.md`. No code changed.

## v2026-11-23 (23 Nov 2026)
- Added the demo: `run_demo.sh` and `demo_forecast.py` (the final model forecasts one test setting; OpenFOAM runs one
  setting for 20 s). Tested on our lab PC.
- Added overview figures: `demo_overview.py` (final model), `sensor_overview.py` and `sensor_forecast.py` (sensor model,
  forecast at a chosen point from a sensor file).
- `run_rans_case.sh` and `run_full_case.sh`: `TRACE=1` writes every command with time and line into the log; the log
  starts with the bash version and the system; a failure while loading OpenFOAM and a stopped OpenFOAM run are now reported.
- `README.md`: new section "Start here" with the three levels; this changelog.
