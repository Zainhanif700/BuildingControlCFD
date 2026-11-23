# Changelog

Versions are named by their date, `vYYYY-MM-DD`. The newest entry is on top. A script that was changed has an
"Updated" line with the date in its first lines. Older changes: `git log`.

## v2026-11-23c
- `experiments/gnot_openfoam_rans/README_batch.md` rewritten in simpler words, with a table "Which script do I use?".
- New short READMEs for the details that are no longer in the main README: `experiments/co2_sampling/README.md` (how the
  dataset is made, `collect_v4.sh`, `server_run.sh`, `build_v3.py`) and `experiments/pinn/README.md`; the sensor README
  lists the scripts `warning_rules.py`, `lecture_demo.py` and `diagnose_position.py` again.
- `README.md`: two details put back (the weak first minute after a window change; `oracle_mean_bound.py` in step 4).
  No code changed.

## v2026-11-23b
- `README.md` restructured and shortened (from 261 to 143 lines): "Start here" with the three levels first, then
  what is where, the room and the data, the method, the results, how to run, limits. The instructions for reproducing the
  paper moved to `learning/README.md`. No code changed.

## v2026-11-23
- Added the demo: `run_demo.sh` and `demo_forecast.py` (the final model forecasts one test setting; OpenFOAM runs one
  setting for 20 s). Tested on our lab PC.
- Added overview figures: `demo_overview.py` (final model), `sensor_overview.py` and `sensor_forecast.py` (sensor model,
  forecast at a chosen point from a sensor file).
- `run_rans_case.sh` and `run_full_case.sh`: `TRACE=1` writes every command with time and line into the log; the log
  starts with the bash version and the system; a failure while loading OpenFOAM and a stopped OpenFOAM run are now reported.
- `README.md`: new section "Start here" with the three levels; this changelog.
- Not yet tested on a Mac or on the DGX Spark.
