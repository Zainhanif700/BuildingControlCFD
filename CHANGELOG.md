# Changelog

Versions are named by their date, `vYYYY-MM-DD`. The newest entry is on top. A script that was changed has an
"Updated" line with the date in its first lines. Older changes: `git log`.

## v2026-11-23
- Added the demo: `run_demo.sh` and `demo_forecast.py` (the final model forecasts one test setting; OpenFOAM runs one
  setting for 20 s). Tested on our lab PC.
- Added overview figures: `demo_overview.py` (final model), `sensor_overview.py` and `sensor_forecast.py` (sensor model,
  forecast at a chosen point from a sensor file).
- `run_rans_case.sh` and `run_full_case.sh`: `TRACE=1` writes every command with time and line into the log; the log
  starts with the bash version and the system; a failure while loading OpenFOAM and a stopped OpenFOAM run are now reported.
- `README.md`: new section "Start here" with the three levels; this changelog.
- Not yet tested on a Mac or on the DGX Spark.
