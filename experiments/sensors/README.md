# Sensor code (classroom 1-130)

Code for the forecast with the real sensor stations (CO2 at 8 stations: 6 windows and both doors). Everything here
runs on simulated data; the real stations are not delivering data yet.

## Files
| File | What it does |
|---|---|
| `stations.py` | The 8 stations (names, assumed positions 0.3 m from the wall), which windows stay closed, and the virtual stations used for the "what if a sensor stood in the middle" tests. |
| `synthetic.py` | Makes sensor data in the format of the sensor server from one simulation file (with noise). |
| `readers.py` | Reads sensor data: server CSV export, old CSV dump, database (the database reader is not tested). |
| `mapper.py` | Turns sensor records into the model input: 12 CO2 values per station (6 min), window speeds, door states. |
| `sensor_model.py` | The forecast model that only sees the station values (and options such as the physics token). |
| `train_sensor.py` | Trains it (5 members by default). |
| `diagnose_sensor.py` | Errors per scenario, error in ppm, the warning test, error by time. |
| `demo_chain.py` | The whole path: sensor file, reader, mapper, trained model, forecast, compared with the simulation. |
| `sensor_forecast.py` | Forecast at any point of the room from a sensor file (input: the last 6 min of the 8 stations; output: the next 3 min); makes one overview image. |
| `sensor_overview.py` | Overview image for a trained sensor model on one test setting (maps, seating-area average, one point). |
| `warning_rules.py` | Found cases and false alarms for other warning rules (no training; uses trained sensor models). |
| `lecture_demo.py` | A synthetic lecture (420 ppm start, students come and go) through the whole chain, with the warning. |
| `diagnose_position.py` | How well a place in the room tells the average CO2 of the seating area (why one place gives no gain). |
| `test_sensors.py` | Small tests with fake data. They run on a CPU in a few minutes. |
| `runs/run_*.sh` | The test series I ran (each script says what it does in its first lines; start them as `bash runs/run_final.sh <data dir> <scenario file>`). |

## Quick start
```
python test_sensors.py
python train_sensor.py --data-dir <data_transitions dir> --transitions --scenarios <scenarios_v4.txt> --tag my_run
python diagnose_sensor.py --data-dir <data_transitions dir> --transitions --scenarios <scenarios_v4.txt> --tag my_run
```
The data files and the trained models are not in the repository (they are large). The scenario file is
`experiments/gnot_openfoam_rans/scenarios_v4.txt`. The dataset with the all-openings scenarios is made with
`experiments/co2_sampling/run_full_case.sh` (OpenFOAM with a save every second, then my CO2 solver).

## Results so far (simulation only)
See the thesis chapter on the sensors (chapter 9). The tests in `runs/run_*.sh` give the numbers.
