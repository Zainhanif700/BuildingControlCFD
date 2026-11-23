# The final dataset (68 settings)

*Updated 2026-11-23.* How the data of the thesis is made. The main README has the short version; this is the detail.

1. **OpenFOAM, one setting** (CPU): `bash run_full_case.sh N01 4` (setting name, cores). It runs 0–150 s with a save of the
   flow every second (the real start-up flow), then up to 600 s with a save every 10 s (for the mean flow; the second part is
   `continue_case.sh`, which can also be used alone). It writes `cases/N01ws1_dx0.08/` and a file `done`, and continues where it
   stopped if you start it again.
2. **CO₂ dataset and transition file** (GPU): `python3 build_v4.py --name A01 --case cases/A01ws1_dx0.08 --out data_v4 --scen
   ../gnot_openfoam_rans/scenarios_v4.txt --trans-out data_transitions_v4`. My CO₂ solver runs on the real flow of the first
   150 s (turbulent Schmidt number 0.7), then on the mean flow of 150–600 s (0.3). The CO₂ is saved every 30 s for 30 minutes
   and is checked against OpenFOAM's own CO₂ at 150 s and 600 s.
3. **All finished cases:** `bash collect_v4.sh <file with the machines, or /dev/null> [interval in s]`. Every few minutes it
   copies finished cases from other machines, builds the dataset file of each finished case, and builds the transition file of
   a new setting as soon as its partner's file exists. With `DELETE_AFTER_BUILD=1` it deletes the local case after the build,
   because the dataset file keeps the flow. It stops when every new setting has its transition file.
4. **Another machine:** `bash server_run.sh setup` installs OpenFOAM and Python in the home folder; then
   `bash server_run.sh run 4 A01 A02 ...` runs a list of settings at low priority. The line in the first lines of the script
   removes everything again.

The 28 N settings (N01–N28) were built earlier with `build_v3.py`, which uses the same method. Other files here are tests and
comparisons of the method (`compare_*.py`, `eval_cross.py`, `check_openings.py`, `run_sampling_test.sh`); each says in its first
lines what it does.
