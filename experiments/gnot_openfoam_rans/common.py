"""
All settings of the CFD-data approach in one place: air, turbulence, CO2 from people and the seating area.
Geometry and some solver parts are imported from experiments/pinn.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GNOT_DIR = os.path.normpath(os.path.join(HERE, "..", "pinn"))
OPENFOAM_DIR = os.path.join(GNOT_DIR, "openfoam")
for p in (OPENFOAM_DIR, GNOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
CASES_DIR = os.path.join(HERE, "cases")
DATA_DIR = os.path.join(HERE, "data")

# air and CO2
NU_AIR = 1.5e-5            # m^2/s
D_CO2 = 1.6e-5             # m^2/s
SC_T = 0.7                 # turbulent Schmidt number inside OpenFOAM
# dataset CO2: mass-conserving solver with Sc_t = 0.3 (more mixing makes up for the averaged-out fluctuations;
# chosen on cases V02, V04, V08: error vs OpenFOAM 6-9 % -> 2-4 %)
DATA_CO2_SOLVER = "cons"
SC_T_DATA = 0.3

# turbulence at the open windows (assumed) and in the quiet room
TURB_INTENSITY = 0.05
TURB_LENGTH = 0.1          # m
C_MU = 0.09
K_AMBIENT, OMEGA_AMBIENT = 1e-5, 0.1

# CO2 from people (Bian & Shi 2025): 6 L/min exhaled at 40,000 ppm; outdoor air 400 ppm
EXHALE_M3S = 6e-3 / 60.0
PPM_EXHALED = 40000.0
PPM_OUTDOOR = 400.0        # the computed field is the excess over this value
PPM_M3S_PER_PERSON = EXHALE_M3S * (PPM_EXHALED - PPM_OUTDOOR)
N_REF = 20.0               # people in every simulation; the CO2 scales exactly with the number

# seating area where the CO2 is released (placeholder until the real seating plan is known)
SEAT_BOX = ((3.0, 12.5), (2.0, 7.0), (1.0, 1.2))
SEAT_BOX_IS_PLACEHOLDER = True
BREATHING_Z = 1.1          # m, plane stored in the dataset files
