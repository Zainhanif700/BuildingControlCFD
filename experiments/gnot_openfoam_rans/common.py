"""
Track 2b: realistic physics, following Bian & Shi (2025) -- k-omega SST turbulence, CO2 in ppm from
seated occupants (exhaled air), fresh air 400 ppm. Geometry, mesh steps, readers and solvers are
IMPORTED from experiments/gnot and experiments/gnot_openfoam (nothing there is changed).

All model choices of this folder are collected HERE, so they can be listed and changed in one place.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GNOT_DIR = os.path.normpath(os.path.join(HERE, "..", "gnot"))
OPENFOAM_DIR = os.path.join(GNOT_DIR, "openfoam")
GNOT_OF_DIR = os.path.normpath(os.path.join(HERE, "..", "gnot_openfoam"))
for p in (GNOT_OF_DIR, OPENFOAM_DIR, GNOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
CASES_DIR = os.path.join(HERE, "cases")
DATA_DIR = os.path.join(HERE, "data")

# --- air and CO2 (physical values; Track 1 / laminar Track 2 used effective nu = 0.01, D = 0.005) ---
NU_AIR = 1.5e-5            # m^2/s, kinematic viscosity of air at ~20 C
D_CO2 = 1.6e-5             # m^2/s, molecular diffusivity of CO2 in air
SC_T = 0.7                 # turbulent Schmidt number (common default, also ANSYS Fluent's)

# --- turbulence at the open windows (assumption; no measured value) ---
TURB_INTENSITY = 0.05      # 5 % of the window speed
TURB_LENGTH = 0.1          # m, mixing length at the window
C_MU = 0.09
K_AMBIENT, OMEGA_AMBIENT = 1e-5, 0.1   # initial / backflow values in the room (quiet air)

# --- CO2 from people (Bian & Shi 2025, Sec. 3.2.3: exhaled air 6 L/min per person at 40,000 ppm) ---
EXHALE_M3S = 6e-3 / 60.0   # 6 L/min per person
PPM_EXHALED = 40000.0
PPM_OUTDOOR = 400.0        # fresh air; the transported field is the EXCESS over this value
# excess-CO2 emission per person [ppm * m^3 / s]: exhaled volume flow x (40,000 - 400) ppm
PPM_M3S_PER_PERSON = EXHALE_M3S * (PPM_EXHALED - PPM_OUTDOOR)
N_REF = 20.0

# --- seating area (PLACEHOLDER until the real seating plan is known): source spread uniformly over
#     this box at seated head height. x, y in m (room 15.53 x 9.16), z 1.0-1.2 m.
SEAT_BOX = ((3.0, 12.5), (2.0, 7.0), (1.0, 1.2))
SEAT_BOX_IS_PLACEHOLDER = True
BREATHING_Z = 1.1          # evaluation plane (seated)
