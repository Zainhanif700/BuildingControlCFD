"""
Turns sensor records into the inputs of the forecast model:
  - CO2 of the last 6 minutes at every station (12 values, 30 s apart, ppm above outdoor air)
  - the 8 window speeds and the 2 door states
"""
import numpy as np

import stations as st

STEP = 30.0     # s
N_HIST = 12


def model_inputs(records, t_end, people):
    """records: list from readers.py. t_end: time of the newest value to use (unix seconds)."""
    grid = t_end - STEP * np.arange(N_HIST - 1, -1, -1)
    by_station = {}
    for r in records:
        by_station.setdefault(r["station"], []).append(r)

    hist = np.zeros((len(st.STATIONS), N_HIST), np.float32)
    V = np.zeros(8, np.float32)
    doors = np.zeros(2, np.float32)
    for i, s in enumerate(st.STATIONS):
        recs = sorted(by_station.get(s.id, []), key=lambda r: r["time"])
        if not recs:
            raise ValueError(f"no data from station {s.name}")
        times = np.array([r["time"] for r in recs])
        for k, g in enumerate(grid):
            # newest value that is not newer than the grid time
            j = int(np.searchsorted(times, g + 1e-6)) - 1
            if j < 0 or g - times[j] > 3 * STEP:
                raise ValueError(f"station {s.name}: no value near {g:.0f} s")
            hist[i, k] = recs[j]["co2"] - st.CO2_OUTDOOR

        last = recs[int(np.searchsorted(times, t_end + 1e-6)) - 1]
        is_open = last["distance"] is not None and last["distance"] > st.OPEN_ABOVE_MM
        if s.kind == "window":
            speed = last["speed"] or 0.0
            V[s.number - 1] = speed * st.SPEED_FACTOR[s.number] if is_open else 0.0
        else:
            doors[s.number - 1] = 1.0 if is_open else 0.0

    xyz = np.array([[s.x, s.y, s.z] for s in st.STATIONS], np.float32)
    return {"hist": hist, "xyz": xyz, "V": V, "doors": doors, "people": float(people), "times": grid}
