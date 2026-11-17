"""
Sensor stations in classroom 1-130 and the values we assume about them.
The positions are assumed (0.3 m in front of the window or door, in the middle). Change them here when the
real positions are known.
"""
from dataclasses import dataclass

ROOM_Y = 9.16
Z_STATION = 1.6          # m, for now the same height as the forecast plane
OFFSET = 0.30            # m, distance of a station from the wall

# middle of each window on the window wall (W1..W8) and of each door on the door wall
WINDOW_X = [2.395, 4.415, 5.095, 7.795, 9.805, 10.495, 13.195, 15.215]
DOOR_X = [2.12, 12.85]

CLOSED_WINDOWS = [5, 8]  # these windows stay closed and have no station
CLOSED_DOORS = []        # door 2 can also be open, so it gets a station too (use [2] for the 7-station setup)

# distance sensor: a larger distance than this means the window or door is open (assumed)
OPEN_ABOVE_MM = 100.0
DIST_OPEN_MM, DIST_CLOSED_MM = 300.0, 20.0     # values used for synthetic data

CO2_OUTDOOR = 400.0      # ppm
CO2_NOISE_PPM = 10.0     # noise of the synthetic CO2 values (assumed)

# measured air speed -> speed of the model (full height opening). 1.0 until the real opening size is known.
SPEED_FACTOR = {w: 1.0 for w in range(1, 9)}


@dataclass
class Station:
    id: int
    name: str
    kind: str        # "window" or "door"
    number: int      # window 1..8 or door 1..2
    x: float
    y: float
    z: float = Z_STATION


def all_stations():
    out, i = [], 1
    for w, x in enumerate(WINDOW_X, 1):
        if w not in CLOSED_WINDOWS:
            out.append(Station(i, f"S-W{w}", "window", w, x, ROOM_Y - OFFSET))
            i += 1
    for d, x in enumerate(DOOR_X, 1):
        if d not in CLOSED_DOORS:
            out.append(Station(i, f"S-D{d}", "door", d, x, OFFSET))
            i += 1
    return out


STATIONS = all_stations()

if __name__ == "__main__":
    for s in STATIONS:
        print(f"{s.id}  {s.name:5s} {s.kind:6s} {s.number}  x = {s.x:5.2f}  y = {s.y:4.2f}  z = {s.z:.2f}")
