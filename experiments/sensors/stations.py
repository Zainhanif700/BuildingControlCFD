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

# "What if" test only: virtual stations in the seating area (SEAT_BOX of the simulations is x 3-12.5, y 2-7).
# They do not exist in the room. They show how much one or more sensors in the middle of the room would help.
VIRTUAL_SETS = {
    "1": [(7.75, 4.5)],                                              # middle of the seating area
    "2": [(5.4, 4.5), (10.1, 4.5)],
    "4": [(5.4, 3.25), (10.1, 3.25), (5.4, 5.75), (10.1, 5.75)],
    # one sensor at other places of the seating area
    "front": [(4.0, 4.5)],                                           # blackboard side
    "back": [(11.5, 4.5)],                                           # door 2 side
    "windowside": [(7.75, 6.6)],                                     # towards the window wall
    "doorside": [(7.75, 2.4)],                                       # towards the door wall
    # a grid of 16, to see what many sensors would give
    "16": [(x, y) for y in (2.625, 3.875, 5.125, 6.375) for x in (4.19, 6.56, 8.94, 11.31)],
}


def virtual_stations(n):
    """The virtual stations of one set ("0" = none, "1", "2", "4", "front", ..., "16"), named X<set>_<i>."""
    n = str(n)
    if n == "0":
        return []
    return [Station(100 + i, f"X{n}_{i}", "virtual", i, x, y) for i, (x, y) in enumerate(VIRTUAL_SETS[n], 1)]


def by_name(names):
    """Station objects (real and virtual) for a list of names, e.g. the names saved with a trained model."""
    pool = {s.name: s for s in STATIONS}
    for n in VIRTUAL_SETS:
        pool.update({s.name: s for s in virtual_stations(n)})
    return [pool[name] for name in names]

if __name__ == "__main__":
    for s in STATIONS:
        print(f"{s.id}  {s.name:5s} {s.kind:6s} {s.number}  x = {s.x:5.2f}  y = {s.y:4.2f}  z = {s.z:.2f}")
