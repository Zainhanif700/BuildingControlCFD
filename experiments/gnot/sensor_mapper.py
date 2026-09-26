"""
Sensor -> GNOT input mapper.

PURPOSE
-------
GNOT itself only ever needs 5 numbers per query: (x, y, z, t, V1..V8, N_people)
-> (u, v, w, p, c). It doesn't know or care where those numbers came from --
whether they were randomly made up during physics-only training, or measured
by a real sensor once Alexander's hardware is running.

This file is the ONLY place that should ever need to change once real sensor
data exists. Everything else (point_sampler.py, the GNOT network, training
loop) stays untouched -- they just keep calling the same functions below.

WHAT WE KNOW SO FAR (from Alexander's reply):
  - 18 sensors total: 5 wind speed, 8 CO2/temp/humidity, 5 distance
  - sampling every 1-5 minutes depending on battery
  - velocity sensors give MAGNITUDE only, direction is inferred from the
    sensor's known, fixed mounting orientation
  - sensor data is for INFERENCE only right now, not training

WHAT WE DON'T KNOW YET (see the questions listed at the bottom of this file
and in the chat) -- the functions below are stubs with clearly marked
assumptions until these are answered.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from point_sampler import NUM_WINDOWS, WINDOWS, DOORS, ROOM_X, ROOM_Y, ROOM_Z


# ---------------------------------------------------------------------------
# 1) Raw sensor reading -- one row of whatever Alexander's data export looks
#    like. Field names/units are PLACEHOLDERS until confirmed.
# ---------------------------------------------------------------------------
@dataclass
class RawSensorReading:
    sensor_id: str
    sensor_type: str          # "velocity" | "co2_temp_humidity" | "distance"
    timestamp: float          # seconds (or epoch -- TBD)
    value: float              # magnitude for velocity (m/s), ppm for CO2, etc.
    position_xyz: Optional[tuple] = None  # (x,y,z) in meters -- TBD if provided


# ---------------------------------------------------------------------------
# 2) Sensor -> window mapping.
#    OPEN QUESTION: there are only 5 velocity sensors but 8 windows, so this
#    can't be a simple 1-to-1 table yet. Placeholder assumes some windows
#    share a sensor reading (nearest sensor) until Alexander confirms exact
#    sensor placement (see questions below).
# ---------------------------------------------------------------------------
# window_index -> sensor_id  (FILL IN once Alexander gives sensor positions)
WINDOW_TO_VELOCITY_SENSOR = {
    0: None,  # TODO
    1: None,
    2: None,
    3: None,
    4: None,
    5: None,
    6: None,
    7: None,
}


def map_velocity_readings_to_V(readings: list[RawSensorReading]) -> np.ndarray:
    """Turn raw velocity sensor readings into the 8-length V1..V8 array GNOT
    expects. Windows with no assigned sensor default to 0 (closed) until we
    know the real mapping -- clearly wrong long-term, fine as a placeholder.
    """
    by_id = {r.sensor_id: r.value for r in readings if r.sensor_type == "velocity"}
    V = np.zeros(NUM_WINDOWS)
    for k in range(NUM_WINDOWS):
        sensor_id = WINDOW_TO_VELOCITY_SENSOR.get(k)
        if sensor_id is not None and sensor_id in by_id:
            V[k] = by_id[sensor_id]
    return V


def estimate_occupancy(readings: list[RawSensorReading]) -> float:
    """Turn the 5 distance-sensor readings into an N_people estimate.
    OPEN QUESTION: what counting logic does Alexander's setup actually use
    (people-counting beams? CO2-based inference? manual headcount)? Placeholder
    just returns 0 until this is confirmed.
    NOTE (v12): the model's CO2 is exactly proportional to N_people, so this
    placeholder 0 makes predicted CO2 exactly 0 everywhere -- replace it before
    any sensor-driven use.
    """
    # TODO: replace once we know how occupancy is actually derived
    return 0.0


def get_validation_points(readings: list[RawSensorReading]):
    """CO2/temp/humidity sensors aren't fed INTO GNOT -- they're used to
    CHECK GNOT's own output. Returns [(x,y,z,t,measured_co2), ...] so we can
    compare against GNOT's predicted c(x,y,z,t) at the same spot.
    Requires each sensor's real (x,y,z) position -- currently unknown.
    """
    out = []
    for r in readings:
        if r.sensor_type == "co2_temp_humidity" and r.position_xyz is not None:
            out.append((*r.position_xyz, r.timestamp, r.value))
    return out


def sensor_batch_to_gnot_scenario(readings: list[RawSensorReading]):
    """Single entry point: raw sensor batch -> (V1..V8, N_people) ready to
    feed into a trained GNOT model for inference at any query point."""
    V = map_velocity_readings_to_V(readings)
    N_people = estimate_occupancy(readings)
    return V, N_people


if __name__ == "__main__":
    # placeholder smoke test with made-up readings
    fake = [
        RawSensorReading("vel_1", "velocity", timestamp=0.0, value=1.2),
        RawSensorReading("vel_2", "velocity", timestamp=0.0, value=0.4),
    ]
    V, N = sensor_batch_to_gnot_scenario(fake)
    print("V (all zero until WINDOW_TO_VELOCITY_SENSOR is filled in):", V)
    print("N_people (placeholder):", N)
