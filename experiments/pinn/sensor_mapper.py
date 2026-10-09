"""
Maps real sensor readings to the model inputs (window speeds, number of people, time).
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from point_sampler import NUM_WINDOWS, WINDOWS, DOORS, ROOM_X, ROOM_Y, ROOM_Z


@dataclass
class RawSensorReading:
    sensor_id: str
    sensor_type: str
    timestamp: float
    value: float
    position_xyz: Optional[tuple] = None


WINDOW_TO_VELOCITY_SENSOR = {
    0: None,
    1: None,
    2: None,
    3: None,
    4: None,
    5: None,
    6: None,
    7: None,
}


def map_velocity_readings_to_V(readings: list[RawSensorReading]) -> np.ndarray:
    """Turn raw velocity sensor readings into the 8-length V1..V8 array GNOT expects."""
    by_id = {r.sensor_id: r.value for r in readings if r.sensor_type == "velocity"}
    V = np.zeros(NUM_WINDOWS)
    for k in range(NUM_WINDOWS):
        sensor_id = WINDOW_TO_VELOCITY_SENSOR.get(k)
        if sensor_id is not None and sensor_id in by_id:
            V[k] = by_id[sensor_id]
    return V


def estimate_occupancy(readings: list[RawSensorReading]) -> float:
    """Turn the 5 distance-sensor readings into an N_people estimate."""
    return 0.0


def get_validation_points(readings: list[RawSensorReading]):
    """Positions of the CO2/temperature/humidity sensors, used to check the model's output."""
    out = []
    for r in readings:
        if r.sensor_type == "co2_temp_humidity" and r.position_xyz is not None:
            out.append((*r.position_xyz, r.timestamp, r.value))
    return out


def sensor_batch_to_gnot_scenario(readings: list[RawSensorReading]):
    """Raw sensor readings -> (V1..V8, N_people) for the model."""
    V = map_velocity_readings_to_V(readings)
    N_people = estimate_occupancy(readings)
    return V, N_people


if __name__ == "__main__":
    fake = [
        RawSensorReading("vel_1", "velocity", timestamp=0.0, value=1.2),
        RawSensorReading("vel_2", "velocity", timestamp=0.0, value=0.4),
    ]
    V, N = sensor_batch_to_gnot_scenario(fake)
    print("V (all zero until WINDOW_TO_VELOCITY_SENSOR is filled in):", V)
    print("N_people (placeholder):", N)
