"""
Readers for the sensor data. Every reader returns the same list of records, so the mapper does not care where
the data comes from:
    {"time": seconds (unix), "station": id, "co2": ppm, "speed": m/s or None, "distance": mm or None}
"""
import csv
from datetime import datetime

FIX = 1_000_000


def _num(text, scale=1.0):
    return None if text in ("", None) else float(text) / scale


def read_server_csv(path):
    """CSV export of the server table sensor_data (integers, value * 1,000,000)."""
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.append({"time": datetime.fromisoformat(r["time"]).timestamp(),
                        "station": int(r["station_id"]),
                        "co2": _num(r["scd41_co2"], FIX),
                        "speed": _num(r.get("pav1015v_velocity"), FIX),
                        "distance": _num(r.get("vl53l1x_distance"), FIX)})
    return out


def read_old_csv(path, station_id=None):
    """The older dataset (dump_*.csv): only CO2, values already in ppm, time in unix seconds."""
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if station_id is not None and int(r["station_id"]) != station_id:
                continue
            out.append({"time": float(r["timestamp"]), "station": int(r["station_id"]),
                        "co2": _num(r["scd41_co2"]), "speed": None, "distance": None})
    return out


def read_database(conn, t_from, t_to):
    """Same records straight from the database. conn is an open DB connection (psycopg2 or similar).
    Not tested yet, because I have no access to the server."""
    cur = conn.cursor()
    cur.execute("SELECT time, station_id, scd41_co2, pav1015v_velocity, vl53l1x_distance FROM sensor_data "
                "WHERE time >= %s AND time <= %s ORDER BY time", (t_from, t_to))
    div = lambda v: None if v is None else v / FIX
    return [{"time": t.timestamp(), "station": int(sid), "co2": div(co2), "speed": div(vel), "distance": div(dist)}
            for t, sid, co2, vel, dist in cur.fetchall()]
