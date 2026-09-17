import pickle
import numpy as np

with open("dataset/train_data_norm.pkl", "rb") as f:
    data = pickle.load(f)

sample = data[0]
mesh, future_co2, controls, past_co2 = sample[0], sample[1], sample[2], sample[3][0]

print(f"Occupant count: {int(controls[0])}")
for i in range(6):
    rate, angle = controls[1 + i*2], controls[2 + i*2]
    print(f"  Vent {i+1}: flow rate = {rate:.3f} m/s, angle = {angle:.2f} deg")

print("\n=== Point-by-point view (same row index = same physical point) ===\n")
for i in range(5):
    print(f"--- Point {i} ---")
    print(f"  Location (x, y, z):        {mesh[i]}")
    print(f"  Past CO2 (last 12 steps = 6 min):  {past_co2[i]}")
    print(f"  Future CO2 (next 6 steps = 3 min, normalized): {future_co2[i]}")
    print()