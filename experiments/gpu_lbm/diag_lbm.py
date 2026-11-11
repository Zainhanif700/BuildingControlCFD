"""
Where the GPU pilot differs from OpenFOAM: mean converged?, flow rates, error by speed and height, W1 jet.
Usage: python3 diag_lbm.py data_lbm/N10_lbm_dx0.04.npz <dataset npz of the same setting>
"""
import numpy as np, sys
L = dict(np.load(sys.argv[1])); D = dict(np.load(sys.argv[2]))
assert np.allclose(L["P"], D["P"])
P = D["P"]; x, y, z = P.T
rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
t, Us = L["t_snap"], L["U_snap"]
m1, m2 = Us[(t > 150) & (t <= 375)].mean(0), Us[t > 375].mean(0)
print(f"1) LBM mean 150-375 s vs 375-600 s: {100 * rel(m1, m2):.1f} %   (snapshot mean vs stored mean {100 * rel(Us[t > 150].mean(0), L['U_mean']):.1f} %)")
fl = np.mean([np.linalg.norm(u - L["U_mean"]) / np.linalg.norm(L["U_mean"]) for u in Us[t > 150]])
print(f"2) fluctuation around the mean: LBM {100 * fl:.1f} %, OpenFOAM {100 * float(D['fluctuation']):.1f} %")
ux = np.unique(x); uz = np.unique(z); uy = np.unique(y)
hx, hz = ux[1] - ux[0], uz[1] - uz[0]
door = (y == uy[0]) & (x >= 1.59) & (x <= 2.65) & (z <= 2.11)
top = y == uy[-1]
win = np.zeros_like(top)
for a, b in ((2.09, 2.70), (4.79, 5.40), (10.19, 10.80)):
    win |= top & (x >= a) & (x <= b)
for n, U in (("LBM", L["U_mean"]), ("OpenFOAM", D["U_mean"])):
    print(f"3) {n}: inflow through W1/W3/W6 {-U[win, 1].sum() * hx * hz:.2f} m3/s, outflow at door 1 {-U[door, 1].sum() * hx * hz:.2f} m3/s "
          f"(set: {3 * 0.5 * 0.61 * 3.15:.2f})")
sl, so = np.linalg.norm(L["U_mean"], axis=1), np.linalg.norm(D["U_mean"], axis=1)
for lo, hi in ((0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 9)):
    m = (so >= lo) & (so < hi)
    print(f"4) cells with OpenFOAM speed {lo}-{hi} m/s ({100 * m.mean():.0f} % of room): mean speed LBM {sl[m].mean():.3f} vs OF {so[m].mean():.3f}, velocity error {100 * rel(L['U_mean'][m], D['U_mean'][m]):.0f} %")
for zz in (0.5, 1.1, 2.0, 2.8):
    m = np.abs(z - uz[np.argmin(np.abs(uz - zz))]) < 1e-5
    print(f"5) height {zz} m: mean speed LBM {sl[m].mean():.3f} vs OF {so[m].mean():.3f}, velocity error {100 * rel(L['U_mean'][m], D['U_mean'][m]):.0f} %")
xc = ux[np.argmin(np.abs(ux - 2.395))]; zc = uz[np.argmin(np.abs(uz - 1.6))]
line = (x == xc) & (z == zc)
o = np.argsort(-y[line])
print("6) jet of W1 (x 2.4 m, z 1.6 m), distance from window [m]: speed LBM / OF")
for i in o[::8]:
    print(f"   {9.16 - y[line][i]:5.2f}: {sl[line][i]:.3f} / {so[line][i]:.3f}")
