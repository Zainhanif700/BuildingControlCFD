"""
Small test of compare_versions.py with fake OpenFOAM cases (no OpenFOAM needed):  python test_compare.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N = 50


def write_field(path, arr, cls):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = "\n".join("(" + " ".join(f"{v:.8g}" for v in r) + ")" for r in arr)
    with open(path, "w") as f:
        f.write(f"FoamFile\n{{\n    class {cls};\n}}\n\ninternalField   nonuniform List<vector> \n{len(arr)}\n(\n{rows}\n)\n;\n\n"
                "boundaryField\n{\n    walls\n    {\n        type fixedValue;\n        value uniform (0 0 0);\n    }\n}\n")


def fake_case(root, name, centres, noise, seed):
    rng = np.random.default_rng(seed)
    base = np.random.default_rng(0).normal(0, 0.2, (N, 3))          # the same "true" flow for all runs
    case = os.path.join(root, name)
    write_field(os.path.join(case, "0", "C"), centres, "volVectorField")
    for t in range(0, 151):
        write_field(os.path.join(case, f"{t}", "U"), base + rng.normal(0, noise, (N, 3)), "volVectorField")
    return case


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        centres = np.random.default_rng(1).uniform(0, 10, (N, 3))
        a = fake_case(tmp, "A", centres, 0.01, 1)
        b = fake_case(tmp, "B", centres, 0.02, 2)
        c = fake_case(tmp, "C", centres[::-1].copy(), 0.01, 3)          # other cell order
        out = subprocess.run([sys.executable, os.path.join(HERE, "compare_versions.py"), f"A={a}", f"B={b}", f"C={c}"],
                             capture_output=True, text=True)
        print(out.stdout, out.stderr)
        assert out.returncode == 0, out.stderr
        assert "A vs B: same mesh" in out.stdout
        assert "A vs C: the meshes are NOT the same" in out.stdout
        print("ALL TESTS PASSED")
