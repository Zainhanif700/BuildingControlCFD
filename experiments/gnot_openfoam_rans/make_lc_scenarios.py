"""
Scenario files with fewer training cases (5, 10, 20) for the learning curve; the 8 test cases stay the same.
Usage: python3 make_lc_scenarios.py 5 10 20
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = os.path.join(HERE, "scenarios.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", type=int, nargs="+")
    ap.add_argument("--scenarios", default=SCEN)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data_transitions"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    lines = open(args.scenarios).read().splitlines()
    head = [l for l in lines if l.startswith("#")]
    rows = [l for l in lines if l.strip() and not l.startswith("#")]
    have = lambda l: os.path.isfile(os.path.join(args.data_dir, l.split()[0] + ".npz"))
    closed = lambda l: l.split()[2] == "0,0,0,0,0,0,0,0"
    test = [l for l in rows if l.split()[1] == "test" and not closed(l)]
    train = [l for l in rows if l.split()[1] == "train" and not closed(l) and have(l)]
    perm = np.random.default_rng(args.seed).permutation(len(train))
    os.makedirs(os.path.join(HERE, "scenarios_lc"), exist_ok=True)
    print(f"{len(train)} training cases, {len(test)} test cases")
    for n in sorted(args.sizes):
        assert n <= len(train), f"only {len(train)} training cases"
        sub = [train[i] for i in sorted(perm[:n])]
        out = os.path.join(HERE, "scenarios_lc", f"scen_n{n}.txt")
        with open(out, "w") as f:
            f.write("\n".join(head + sub + test) + "\n")
        print(f"n = {n:2d}: {' '.join(l.split()[0] for l in sub)}  -> {out}")


if __name__ == "__main__":
    main()
