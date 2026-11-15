"""
Draft of the window scenarios for the room as it will be used with the sensors:
W5 and W8 always closed, door 2 always closed, door 1 open or closed.
Same line format as scenarios_v4.txt:  name  train/test  V1,...,V8  door1,door2
(V > 0: air comes in with this speed, 0: closed, -1: air goes out through this window)
This is only a draft. The speed range has to be fixed when the real window opening is known.
Usage: python make_scenarios_v5.py > scenarios_v5_draft.txt
"""
import numpy as np

WINDOWS = [1, 2, 3, 4, 6, 7]        # windows that can be opened
V_MIN, V_MAX = 0.05, 3.0            # m/s, over the full height opening of the model
N_DOOR_OPEN, N_DOOR_CLOSED = 24, 6
TEST_EVERY = 5


def speed(rng):
    # log scale, so slow and fast speeds are both covered
    v = np.exp(rng.uniform(np.log(V_MIN), np.log(V_MAX)))
    return round(float(v) / 0.05) * 0.05


def scenario(rng, door1_open):
    while True:
        V = [0.0] * 8
        for w in WINDOWS:
            if rng.random() < 0.5:
                V[w - 1] = speed(rng)
        inlets = [w for w in WINDOWS if V[w - 1] > 0]
        if not inlets:
            continue
        if not door1_open:
            # with both doors closed the air has to leave through another window
            free = [w for w in WINDOWS if V[w - 1] == 0]
            if not free:
                continue
            for w in rng.choice(free, size=rng.integers(1, min(2, len(free)) + 1), replace=False):
                V[w - 1] = -1.0
        return V


def main():
    rng = np.random.default_rng(5)
    print("# name split V1..V8 doors(door1,door2) -- draft for W5, W8 and door 2 closed")
    for i in range(1, N_DOOR_OPEN + N_DOOR_CLOSED + 1):
        door1 = i <= N_DOOR_OPEN
        V = scenario(rng, door1)
        split = "test" if i % TEST_EVERY == 0 else "train"
        print(f"B{i:02d} {split} {','.join(f'{v:g}' for v in V)} {int(door1)},0")


if __name__ == "__main__":
    main()
