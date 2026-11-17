"""
A fairer choice of the warning margin: leave one split out.
Reads the diagnosis files of the three random splits (~/diagm_st_chk_<kind><split>.txt, made by run_diag_margins.sh).
For every split the margin is chosen on the OTHER two splits (the smallest margin that finds at least TARGET of the real
events there), then found / missed / false alarms are counted on the left-out split with that margin.
Usage: python margin_loso.py [directory with the diagm files, default ~]   (kind d = default, ps = tuned + physics)
"""
import os
import re
import sys

TARGET = 0.85                       # share of the real events the margin has to find on the two choosing splits
HOME = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~")
ROW = re.compile(r"^\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$")


def read(path):
    """{(limit, margin): (real, found, missed, false alarms)} from one diagnosis file."""
    out, on = {}, False
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("warning with a safety margin"):
            on = True
            continue
        m = ROW.match(line) if on else None
        if m:
            lim, mar, real, found, missed, fa = map(int, m.groups())
            out[(lim, mar)] = (real, found, missed, fa)
    return out


for kind, name in (("d", "default"), ("ps", "tuned + physics")):
    splits = [read(os.path.join(HOME, f"diagm_st_chk_{kind}{s}.txt")) for s in (1, 2, 3)]
    print(f"== {name}: margin chosen on two splits, counted on the third")
    for lim in (700, 800):
        margins = sorted({m for (l, m) in splits[0] if l == lim})
        tot = [0, 0, 0, 0]
        picks = []
        for held in range(3):
            others = [s for i, s in enumerate(splits) if i != held]
            chosen = margins[-1]
            for m in margins:
                real = sum(o[(lim, m)][0] for o in others)
                found = sum(o[(lim, m)][1] for o in others)
                if real and found / real >= TARGET:
                    chosen = m
                    break
            picks.append(chosen)
            r = splits[held][(lim, chosen)]
            tot = [a + b for a, b in zip(tot, r)]
        print(f"  limit {lim} ppm: margins chosen {picks}; held-out result: real {tot[0]}, found {tot[1]} "
              f"({100 * tot[1] / max(tot[0], 1):.0f} %), missed {tot[2]}, false alarms {tot[3]}")
