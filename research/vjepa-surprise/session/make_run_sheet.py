"""
Generate the randomised run sheet for the staged near-miss session.

Conditions are shuffled within blocks so lighting drift and actor fatigue
spread evenly across conditions; entry side and actor alternate within each
condition. Same seed → same sheet, so the order is fixed before the session
(part of the pre-registration in STAGED_SESSION_PLAN.md).

  python research/vjepa-surprise/session/make_run_sheet.py > run_sheet.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys

# condition: (count per block, description shown on the sheet)
CONDITIONS = {
    "O1": (5, "Occluded step-out: wait hidden between parked cars, step into the lane"),
    "O2": (5, "Edge entry: walk in from outside the frame edge, straight into the lane"),
    "V1": (5, "Visible approach: stand in view >=5 s at the far side, then walk into the lane"),
    "V2": (3, "Vehicle-led: car pulls out of a bay; actor already standing >=5 m away in the lane"),
    "N":  (2, "Negative control: walk the foreground path only, never enter the lane"),
}
SIDES = ("left", "right")
ACTORS = ("A", "B")


def build(blocks: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    counters = {c: 0 for c in CONDITIONS}
    rows = []
    for b in range(1, blocks + 1):
        block = [c for c, (n, _) in CONDITIONS.items() for _ in range(n)]
        rng.shuffle(block)
        for c in block:
            k = counters[c]
            counters[c] += 1
            rows.append({
                "trial": len(rows) + 1, "block": b, "condition": c,
                "side": SIDES[k % 2],
                "actor": ACTORS[(k // 2) % 2] if c != "V2" else ACTORS[k % 2],
                "instruction": CONDITIONS[c][1],
                "wall_clock_start": "", "slate_shown": "", "valid": "", "notes": "",
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261120)
    args = ap.parse_args()
    rows = build(args.blocks, args.seed)
    w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


if __name__ == "__main__":
    main()
