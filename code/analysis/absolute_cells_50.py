"""
Absolute-cell accounting: the result that says percentage coverage misleads.

There is a reading of the main results that looks obvious and is wrong. APF's
coverage percentage RISES with obstacle density -- 91.4% at 0% obstacles up to
94.7% at 30% -- which invites the conclusion that structured obstacles somehow
help it search. Corridors to follow, fewer open spaces to get lost in.

That is not what is happening. Coverage is a ratio, and as obstacles increase
the denominator shrinks faster than the numerator does. Counting actual cells
instead of percentages:

    APF   covers 640 fewer cells at 30% than at 0%, a fall of 28%
    Boids 727 fewer, 33%
    MARL  824 fewer, 34%
    free area itself falls from 2,500 to 1,737 cells, 31%

Every controller sweeps LESS ground as density rises. APF's percentage climbs
only because its absolute coverage falls more slowly than the denominator.
Nobody becomes a better searcher in clutter; APF just degrades most
gracefully.

This generalises past this project. Coverage over free cells is the standard
metric in the literature surveyed in the report, so any study that varies
obstacle density while reporting only that ratio invites the same misreading.

NO NEW SIMULATION IS NEEDED to establish this, which is worth noticing.
Absolute cells is coverage percentage times the free area of that layout, and
both are already stored: coverage per seed in each controller's result file,
and the layout from the deterministic generator. This script just multiplies
them back together across all fifty seeds.

An earlier version of this analysis ran on 12 seeds while every other result
used 50, which left one of the stated contributions resting on a quarter of
the evidence behind the rest. It also gave slightly different numbers -- APF
fell 588 cells rather than 640. This is the 50-seed version.

Run:  python3 -m analysis.absolute_cells_50      (from code/)
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
for _sub in ["", "phase2_boids"]:
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phase2_boids.obstacle_generator import (
    generate_structured_obstacles)

DENSITIES = [0.0, 0.1, 0.2, 0.3]
N_SEEDS = 50
GRID = 50

SOURCES = {
    "Boids": "phase2_boids/results/boids_results_A.npy",
    "APF":   "phase3_apf/results/apf_results_A.npy",
    "MARL":  "phase5_marl/results/marl_results_A.npy",
}
OUT = os.path.join(_HERE, "results", "absolute_cells_50.npy")


def free_cells(density, seed):
    """Obstacle-free cell count for one layout.

    Regenerated rather than stored. The generator is deterministic given the
    seed, so this returns exactly the map the controllers actually ran on.
    """
    grid = generate_structured_obstacles(
        grid_size=GRID, obstacle_density=density,
        seed=int(seed), style="urban").astype(bool)
    return int(np.sum(~grid))


def main():
    """Recover absolute cells from stored coverage and regenerated layouts."""
    free = {d: np.array([free_cells(d, s) for s in range(N_SEEDS)])
            for d in DENSITIES}

    out = {}
    for name, rel in SOURCES.items():
        store = np.load(os.path.join(_CODE, rel), allow_pickle=True).item()
        out[name] = {}
        for d in DENSITIES:
            runs = sorted(store[d], key=lambda r: r["seed"])
            # Hard fail rather than silently averaging a partial set: if the
            # seeds are not the full matched 0-49, the multiplication below
            # would pair a coverage figure with the wrong layout.
            assert [r["seed"] for r in runs] == list(range(N_SEEDS)), \
                f"{name} at {d} is not the full matched seed set"
            pct = np.array([r["coverage"] for r in runs])
            # The whole computation: percentage x free area = cells.
            out[name][d] = {"pct": pct,
                            "free": free[d],
                            "cells": pct / 100.0 * free[d]}
    out["free"] = free
    np.save(OUT, np.array(out, dtype=object))

    hdr = f"{'':16s}" + "".join(f"{int(d*100):>9}%" for d in DENSITIES)
    print(f"\nAbsolute cells covered, mean over {N_SEEDS} seeds "
          "(Seed Set A)\n")
    print(hdr)
    print(f"{'Free available':16s}"
          + "".join(f"{free[d].mean():>10.0f}" for d in DENSITIES))
    for name in SOURCES:
        cells = [out[name][d]["cells"].mean() for d in DENSITIES]
        pct = [out[name][d]["pct"].mean() for d in DENSITIES]
        drop = cells[0] - cells[-1]
        print(f"{name + ', cells':16s}"
              + "".join(f"{c:>10.0f}" for c in cells)
              + f"   fall {drop:.0f} ({100*drop/cells[0]:.0f}%)")
        print(f"{name + ', %':16s}" + "".join(f"{p:>10.1f}" for p in pct))
    fd = free[DENSITIES[0]].mean() - free[DENSITIES[-1]].mean()
    print(f"\nFree area falls {fd:.0f} cells "
          f"({100*fd/free[DENSITIES[0]].mean():.0f}%) over the same range.")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
