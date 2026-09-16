"""
Two things at once: the APF parameter sweep, and the first pass at the
absolute-cell question.

THE SWEEP exists to answer an objection. APF is the strongest comparator in
the study, so "you just got lucky with the parameters" is the obvious
challenge. Sweeping k_rep over five values and rep_radius over four, at every
density, shows the reported configuration (1.5, 2) sits on a broad plateau
rather than a lucky peak: coverage varies by under two points across the
centre of the grid and only degrades sharply at rep_radius=4, where the
repulsion radius approaches the corridor width between buildings.

That matters for how the comparison reads. A baseline tuned to a knife-edge
would be a weak comparator; one sitting on a plateau is a fair one.

THE ABSOLUTE-CELL SECTION is the earlier, smaller version of what
absolute_cells_50.py does properly. It runs 20 seeds and actually re-simulates
to get the counts. The later script derives the same quantity from stored
coverage and regenerated layouts across all 50 seeds, which is both cheaper
and consistent with every other reported result.

The interpret() function at the bottom is deliberately written to let the
data choose between two explanations rather than assuming one. It spells out
what pattern would support each, then reports which appeared. As it happens,
percentage rises while absolute cells fall -- the denominator effect.

Run:  python3 -m analysis.apf_parameter_sweep      (from code/)
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
for sub in ["", "phase2_boids", "phase3_apf", "phase4_ppo"]:
    p = os.path.join(CODE, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from phase3_apf.model_apf import APFModel
from phase2_boids.model_phase2 import SwarmCoveragePhase2

RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

MAX_STEPS = 700
N_UAVS = 6
DENSITIES = [0.0, 0.1, 0.2, 0.3]

# The reported configuration is (k_rep=1.5, rep_radius=2), mid-grid on both
# axes. Sweeping around it is what shows that choice was not load-bearing.
K_REP_VALUES = [0.5, 1.0, 1.5, 2.5, 4.0]
R_REP_VALUES = [1, 2, 3, 4]

SWEEP_SEEDS = 8          # 20 cells x 4 densities x 8 seeds is already 640 runs
CELL_SEEDS = 20


def run_apf(density, seed, k_rep=None, r_rep=None):
    """One APF episode, optionally overriding the repulsion parameters.

    The overrides are applied after construction, reaching into the agents
    directly. Slightly blunt, but it keeps the model constructor identical
    to the one every reported result uses -- the sweep cannot accidentally
    change anything else about the controller.
    """
    model = APFModel(n_agents=N_UAVS, obstacle_density=density,
                     obstacle_seed=seed)
    if k_rep is not None or r_rep is not None:
        for a in model.agents:
            if hasattr(a, "k_rep"):
                if k_rep is not None:
                    a.k_rep = k_rep
                if r_rep is not None:
                    a.rep_radius = r_rep
    for _ in range(MAX_STEPS):
        model.step()
    grid = model.coverage_grid
    covered = int(np.sum(grid == 1))
    free = int(np.sum(grid >= 0))
    return 100.0 * covered / max(1, free), covered, free


def run_boids(density, seed):
    """One Boids episode, for the absolute-cell comparison."""
    model = SwarmCoveragePhase2(n_agents=N_UAVS, obstacle_density=density,
                                obstacle_seed=seed)
    for _ in range(MAX_STEPS):
        model.step()
    grid = model.coverage_grid
    covered = int(np.sum(grid == 1))
    free = int(np.sum(grid >= 0))
    return 100.0 * covered / max(1, free), covered, free


def parameter_sweep():
    """The 5 x 4 grid of repulsion parameters, at every density."""
    out = {}
    print("=" * 70)
    print("APF PARAMETER SWEEP  (coverage %, mean over "
          f"{SWEEP_SEEDS} seeds)")
    print("=" * 70)
    for d in DENSITIES:
        print(f"\nDensity {int(d*100)}%")
        header = "  k_rep \\ r_rep " + "".join(f"{r:>9}" for r in R_REP_VALUES)
        print(header)
        out[d] = {}
        for k in K_REP_VALUES:
            row = []
            for r in R_REP_VALUES:
                covs = [run_apf(d, s, k_rep=k, r_rep=r)[0]
                        for s in range(SWEEP_SEEDS)]
                out[d][(k, r)] = covs
                row.append(np.mean(covs))
            marker = "  <- reported" if k == 1.5 else ""
            print(f"  {k:>6.1f}        "
                  + "".join(f"{v:>9.2f}" for v in row) + marker)
    return out


def absolute_cells():
    """Cells covered against coverage percentage. Superseded.

    absolute_cells_50.py does this properly: all 50 seeds, derived from
    stored results rather than re-simulated, and consistent with every other
    reported number. This 20-seed version is what first surfaced the effect.
    """
    out = {"APF": {}, "Boids": {}}
    print("\n" + "=" * 70)
    print("ABSOLUTE CELLS COVERED vs PERCENTAGE "
          f"(mean over {CELL_SEEDS} seeds)")
    print("=" * 70)
    print(f"{'Density':>8} {'Method':<7} {'Free cells':>11} "
          f"{'Cells covered':>14} {'Coverage %':>11}")
    for d in DENSITIES:
        for name, fn in [("Boids", run_boids), ("APF", run_apf)]:
            recs = [fn(d, s) for s in range(CELL_SEEDS)]
            pct = np.mean([r[0] for r in recs])
            cov = np.mean([r[1] for r in recs])
            free = np.mean([r[2] for r in recs])
            out[name][d] = {"pct": [r[0] for r in recs],
                            "cells": [r[1] for r in recs],
                            "free": [r[2] for r in recs]}
            print(f"{int(d*100):>7}% {name:<7} {free:>11.1f} "
                  f"{cov:>14.1f} {pct:>11.2f}")
    return out


def interpret(cells):
    """State both candidate explanations, then report which the data support.

    (a) obstacles genuinely help search -- percentage AND absolute cells rise
    (b) the denominator shrinks faster -- percentage rises, absolute falls

    Writing the test this way round matters. Deciding what each outcome would
    mean before looking at the answer is the difference between a test and a
    rationalisation.
    """
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    for name in ["APF", "Boids"]:
        c0 = np.mean(cells[name][0.0]["cells"])
        c30 = np.mean(cells[name][0.3]["cells"])
        p0 = np.mean(cells[name][0.0]["pct"])
        p30 = np.mean(cells[name][0.3]["pct"])
        print(f"\n{name}: 0% -> 30% obstacles")
        print(f"  percentage coverage {p0:.2f}% -> {p30:.2f}% "
              f"({p30-p0:+.2f} pp)")
        print(f"  absolute cells      {c0:.1f} -> {c30:.1f} "
              f"({c30-c0:+.1f} cells)")
        if p30 > p0 and c30 < c0:
            print("  => percentage rises while absolute cells FALL.")
            print("     The rise is a denominator effect, not improved")
            print("     search. Explanation (b).")
        elif p30 > p0 and c30 >= c0:
            print("  => percentage and absolute cells both rise.")
            print("     Genuine improvement in search efficiency.")
            print("     Explanation (a).")
        else:
            print("  => percentage falls; no ambiguity to resolve.")


if __name__ == "__main__":
    sweep = parameter_sweep()
    np.save(os.path.join(RESULTS, "apf_param_sweep.npy"), sweep)

    cells = absolute_cells()
    np.save(os.path.join(RESULTS, "absolute_cells.npy"), cells)

    interpret(cells)
    print(f"\nSaved to {RESULTS}")
