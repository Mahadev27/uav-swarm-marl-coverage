"""
Evaluates the trained MARL policies and writes the result files the whole
statistical analysis reads.

For each obstacle density and each seed set, this runs the matching policy on
all fifty layouts and stores the per-run record. Output lands in
marl_results_A.npy and marl_results_B.npy, which is what
analysis/statistical_tests.py consumes.

Seed Set A (0-49) is what the policies trained on. Seed Set B (50-99) they
have never seen. Both are evaluated identically, so the difference between
them is the generalisation result.

Evaluation is stochastic, not greedy, because sampling is the deployed
controller's actual behaviour rather than exploration left switched on. See
the note on rollout() in marl_env.py.

Results are written per density-and-set as `_part_*.npy` before being merged,
so an interrupted run does not lose the densities it already finished.

Usage:
    python3 test_marl.py              # everything: 4 densities x 2 seed sets
    python3 test_marl.py 20 A         # just one cell
"""

import os
import sys

import numpy as np
import torch
from stable_baselines3 import PPO

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
for _sub in ["", "phase2_boids", "phase5_marl"]:
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phase5_marl.marl_env import rollout

RESULTS = os.path.join(_HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

DENSITIES = [0.0, 0.1, 0.2, 0.3]
SEED_SETS = {"A": list(range(0, 50)),      # development and training
             "B": list(range(50, 100))}    # held out until the very end

torch.set_num_threads(4)


def evaluate(density, seed_set):
    """Run the policy for this density across all fifty seeds of one set."""
    pct = int(round(density * 100))
    model = PPO.load(os.path.join(RESULTS, f"marl_d{pct}.zip"),
                     device="cpu")
    runs = []
    for s in SEED_SETS[seed_set]:
        r = rollout(model, density, s, deterministic=False)
        runs.append(r)
    covs = [r["coverage"] for r in runs]
    # The >=90 count is the one that matters: the 90% figure is a
    # search-and-rescue planning target, so how often it is met is more
    # informative than the mean alone.
    print(f"  {seed_set} {pct:>2}% | mean {np.mean(covs):6.2f}%  "
          f"sd {np.std(covs, ddof=1):5.2f}  "
          f">=90%: {sum(c >= 90 for c in covs):>2}/50", flush=True)
    return runs


def merge_and_save():
    """Combine the per-density parts into one file per seed set.

    Only writes once all four densities are present, so a partial run never
    produces a results file the analysis would silently read as complete.
    """
    for seed_set in ["A", "B"]:
        out = {}
        for d in DENSITIES:
            pct = int(round(d * 100))
            part = os.path.join(RESULTS, f"_part_{seed_set}_{pct}.npy")
            if os.path.exists(part):
                out[d] = list(np.load(part, allow_pickle=True))
        if len(out) == len(DENSITIES):
            path = os.path.join(RESULTS, f"marl_results_{seed_set}.npy")
            np.save(path, out)
            print(f"  saved -> {path}")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        # Single cell, for resuming or re-running one condition.
        pct, ss = int(sys.argv[1]), sys.argv[2].upper()
        runs = evaluate(pct / 100.0, ss)
        np.save(os.path.join(RESULTS, f"_part_{ss}_{pct}.npy"),
                np.array(runs, dtype=object))
        merge_and_save()
    else:
        for ss in ["A", "B"]:
            for d in DENSITIES:
                runs = evaluate(d, ss)
                np.save(os.path.join(RESULTS,
                                     f"_part_{ss}_{int(d*100)}.npy"),
                        np.array(runs, dtype=object))
        merge_and_save()
