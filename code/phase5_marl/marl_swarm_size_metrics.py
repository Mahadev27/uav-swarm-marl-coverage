"""
Recomputes the MARL swarm-size near-miss column under the same definition the
classical controllers use.

Table V of the report gives coverage and near-misses for all four controllers
at 3, 6, 9 and 12 agents. The Boids, APF and hybrid columns all count a
near-miss identically, in analysis/extended_metrics.py,
analysis/uav_count_sensitivity.py and analysis/uav_count_sensitivity_ppo.py:

    Euclidean distance <= 1 cell, sampled BEFORE each step.

The MARL column originally did not. It came from the environment's own
counter in marl_env.CoverageWorld.step(), which uses

    Chebyshev distance <= 1 cell, counted AFTER moving,

so it also admits diagonal pairs and counts strictly more events. That made
the MARL column incomparable with the three beside it, and inconsistent with
the MARL row of Table IV, where the classical definition was already used.

The direction of the error flattered the classical controllers rather than
this work — it inflated MARL's apparent collision risk by about 45% — but a
table whose columns are measured two different ways cannot be read at all.

Coverage is recomputed in the same pass and reproduces the published values
exactly, which is the check that this episode loop matches the original.

Run:  python3 phase5_marl/marl_swarm_size_metrics.py     (from code/)
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
for _sub in ["", "phase2_boids", "phase5_marl"]:
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from stable_baselines3 import PPO
from phase5_marl.marl_env import CoverageWorld

torch.set_num_threads(1)

RESULTS = os.path.join(_HERE, "results")
OUT = os.path.join(RESULTS, "marl_swarm_size_classical.json")

DENSITY = 0.2
SIZES = [3, 6, 9, 12]
N_SEEDS = 15                 # matches uav_count_sensitivity.py
COLLISION_RADIUS = 1         # cells, Euclidean, as in extended_metrics.py


def episode(policy, n_agents, seed):
    """One episode, counting near-misses the classical way.

    Returns both counts so the two definitions can be compared directly.
    """
    world = CoverageWorld(DENSITY, seed, n_agents=n_agents)
    policy.set_random_seed(int(90000 + seed))
    obs = world.observations()
    near = 0
    done = False
    while not done:
        # Sampled before the step, Euclidean radius: the same way the
        # classical controllers are measured. Note this excludes diagonal
        # neighbours, since sqrt(2) > 1.
        pos = [tuple(p) for p in world.positions]
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dr = pos[i][0] - pos[j][0]
                dc = pos[i][1] - pos[j][1]
                if (dr * dr + dc * dc) ** 0.5 <= COLLISION_RADIUS:
                    near += 1
        actions, _ = policy.predict(obs, deterministic=False)
        _, done = world.step(np.atleast_1d(actions))
        obs = world.observations()
    return {"coverage": world.coverage(),
            "near_classical": int(near),
            "near_env_chebyshev": int(world.near_misses)}


def main():
    model = PPO.load(os.path.join(RESULTS, f"marl_d{int(DENSITY*100)}.zip"),
                     device="cpu")
    store = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for n in SIZES:
        key = str(n)
        runs = store.get(key, [])
        for s in range(len(runs), N_SEEDS):
            runs.append(episode(model, n, s))
        store[key] = runs
        # Written after every swarm size, so an interrupted run resumes
        # from the size it reached rather than starting over.
        json.dump(store, open(OUT, "w"))
        print(f"  n={n}: {len(runs)}/{N_SEEDS} done", flush=True)

    print(f"\nMARL at 20% density, {N_SEEDS} seeds\n")
    print(f"{'N':>3} {'coverage %':>11} {'near (classical)':>18} "
          f"{'near (env, cheby)':>19}")
    for n in SIZES:
        r = store[str(n)]
        cov = np.mean([x["coverage"] for x in r])
        cl = np.mean([x["near_classical"] for x in r])
        ch = np.mean([x["near_env_chebyshev"] for x in r])
        print(f"{n:>3} {cov:>11.1f} {cl:>18.1f} {ch:>19.1f}")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
