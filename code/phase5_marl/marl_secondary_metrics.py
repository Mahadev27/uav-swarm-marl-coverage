"""
Fills in the MARL row of Table IV: path length, load balance, near-misses and
per-step cost.

That table originally compared only Boids, APF and the hybrid — it omitted
the method the paper proposes. Filling the gap is not simply a matter of
reading the environment's own counters, because three of them measure
something different from what the classical controllers measure. Mixing the
two would produce a row that looks comparable and is not.

The definitions below are taken verbatim from analysis/extended_metrics.py:

  PATH LENGTH   Euclidean step distance summed along each agent's trajectory,
                converted to metres at 3 m per cell. The environment's own
                counter is in cells.

  LOAD BALANCE  Gini over the number of DISTINCT cells each agent occupied,
                whether or not it got there first. The environment credits
                only first visits, which is a different quantity and would
                overstate imbalance.

  NEAR-MISSES   Pairs within Euclidean radius one, sampled before each step.
                The environment counts Chebyshev neighbours after moving,
                which also admits diagonals.

Per-step cost is wall-clock time for the episode loop divided by timesteps,
so it includes policy inference. The published timings for the other three
controllers were taken on different hardware, so the `timing` stage
re-measures all four on one machine; only the relative ordering means
anything.

Usage:
    python3 phase5_marl/marl_secondary_metrics.py [stage] [seconds]
    stage = metrics (default) | timing | report
"""

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
for _sub in ["", "phase2_boids", "phase3_apf", "phase5_marl"]:
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from stable_baselines3 import PPO
from phase5_marl.marl_env import CoverageWorld

torch.set_num_threads(1)

RESULTS = os.path.join(_HERE, "results")
DENSITIES = [0.0, 0.1, 0.2, 0.3]
N_SEEDS = 50                     # matches extended_metrics.py
MAX_STEPS = 700
NUM_UAVS = 6
COLLISION_RADIUS = 1             # cells, Euclidean
CELL_SIZE_M = 3.0

OUT = os.path.join(RESULTS, "marl_secondary.json")
TIMING = os.path.join(RESULTS, "timing_same_machine.json")


def gini_coefficient(values):
    """Identical to analysis/extended_metrics.py. 0 even, 1 maximally unequal."""
    values = np.array(sorted(values), dtype=float)
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * values)) / (n * np.sum(values))
                 - (n + 1) / n)


def marl_episode(policy, density, seed, time_it=False):
    """One episode measured under the classical definitions, not the env's."""
    world = CoverageWorld(density, seed)
    policy.set_random_seed(int(90000 + seed))
    obs = world.observations()
    traj = {i: [tuple(world.positions[i])] for i in range(NUM_UAVS)}
    collisions = 0
    done = False
    t0 = time.perf_counter()
    while not done:
        # Near-misses: before the step, Euclidean radius. Diagonal pairs are
        # excluded because sqrt(2) > 1.
        pos = [tuple(p) for p in world.positions]
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dr = pos[i][0] - pos[j][0]
                dc = pos[i][1] - pos[j][1]
                if (dr * dr + dc * dc) ** 0.5 <= COLLISION_RADIUS:
                    collisions += 1
        actions, _ = policy.predict(obs, deterministic=False)
        _, done = world.step(np.atleast_1d(actions))
        obs = world.observations()
        for i in range(NUM_UAVS):
            traj[i].append(tuple(world.positions[i]))
    elapsed = time.perf_counter() - t0

    # Path length in metres, and distinct cells occupied per agent. The
    # set() is what makes this "distinct cells occupied" rather than "cells
    # first visited" -- the distinction that matters for the Gini.
    path_lengths, cells_per_uav = [], []
    for i in range(NUM_UAVS):
        t = traj[i]
        length = sum(((t[k][0] - t[k - 1][0]) ** 2
                      + (t[k][1] - t[k - 1][1]) ** 2) ** 0.5
                     for k in range(1, len(t)))
        path_lengths.append(length * CELL_SIZE_M)
        cells_per_uav.append(len(set(t)))

    out = {
        "coverage": world.coverage(),
        "total_path_length": float(sum(path_lengths)),
        "cells_per_uav": cells_per_uav,
        "collision_events": int(collisions),
    }
    if time_it:
        out["ms_per_step"] = 1000.0 * elapsed / max(1, world.timestep)
    return out


def stage_metrics(budget):
    """Run 50 seeds per density, resuming from whatever is already stored."""
    store = json.load(open(OUT)) if os.path.exists(OUT) else {}
    deadline = time.time() + budget
    for density in DENSITIES:
        key = str(int(round(density * 100)))
        runs = store.get(key, [])
        if len(runs) >= N_SEEDS:
            continue
        path = os.path.join(RESULTS, f"marl_d{key}.zip")
        model = PPO.load(path, device="cpu")
        for s in range(len(runs), N_SEEDS):
            runs.append(marl_episode(model, density, s))
            if time.time() > deadline:
                break
        store[key] = runs
        json.dump(store, open(OUT, "w"))
        print(f"  density {key}%: {len(runs)}/{N_SEEDS}", flush=True)
        if time.time() > deadline:
            return False
    print("metrics complete", flush=True)
    return True


def stage_timing(budget):
    """Re-time all four controllers on this machine, 3 seeds each.

    The published per-step costs were measured on different hardware at
    different times, which makes them uncomparable. This pass runs all four
    here and now, so only the ordering between them is claimed.
    """
    from phase2_boids.model_phase2 import SwarmCoveragePhase2
    from phase3_apf.model_apf import APFModel

    store = json.load(open(TIMING)) if os.path.exists(TIMING) else {}
    deadline = time.time() + budget
    for density in DENSITIES:
        key = str(int(round(density * 100)))
        rec = store.get(key, {})
        for name, cls in [("Boids", SwarmCoveragePhase2), ("APF", APFModel)]:
            if name in rec:
                continue
            ms = []
            for s in range(3):
                m = cls(n_agents=NUM_UAVS, obstacle_density=density,
                        obstacle_seed=s)
                t0 = time.perf_counter()
                for _ in range(MAX_STEPS):
                    m.step()
                ms.append(1000.0 * (time.perf_counter() - t0) / MAX_STEPS)
            rec[name] = float(np.mean(ms))
            store[key] = rec
            json.dump(store, open(TIMING, "w"))
            print(f"  {key}% {name}: {rec[name]:.3f} ms/step", flush=True)
            if time.time() > deadline:
                return False
        if "Hybrid" not in rec:
            from phase4_ppo.uav_coverage_env import UAVCoverageEnv
            hp = os.path.join(_CODE, "phase4_ppo/results",
                              f"ppo_model_d{key}.zip")
            hm = PPO.load(hp, device="cpu")
            ms = []
            for s in range(3):
                env = UAVCoverageEnv(obstacle_density=density)
                obs, _ = env.reset(seed=s)
                n, t0 = 0, time.perf_counter()
                done = False
                while not done:
                    a, _ = hm.predict(obs, deterministic=False)
                    obs, _, term, trunc, _ = env.step(a)
                    done = term or trunc
                    n += 1
                ms.append(1000.0 * (time.perf_counter() - t0) / max(1, n))
            rec["Hybrid"] = float(np.mean(ms))
            store[key] = rec
            json.dump(store, open(TIMING, "w"))
            print(f"  {key}% Hybrid: {rec['Hybrid']:.3f} ms/step", flush=True)
            if time.time() > deadline:
                return False
        if "MARL" not in rec:
            model = PPO.load(os.path.join(RESULTS, f"marl_d{key}.zip"),
                             device="cpu")
            ms = [marl_episode(model, density, s, time_it=True)["ms_per_step"]
                  for s in range(3)]
            rec["MARL"] = float(np.mean(ms))
            store[key] = rec
            json.dump(store, open(TIMING, "w"))
            print(f"  {key}% MARL: {rec['MARL']:.3f} ms/step", flush=True)
            if time.time() > deadline:
                return False
    print("timing complete", flush=True)
    return True


def stage_report():
    """Print the finished table. Reads stored results; runs nothing."""
    store = json.load(open(OUT))
    print(f"\nMARL secondary measures, mean over {N_SEEDS} seeds, "
          "classical definitions\n")
    for label, fn, fmt in [
        ("Path length (m)", lambda r: r["total_path_length"], "{:>10.0f}"),
        ("Gini", lambda r: gini_coefficient(r["cells_per_uav"]), "{:>10.3f}"),
        ("Near-misses", lambda r: r["collision_events"], "{:>10.1f}"),
        ("Coverage %", lambda r: r["coverage"], "{:>10.2f}"),
    ]:
        vals = [np.mean([fn(r) for r in store[str(int(round(d * 100)))]])
                for d in DENSITIES]
        print(f"{label:<17}" + "".join(fmt.format(v) for v in vals))
    if os.path.exists(TIMING):
        t = json.load(open(TIMING))
        print("\nms/step, all four re-measured on one machine, 3 seeds\n")
        for name in ["Boids", "APF", "Hybrid", "MARL"]:
            row = [t[str(int(round(d * 100)))].get(name, float("nan"))
                   for d in DENSITIES]
            print(f"{name:<17}" + "".join(f"{v:>10.3f}" for v in row))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "metrics"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 35.0
    if stage == "metrics":
        stage_metrics(secs)
    elif stage == "timing":
        stage_timing(secs)
    else:
        stage_report()
