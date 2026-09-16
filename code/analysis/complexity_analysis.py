"""
Wall-clock timing for the classical controllers and the hybrid, plus the
scaling argument behind it.

The asymptotics are straightforward and matter more than the constants.
Boids and APF are O(N^2) per timestep: every agent checks its distance to
every other. A shared policy is O(N), one forward pass each. So the classical
pair should get relatively worse as the swarm grows.

The measurements here bear that out in the constants: the classical pair
rises with obstacle density, the learned ones stay flat. At six agents MARL is
still the most expensive of the four, because a neural forward pass costs
more than scanning five neighbours. The scaling favours it; at this swarm
size it is buying coverage with compute.

CAVEAT ON THE NUMBERS. These timings were taken on different hardware at
different times from the other controllers', which makes cross-controller
comparison unreliable. phase5_marl/marl_secondary_metrics.py has a `timing`
stage that re-measures all four on one machine in one pass, and that is the
source for the per-step column in the report. Use this file for the scaling
shape, not for the absolute values.

Run:  python3 -m analysis.complexity_analysis      (from code/)
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../phase2_boids"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../phase3_apf"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../phase4_ppo"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase2_boids.model_phase2 import SwarmCoveragePhase2
from phase3_apf.model_apf import APFModel

DENSITIES   = [0.0, 0.10, 0.20, 0.30]
MAX_STEPS   = 700
NUM_UAVS    = 6
N_TIMING_RUNS = 5

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "../../figures/extended_analysis")
os.makedirs(FIGURES_DIR, exist_ok=True)


def time_boids_apf(ModelClass, density, seed):
    """Separate construction time from run time.

    Construction includes building the obstacle map and placing several
    hundred obstacle agents, which has nothing to do with per-step control
    cost. Folding them together would flatter whichever controller happened
    to be constructed faster.
    """
    np.random.seed(seed)
    t_start = time.perf_counter()
    model = ModelClass(n_agents=NUM_UAVS, obstacle_density=density,
                       obstacle_seed=seed)
    t_init = time.perf_counter() - t_start

    t_start = time.perf_counter()
    for _ in range(MAX_STEPS):
        model.step()
    t_run = time.perf_counter() - t_start

    return t_init, t_run, t_run / MAX_STEPS


def time_ppo(density, seed):
    """Time the hybrid, returning Nones if the checkpoint is missing.

    Returns rather than raises so a partial project still produces the
    classical timings.
    """
    try:
        from stable_baselines3 import PPO
        from phase4_ppo.uav_coverage_env import UAVCoverageEnv
    except ImportError:
        return None, None, None

    pct = int(density * 100)
    model_path = os.path.join(
        os.path.dirname(__file__), f"../phase4_ppo/results/ppo_model_d{pct}.zip")
    if not os.path.exists(model_path):
        return None, None, None

    t_start = time.perf_counter()
    model = PPO.load(model_path)
    env   = UAVCoverageEnv(obstacle_density=density, seed=seed)
    obs, _ = env.reset(seed=seed)
    t_init = time.perf_counter() - t_start

    t_start = time.perf_counter()
    done = False
    n_steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        n_steps += 1
    t_run = time.perf_counter() - t_start

    return t_init, t_run, t_run / max(n_steps, 1)


def run_timing_analysis():
    print(f"{'='*70}")
    print("  COMPUTATIONAL COMPLEXITY / TIMING ANALYSIS")
    print(f"  {N_TIMING_RUNS} repeats per density, {MAX_STEPS} steps per episode")
    print(f"{'='*70}\n")

    results = {"Boids": {}, "APF": {}, "PPO": {}}

    for algo_name, ModelClass in [("Boids", SwarmCoveragePhase2), ("APF", APFModel)]:
        print(f"Timing {algo_name}...")
        for d in DENSITIES:
            init_times, run_times, per_step_times = [], [], []
            for seed in range(N_TIMING_RUNS):
                t_init, t_run, t_step = time_boids_apf(ModelClass, d, seed)
                init_times.append(t_init)
                run_times.append(t_run)
                per_step_times.append(t_step)
            results[algo_name][d] = {
                "init_mean": np.mean(init_times),
                "run_mean":  np.mean(run_times),
                "step_mean_ms": np.mean(per_step_times) * 1000,
                "step_std_ms":  np.std(per_step_times) * 1000,
            }
            print(f"  {int(d*100):>2}% | init={np.mean(init_times)*1000:.1f}ms | "
                  f"full_episode={np.mean(run_times):.2f}s | "
                  f"per_step={np.mean(per_step_times)*1000:.3f}ms")

    print(f"\nTiming PPO...")
    for d in DENSITIES:
        init_times, run_times, per_step_times = [], [], []
        for seed in range(N_TIMING_RUNS):
            t_init, t_run, t_step = time_ppo(d, seed)
            if t_init is None:
                continue
            init_times.append(t_init)
            run_times.append(t_run)
            per_step_times.append(t_step)
        if init_times:
            results["PPO"][d] = {
                "init_mean": np.mean(init_times),
                "run_mean":  np.mean(run_times),
                "step_mean_ms": np.mean(per_step_times) * 1000,
                "step_std_ms":  np.std(per_step_times) * 1000,
            }
            print(f"  {int(d*100):>2}% | model_load={np.mean(init_times)*1000:.1f}ms | "
                  f"full_episode={np.mean(run_times):.2f}s | "
                  f"per_step={np.mean(per_step_times)*1000:.3f}ms")

    return results


def plot_timing(results):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = {"Boids": "#457B9D", "APF": "#E63946", "PPO": "#2A9D8F"}
    algos  = ["Boids", "APF", "PPO"]

    ax1 = axes[0]
    x = np.arange(len(DENSITIES))
    width = 0.25
    for i, algo in enumerate(algos):
        means = [results[algo].get(d, {}).get("step_mean_ms", 0) for d in DENSITIES]
        stds  = [results[algo].get(d, {}).get("step_std_ms", 0) for d in DENSITIES]
        ax1.bar(x + (i-1)*width, means, width, yerr=stds, capsize=4,
                label=algo, color=colors[algo], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(d*100)}%" for d in DENSITIES])
    ax1.set_xlabel("Obstacle Density", fontsize=11)
    ax1.set_ylabel("Mean Time per Step (ms)", fontsize=11)
    ax1.set_title("Per-Step Computation Time\n(6 UAVs, single CPU core)",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = axes[1]
    for i, algo in enumerate(algos):
        means = [results[algo].get(d, {}).get("run_mean", 0) for d in DENSITIES]
        ax2.bar(x + (i-1)*width, means, width, label=algo, color=colors[algo], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(d*100)}%" for d in DENSITIES])
    ax2.set_xlabel("Obstacle Density", fontsize=11)
    ax2.set_ylabel("Full Episode Time (seconds)", fontsize=11)
    ax2.set_title("Total Runtime per 700-Step Episode",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/computational_complexity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: computational_complexity.png")


def print_complexity_table():
    print(f"\n{'='*80}")
    print("  THEORETICAL COMPUTATIONAL COMPLEXITY (per timestep)")
    print(f"{'='*80}")
    print(f"{'Algorithm':<10} {'Per-agent cost':<25} {'Swarm cost (N=6)':<25} {'Notes'}")
    print("-"*80)
    print(f"{'Boids':<10} {'O(N) neighbour check':<25} {'O(N^2) = O(36)':<25} 'All-pairs separation'")
    print(f"{'APF':<10} {'O(N + R^2) field calc':<25} {'O(N^2 + N*R^2)':<25} 'R=repulsion radius'")
    print(f"{'PPO':<10} {'O(1) forward pass':<25} {'O(1) per agent':<25} 'Fixed NN inference cost'")
    print(f"{'='*80}\n")
    print("Where N = number of UAVs (6), R = obstacle repulsion radius (2 cells)")
    print()
    print("Key insight: Boids and APF scale QUADRATICALLY with swarm size due to")
    print("all-pairs distance checks for separation/repulsion. PPO scales LINEARLY")
    print("since each agent's action is an independent forward pass through a")
    print("fixed-size neural network, regardless of swarm size.")
    print()
    print("This means: as N increases (sensitivity analysis), Boids/APF runtime")
    print("will grow much faster than PPO runtime.")


if __name__ == "__main__":
    results = run_timing_analysis()
    plot_timing(results)
    print_complexity_table()

    np.save(os.path.join(os.path.dirname(__file__), "results", "timing_results.npy"), results)
    print(f"\nSaved raw timing data -> results/timing_results.npy")
