"""
The secondary measures for Boids, APF and the hybrid: path length, load
balance, near-misses and coverage over time.

Coverage alone is a poor guide to whether a controller would work in the
field. A swarm can hit 95% by flying every agent over every cell, burning
battery it does not have and nearly colliding constantly. These are the
measures that separate a usable controller from a merely high-scoring one.

THIS FILE DEFINES THE MEASURES. Everything else in the project follows the
definitions here, and where something could not, that is documented as a
defect. Three of them admit more than one reasonable definition, so they are
pinned down explicitly:

  PATH LENGTH   Euclidean step distance summed along each agent's trajectory,
                converted to metres at 3 m per cell. A diagonal move counts
                sqrt(2), not 1. Stands in for energy.

  LOAD BALANCE  Gini coefficient over the number of DISTINCT cells each agent
                occupied -- whether or not it was first there. Not "cells
                first visited", which is a different quantity and overstates
                imbalance. 0 is perfectly even work, 1 maximally unequal.

  NEAR-MISSES   Pairs within Euclidean distance 1, sampled BEFORE each step.
                Euclidean rather than Chebyshev means diagonal neighbours do
                not count, since sqrt(2) > 1. Before rather than after the
                move.

That last one caused a real problem. The MARL environment counts near-misses
its own way -- Chebyshev, after moving -- so the MARL figures were initially
incomparable with these. phase5_marl/marl_secondary_metrics.py and
marl_swarm_size_metrics.py exist to recompute them under the definitions
above.

A near-miss is a proximity proxy and not an observed collision. The grid
gives a vehicle no body, so what this counts is how often a real airframe
would have had to give way.

Writes extended_metrics.npy, read by analysis/make_report_figures.py.

Run:  python3 -m analysis.extended_metrics      (from code/)
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
N_SEEDS     = 50
COLLISION_RADIUS = 1     # cells, Euclidean: diagonals excluded
CELL_SIZE_M = 3.0        # metres per cell, for path length

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "../../figures/extended_analysis")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def run_mesa_with_metrics(ModelClass, density, seed):
    """One episode of a Mesa controller, instrumented.

    Works for both Boids and APF: they share the same model interface, so
    the same instrumentation applies to each without special-casing.
    """
    np.random.seed(seed)
    model = ModelClass(n_agents=NUM_UAVS, obstacle_density=density,
                       obstacle_seed=seed)

    positions_history = {i: [] for i in range(NUM_UAVS)}
    coverage_over_time = []
    collision_events    = 0

    for step in range(MAX_STEPS):
        uavs = [a for a in model.agents if hasattr(a, 'cells_visited')]
        for i, agent in enumerate(uavs[:NUM_UAVS]):
            positions_history[i].append(agent.pos)

        # Near-misses sampled BEFORE model.step(), so this counts proximity
        # as the agents stood at the start of the tick.
        current_positions = [uavs[i].pos for i in range(min(NUM_UAVS, len(uavs)))]
        for i in range(len(current_positions)):
            for j in range(i+1, len(current_positions)):
                dr = current_positions[i][0] - current_positions[j][0]
                dc = current_positions[i][1] - current_positions[j][1]
                dist = (dr**2 + dc**2) ** 0.5
                if dist <= COLLISION_RADIUS:
                    collision_events += 1

        model.step()
        cov = model.datacollector.get_model_vars_dataframe()["Coverage_%"].iloc[-1] \
              if len(model.datacollector.get_model_vars_dataframe()) > 0 else 0.0
        coverage_over_time.append(cov)

    path_lengths = []
    cells_per_uav = []
    for i in range(NUM_UAVS):
        traj = positions_history[i]
        length = 0.0
        for k in range(1, len(traj)):
            dr = traj[k][0] - traj[k-1][0]
            dc = traj[k][1] - traj[k-1][1]
            length += (dr**2 + dc**2) ** 0.5
        path_lengths.append(length * CELL_SIZE_M)
        # set() makes this DISTINCT cells occupied, not cells first visited.
        # The distinction is what the Gini is computed over.
        cells_per_uav.append(len(set(traj)))

    final_coverage = coverage_over_time[-1] if coverage_over_time else 0.0

    return {
        "coverage_curve":   coverage_over_time,
        "final_coverage":   final_coverage,
        "path_lengths":     path_lengths,
        "total_path_length": sum(path_lengths),
        "cells_per_uav":    cells_per_uav,
        "collision_events": collision_events,
    }


def run_ppo_with_metrics(density, seed):
    """Same instrumentation for the PPO hybrid.

    Separate from the Mesa version because the hybrid is a Gymnasium env with
    a different interface, not because the measures differ. They do not: the
    definitions are identical, which is what keeps the four columns
    comparable.
    """
    try:
        from stable_baselines3 import PPO
        from phase4_ppo.uav_coverage_env import UAVCoverageEnv
    except ImportError:
        return None

    pct = int(density * 100)
    model_path = os.path.join(
        os.path.dirname(__file__), f"../phase4_ppo/results/ppo_model_d{pct}.zip")
    if not os.path.exists(model_path):
        return None

    model = PPO.load(model_path)
    env   = UAVCoverageEnv(obstacle_density=density, seed=seed)
    obs, _ = env.reset(seed=seed)

    positions_history   = {i: [tuple(env.positions[i])] for i in range(NUM_UAVS)}
    coverage_over_time   = []
    collision_events     = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        for i in range(NUM_UAVS):
            positions_history[i].append(tuple(env.positions[i]))

        current_positions = [tuple(env.positions[i]) for i in range(NUM_UAVS)]
        for i in range(len(current_positions)):
            for j in range(i+1, len(current_positions)):
                dr = current_positions[i][0] - current_positions[j][0]
                dc = current_positions[i][1] - current_positions[j][1]
                dist = (dr**2 + dc**2) ** 0.5
                if dist <= COLLISION_RADIUS:
                    collision_events += 1

        coverage_over_time.append(info["coverage"])
        done = terminated or truncated

    path_lengths  = []
    cells_per_uav = []
    for i in range(NUM_UAVS):
        traj = positions_history[i]
        length = 0.0
        for k in range(1, len(traj)):
            dr = traj[k][0] - traj[k-1][0]
            dc = traj[k][1] - traj[k-1][1]
            length += (dr**2 + dc**2) ** 0.5
        path_lengths.append(length * CELL_SIZE_M)
        cells_per_uav.append(len(set(traj)))

    return {
        "coverage_curve":    coverage_over_time,
        "final_coverage":    coverage_over_time[-1] if coverage_over_time else 0.0,
        "path_lengths":      path_lengths,
        "total_path_length": sum(path_lengths),
        "cells_per_uav":     cells_per_uav,
        "collision_events":  collision_events,
    }


def gini_coefficient(values):
    """Gini over per-agent workload. 0 perfectly even, 1 maximally unequal.

    Borrowed from economics, where it measures income inequality. Here the
    "income" is distinct cells swept. It catches the hybrid's pathology
    cleanly: its six agents contribute 0.4%, 19.6%, 19.7%, 20.9%, 19.6% and
    19.8%, because the one learned agent barely participates.
    """
    values = np.array(sorted(values), dtype=float)
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0.0
    cumsum = np.cumsum(values)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def collect_all_metrics():
    algorithms = {
        "Boids": lambda d, s: run_mesa_with_metrics(SwarmCoveragePhase2, d, s),
        "APF":   lambda d, s: run_mesa_with_metrics(APFModel, d, s),
        "PPO":   lambda d, s: run_ppo_with_metrics(d, s),
    }

    all_results = {algo: {d: [] for d in DENSITIES} for algo in algorithms}

    for algo_name, run_fn in algorithms.items():
        print(f"\n{'='*60}")
        print(f"  Running {algo_name} — {N_SEEDS} seeds x {len(DENSITIES)} densities")
        print(f"{'='*60}")
        for d in DENSITIES:
            t0 = time.time()
            for seed in range(N_SEEDS):
                result = run_fn(d, seed)
                if result is not None:
                    all_results[algo_name][d].append(result)
            elapsed = time.time() - t0
            n_done = len(all_results[algo_name][d])
            if n_done > 0:
                mean_cov  = np.mean([r["final_coverage"] for r in all_results[algo_name][d]])
                mean_path = np.mean([r["total_path_length"] for r in all_results[algo_name][d]])
                mean_coll = np.mean([r["collision_events"] for r in all_results[algo_name][d]])
                print(f"  {int(d*100):>2}% | n={n_done:>3} | "
                      f"coverage={mean_cov:.1f}% | "
                      f"path={mean_path:.0f}m | "
                      f"collisions={mean_coll:.1f} | "
                      f"time={elapsed:.1f}s")

    np.save(os.path.join(RESULTS_DIR, "extended_metrics.npy"), all_results)
    print(f"\nSaved -> {RESULTS_DIR}/extended_metrics.npy")
    return all_results


def plot_coverage_vs_time(all_results):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    colors = {"Boids": "#457B9D", "APF": "#E63946", "PPO": "#2A9D8F"}

    for ax, d in zip(axes, DENSITIES):
        for algo in ["Boids", "APF", "PPO"]:
            results = all_results[algo][d]
            if not results:
                continue
            curves = [r["coverage_curve"] for r in results if len(r["coverage_curve"]) > 0]
            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            curves  = np.array([c[:min_len] for c in curves])
            mean_c  = np.mean(curves, axis=0)
            std_c   = np.std(curves, axis=0)
            ts = np.arange(min_len)
            ax.plot(ts, mean_c, color=colors[algo], linewidth=2, label=algo)
            ax.fill_between(ts, mean_c-std_c, mean_c+std_c,
                             color=colors[algo], alpha=0.15)

        ax.axhline(y=90, color="black", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_title(f"{int(d*100)}% obstacles", fontsize=12, fontweight="bold")
        ax.set_xlabel("Timestep", fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Coverage (%)", fontsize=11)
    axes[0].legend(fontsize=10, loc="lower right")
    fig.suptitle("Coverage vs Time — All Algorithms, All Densities",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/coverage_vs_time_all.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: coverage_vs_time_all.png")


def plot_path_length(all_results):
    fig, ax = plt.subplots(figsize=(9, 6))
    algos  = ["Boids", "APF", "PPO"]
    colors = {"Boids": "#457B9D", "APF": "#E63946", "PPO": "#2A9D8F"}
    x = np.arange(len(DENSITIES))
    width = 0.25

    for i, algo in enumerate(algos):
        means, stds = [], []
        for d in DENSITIES:
            results = all_results[algo][d]
            if results:
                paths = [r["total_path_length"] for r in results]
                means.append(np.mean(paths))
                stds.append(np.std(paths))
            else:
                means.append(0)
                stds.append(0)
        ax.bar(x + (i-1)*width, means, width, yerr=stds, capsize=4,
               label=algo, color=colors[algo], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(d*100)}%" for d in DENSITIES])
    ax.set_xlabel("Obstacle Density", fontsize=11)
    ax.set_ylabel("Total Swarm Path Length (metres)", fontsize=11)
    ax.set_title("Path Length Comparison — Energy Consumption Proxy\n"
                 "(Total distance travelled by all 6 UAVs combined)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/path_length_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: path_length_comparison.png")


def plot_load_balance(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    algos  = ["Boids", "APF", "PPO"]
    colors = {"Boids": "#457B9D", "APF": "#E63946", "PPO": "#2A9D8F"}

    ax1 = axes[0]
    x = np.arange(len(DENSITIES))
    width = 0.25
    for i, algo in enumerate(algos):
        ginis = []
        for d in DENSITIES:
            results = all_results[algo][d]
            if results:
                g = [gini_coefficient(r["cells_per_uav"]) for r in results]
                ginis.append(np.mean(g))
            else:
                ginis.append(0)
        ax1.bar(x + (i-1)*width, ginis, width, label=algo, color=colors[algo], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(d*100)}%" for d in DENSITIES])
    ax1.set_xlabel("Obstacle Density", fontsize=11)
    ax1.set_ylabel("Gini Coefficient (0 = balanced, 1 = imbalanced)", fontsize=10)
    ax1.set_title("Load Balance Across UAVs", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = axes[1]
    d_example = 0.20
    bar_width = 0.25
    uav_x = np.arange(NUM_UAVS)
    for i, algo in enumerate(algos):
        results = all_results[algo][d_example]
        if not results:
            continue
        avg_cells = np.mean([r["cells_per_uav"] for r in results], axis=0)
        total = np.sum(avg_cells)
        pct_share = avg_cells / total * 100 if total > 0 else avg_cells
        ax2.bar(uav_x + (i-1)*bar_width, pct_share, bar_width,
                label=algo, color=colors[algo], alpha=0.85)
    ax2.set_xticks(uav_x)
    ax2.set_xticklabels([f"UAV {i+1}" for i in range(NUM_UAVS)])
    ax2.set_ylabel("% of Total Coverage Contributed", fontsize=10)
    ax2.set_title(f"Per-UAV Coverage Share at {int(d_example*100)}% Obstacles",
                 fontsize=12, fontweight="bold")
    ax2.axhline(y=100/NUM_UAVS, color="black", linestyle=":", alpha=0.5,
                label=f"Perfect balance ({100/NUM_UAVS:.1f}%)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/load_balance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: load_balance.png")


def plot_collisions(all_results):
    fig, ax = plt.subplots(figsize=(9, 6))
    algos  = ["Boids", "APF", "PPO"]
    colors = {"Boids": "#457B9D", "APF": "#E63946", "PPO": "#2A9D8F"}
    x = np.arange(len(DENSITIES))
    width = 0.25

    for i, algo in enumerate(algos):
        means, stds = [], []
        for d in DENSITIES:
            results = all_results[algo][d]
            if results:
                colls = [r["collision_events"] for r in results]
                means.append(np.mean(colls))
                stds.append(np.std(colls))
            else:
                means.append(0)
                stds.append(0)
        ax.bar(x + (i-1)*width, means, width, yerr=stds, capsize=4,
               label=algo, color=colors[algo], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(d*100)}%" for d in DENSITIES])
    ax.set_xlabel("Obstacle Density", fontsize=11)
    ax.set_ylabel(f"Mean Near-Miss Events (<= {COLLISION_RADIUS} cell separation)", fontsize=10)
    ax.set_title("Inter-UAV Collision Risk Comparison\n"
                 "(Number of timesteps with unsafe agent separation)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/collision_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: collision_comparison.png")


def print_summary_table(all_results):
    print(f"\n{'='*90}")
    print("  EXTENDED METRICS SUMMARY (mean across seeds)")
    print(f"{'='*90}")
    header = f"{'Density':>8} {'Algo':>8} {'Coverage%':>10} {'Path(m)':>10} {'Gini':>8} {'Collisions':>11}"
    print(header)
    print("-"*90)
    for d in DENSITIES:
        for algo in ["Boids", "APF", "PPO"]:
            results = all_results[algo][d]
            if not results:
                continue
            cov  = np.mean([r["final_coverage"] for r in results])
            path = np.mean([r["total_path_length"] for r in results])
            gini = np.mean([gini_coefficient(r["cells_per_uav"]) for r in results])
            coll = np.mean([r["collision_events"] for r in results])
            print(f"{int(d*100):>7}% {algo:>8} {cov:>9.1f}% {path:>9.0f}m "
                  f"{gini:>8.3f} {coll:>11.1f}")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    all_results = collect_all_metrics()
    print_summary_table(all_results)

    print("\nGenerating figures...")
    plot_coverage_vs_time(all_results)
    plot_path_length(all_results)
    plot_load_balance(all_results)
    plot_collisions(all_results)

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("\nFigures generated:")
    print("  coverage_vs_time_all.png   (Item 1: coverage vs time)")
    print("  path_length_comparison.png (Item 2: energy proxy)")
    print("  load_balance.png           (Item 3: load balancing)")
    print("  collision_comparison.png   (Item 5: collision analysis)")
