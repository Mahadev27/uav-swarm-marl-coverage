"""
How Boids and APF scale with swarm size: 3, 6, 9 and 12 agents at 20%
obstacle density.

Two things come out of this, and the second is the more useful.

The obvious one is that more agents cover more ground, with returns falling
off past nine. The arena saturates and any reasonable controller approaches
the ceiling.

The less obvious one is that six agents sits on the steep part of the curve,
which is exactly where the choice of controller matters most. At three agents
there is no redundancy to hide a poor sweep; at twelve, the arena is small
enough that everything works. Six is where coordination differences are
amplified -- which is why the main experiment uses six.

Near-misses also rise four- to sixfold from six to twelve agents, so swarm
size buys coverage at a real cost in proximity risk.

Near-misses here use the classical definition: Euclidean radius 1, sampled
before each step. Same as extended_metrics.py, so the columns are comparable.
The MARL equivalent is in phase5_marl/marl_swarm_size_metrics.py.

Run:  python3 -m analysis.uav_count_sensitivity      (from code/)
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../phase2_boids"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../phase3_apf"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase2_boids.model_phase2 import SwarmCoveragePhase2
from phase3_apf.model_apf import APFModel

UAV_COUNTS  = [3, 6, 9, 12]
DENSITY     = 0.20
MAX_STEPS   = 700
N_SEEDS     = 15
CELL_SIZE_M = 3.0
COLLISION_RADIUS = 1     # cells, Euclidean, before each step

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "../../figures/extended_analysis")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_with_n_agents(ModelClass, n_agents, density, seed):
    """One episode at a given swarm size, instrumented.

    Layout and seed are held fixed across swarm sizes, so the only thing
    changing is how many agents are sweeping it.
    """
    np.random.seed(seed)
    t0 = time.perf_counter()
    model = ModelClass(n_agents=n_agents, obstacle_density=density,
                       obstacle_seed=seed)

    positions_history = {i: [] for i in range(n_agents)}
    collision_events  = 0

    for step in range(MAX_STEPS):
        uavs = [a for a in model.agents if hasattr(a, 'cells_visited')]
        for i, agent in enumerate(uavs[:n_agents]):
            positions_history[i].append(agent.pos)

        current_positions = [uavs[i].pos for i in range(min(n_agents, len(uavs)))]
        for i in range(len(current_positions)):
            for j in range(i+1, len(current_positions)):
                dr = current_positions[i][0] - current_positions[j][0]
                dc = current_positions[i][1] - current_positions[j][1]
                if (dr**2 + dc**2) ** 0.5 <= COLLISION_RADIUS:
                    collision_events += 1

        model.step()

    elapsed = time.perf_counter() - t0
    data = model.datacollector.get_model_vars_dataframe()
    final_cov = data["Coverage_%"].iloc[-1] if len(data) > 0 else 0.0

    path_lengths = []
    for i in range(n_agents):
        traj = positions_history[i]
        length = 0.0
        for k in range(1, len(traj)):
            dr = traj[k][0] - traj[k-1][0]
            dc = traj[k][1] - traj[k-1][1]
            length += (dr**2 + dc**2) ** 0.5
        path_lengths.append(length * CELL_SIZE_M)

    return {
        "final_coverage": final_cov,
        "total_path_length": sum(path_lengths),
        "collision_events": collision_events,
        "runtime_s": elapsed,
    }


def run_sensitivity_analysis():
    """Both classical controllers at every swarm size."""
    algorithms = {
        "Boids": SwarmCoveragePhase2,
        "APF":   APFModel,
    }
    all_results = {algo: {n: [] for n in UAV_COUNTS} for algo in algorithms}

    for algo_name, ModelClass in algorithms.items():
        print(f"\n{'='*70}")
        print(f"  {algo_name} — UAV Count Sensitivity (density={int(DENSITY*100)}%)")
        print(f"{'='*70}")
        for n in UAV_COUNTS:
            for seed in range(N_SEEDS):
                result = run_with_n_agents(ModelClass, n, DENSITY, seed)
                all_results[algo_name][n].append(result)
            covs  = [r["final_coverage"] for r in all_results[algo_name][n]]
            paths = [r["total_path_length"] for r in all_results[algo_name][n]]
            colls = [r["collision_events"] for r in all_results[algo_name][n]]
            times = [r["runtime_s"] for r in all_results[algo_name][n]]
            print(f"  N={n:>2} UAVs | coverage={np.mean(covs):.1f}% "
                  f"| path={np.mean(paths):.0f}m "
                  f"| collisions={np.mean(colls):.1f} "
                  f"| runtime={np.mean(times):.2f}s")

    np.save(os.path.join(RESULTS_DIR, "uav_sensitivity.npy"), all_results)
    return all_results


def plot_sensitivity(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    colors = {"Boids": "#457B9D", "APF": "#E63946"}

    metrics = [
        ("final_coverage",    "Coverage (%)",              axes[0,0]),
        ("total_path_length", "Total Path Length (m)",     axes[0,1]),
        ("collision_events",  "Mean Collision Events",     axes[1,0]),
        ("runtime_s",         "Runtime per Episode (s)",   axes[1,1]),
    ]

    for metric_key, ylabel, ax in metrics:
        for algo in ["Boids", "APF"]:
            means = [np.mean([r[metric_key] for r in all_results[algo][n]])
                     for n in UAV_COUNTS]
            stds  = [np.std([r[metric_key] for r in all_results[algo][n]])
                     for n in UAV_COUNTS]
            ax.errorbar(UAV_COUNTS, means, yerr=stds, marker="o",
                        markersize=8, linewidth=2, capsize=5,
                        color=colors[algo], label=algo)
        ax.set_xlabel("Number of UAVs", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(UAV_COUNTS)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0,0].axhline(y=90, color="black", linestyle=":", alpha=0.5)
    axes[0,0].set_title("Coverage vs Swarm Size", fontsize=12, fontweight="bold")
    axes[0,1].set_title("Energy Proxy vs Swarm Size", fontsize=12, fontweight="bold")
    axes[1,0].set_title("Collision Risk vs Swarm Size", fontsize=12, fontweight="bold")
    axes[1,1].set_title("Computational Cost vs Swarm Size", fontsize=12, fontweight="bold")

    fig.suptitle(f"UAV Count Sensitivity Analysis (20% Obstacle Density, {N_SEEDS} seeds)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/uav_count_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: uav_count_sensitivity.png")


if __name__ == "__main__":
    all_results = run_sensitivity_analysis()
    plot_sensitivity(all_results)
    print(f"\n{'='*70}")
    print("  NOTE: PPO excluded from this analysis.")
    print("  PPO models are trained for a fixed swarm size (6 UAVs) and")
    print("  the greedy agents 1-5 assume a fixed observation structure.")
    print("  Testing PPO at other swarm sizes would require retraining")
    print("  separate models — noted as a limitation / future work item.")
    print(f"{'='*70}")
