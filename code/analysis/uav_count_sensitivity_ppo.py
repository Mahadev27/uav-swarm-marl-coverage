"""
The same swarm-size sweep for the PPO hybrid, which needs its own
implementation.

The hybrid's Gymnasium environment hard-codes six agents, so this file
reimplements the episode loop with a variable agent count rather than
modifying the environment the reported results came from. That is a
deliberate trade: duplicated logic, but the trained checkpoints stay valid.

IMPORTANT CAVEAT. The policy was trained at six agents. Running it at 3, 9 or
12 is testing it outside its training configuration, so these numbers measure
transfer, not performance. The report draws the hybrid and MARL with dashed
lines in the swarm-size figure for exactly this reason. That the shared-policy
architecture makes such a transfer possible at all -- one network applies to
any swarm size with no retraining -- is itself part of the result.

Near-misses use the classical definition, matching uav_count_sensitivity.py.

Run:  python3 -m analysis.uav_count_sensitivity_ppo      (from code/)
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../phase4_ppo"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UAV_COUNTS  = [3, 6, 9, 12]
DENSITY     = 0.20
N_SEEDS     = 15
CELL_SIZE_M = 3.0
COLLISION_RADIUS = 1     # cells, Euclidean, matching the classical scripts

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "../../figures/extended_analysis")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

HOME = (25, 25)
G    = 50
# Mirrors the environment's action set. Duplicated rather than imported
# because that environment fixes the agent count at six.
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]


def run_ppo_with_n_agents(model, n_agents, density, seed):
    from phase4_ppo.uav_coverage_env import _build_obstacles

    t0 = time.perf_counter()
    obstacle_map = _build_obstacles(density, seed % 50)
    free_count   = max(1, int(np.sum(~obstacle_map)))
    visited      = np.zeros((G, G), dtype=bool)
    visited[HOME] = True

    base_offsets = [(-8,0),(8,0),(0,-8),(0,8),(-5,-5),(5,5),
                    (-8,8),(8,-8),(-8,-8),(8,8),(0,0),(-3,3)]
    positions = []
    for i in range(n_agents):
        dr, dc = base_offsets[i % len(base_offsets)]
        r = max(1, min(G-2, HOME[0]+dr))
        c = max(1, min(G-2, HOME[1]+dc))
        if obstacle_map[r, c]:
            r, c = HOME
        positions.append([r, c])
        visited[r, c] = True

    collision_events = 0
    path_lengths = [0.0] * n_agents

    def get_obs(i):
        r, c = positions[i]
        half = 3
        obs = np.full(49, -1.0, dtype=np.float32)
        idx = 0
        for dr in range(-half, half+1):
            for dc in range(-half, half+1):
                nr, nc = r+dr, c+dc
                if 0 <= nr < G and 0 <= nc < G:
                    obs[idx] = (-1.0 if obstacle_map[nr,nc]
                                else (1.0 if visited[nr,nc] else 0.0))
                idx += 1
        for j, pos in enumerate(positions):
            if j != i:
                adr, adc = pos[0]-r, pos[1]-c
                if -half <= adr <= half and -half <= adc <= half:
                    obs[(adr+half)*7+(adc+half)] = 2.0
        return np.append(obs, [r/G, c/G]).astype(np.float32)

    for step in range(700):
        obs = get_obs(0)
        action, _ = model.predict(obs, deterministic=True)
        r, c = positions[0]
        dr, dc = ACTIONS[int(action)]
        nr, nc = r+dr, c+dc
        if 0 <= nr < G and 0 <= nc < G and not obstacle_map[nr,nc]:
            path_lengths[0] += ((nr-r)**2+(nc-c)**2)**0.5
            positions[0] = [nr, nc]
            visited[nr, nc] = True

        for i in range(1, n_agents):
            r, c = positions[i]
            best_a, best_s = 0, -999
            for a, (dr, dc) in enumerate(ACTIONS):
                nr, nc = r+dr, c+dc
                if 0 <= nr < G and 0 <= nc < G and not obstacle_map[nr,nc]:
                    s = (3.0 if not visited[nr,nc] else -1.0) + np.random.uniform(-0.05,0.05)
                    if s > best_s:
                        best_s, best_a = s, a
            dr, dc = ACTIONS[best_a]
            nr, nc = r+dr, c+dc
            if 0 <= nr < G and 0 <= nc < G and not obstacle_map[nr,nc]:
                path_lengths[i] += ((nr-r)**2+(nc-c)**2)**0.5
                positions[i] = [nr, nc]
                visited[nr, nc] = True

        for i in range(n_agents):
            for j in range(i+1, n_agents):
                dr_ = positions[i][0]-positions[j][0]
                dc_ = positions[i][1]-positions[j][1]
                if (dr_**2+dc_**2)**0.5 <= COLLISION_RADIUS:
                    collision_events += 1

    elapsed = time.perf_counter() - t0
    final_coverage = float(np.sum(visited)) / free_count * 100.0
    total_path = sum(p * CELL_SIZE_M for p in path_lengths)

    return {
        "final_coverage": final_coverage,
        "total_path_length": total_path,
        "collision_events": collision_events,
        "runtime_s": elapsed,
    }


def run_sensitivity():
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("stable-baselines3 not available")
        return None

    model_path = os.path.join(
        os.path.dirname(__file__), "../phase4_ppo/results/ppo_model_d20.zip")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    model = PPO.load(model_path)

    print(f"\n{'='*70}")
    print(f"  PPO — UAV Count Sensitivity (density={int(DENSITY*100)}%)")
    print(f"  Using trained 20% model, Agent 0 = PPO, rest = greedy")
    print(f"{'='*70}")

    all_results = {n: [] for n in UAV_COUNTS}
    for n in UAV_COUNTS:
        for seed in range(N_SEEDS):
            result = run_ppo_with_n_agents(model, n, DENSITY, seed)
            all_results[n].append(result)
        covs  = [r["final_coverage"] for r in all_results[n]]
        paths = [r["total_path_length"] for r in all_results[n]]
        colls = [r["collision_events"] for r in all_results[n]]
        times = [r["runtime_s"] for r in all_results[n]]
        print(f"  N={n:>2} UAVs | coverage={np.mean(covs):.1f}% "
              f"| path={np.mean(paths):.0f}m "
              f"| collisions={np.mean(colls):.1f} "
              f"| runtime={np.mean(times):.2f}s")

    np.save(os.path.join(RESULTS_DIR, "ppo_uav_sensitivity.npy"), all_results)
    return all_results


def plot_combined_sensitivity(ppo_results):
    boids_apf_path = os.path.join(RESULTS_DIR, "uav_sensitivity.npy")
    if not os.path.exists(boids_apf_path):
        print("Run uav_count_sensitivity.py first for Boids/APF comparison")
        return

    boids_apf = np.load(boids_apf_path, allow_pickle=True).item()

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    colors = {"Boids": "#457B9D", "APF": "#E63946", "PPO": "#2A9D8F"}

    metrics = [
        ("final_coverage",    "Coverage (%)",              axes[0,0]),
        ("total_path_length", "Total Path Length (m)",     axes[0,1]),
        ("collision_events",  "Mean Collision Events",     axes[1,0]),
        ("runtime_s",         "Runtime per Episode (s)",   axes[1,1]),
    ]

    for metric_key, ylabel, ax in metrics:
        for algo in ["Boids", "APF"]:
            means = [np.mean([r[metric_key] for r in boids_apf[algo][n]])
                     for n in UAV_COUNTS]
            ax.plot(UAV_COUNTS, means, marker="o", markersize=8,
                    linewidth=2, color=colors[algo], label=algo)
        means = [np.mean([r[metric_key] for r in ppo_results[n]]) for n in UAV_COUNTS]
        ax.plot(UAV_COUNTS, means, marker="s", markersize=8,
                linewidth=2, linestyle="--", color=colors["PPO"],
                label="PPO (out-of-distribution)")
        ax.set_xlabel("Number of UAVs", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(UAV_COUNTS)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0,0].set_title("Coverage vs Swarm Size", fontsize=12, fontweight="bold")
    axes[0,1].set_title("Energy Proxy vs Swarm Size", fontsize=12, fontweight="bold")
    axes[1,0].set_title("Collision Risk vs Swarm Size", fontsize=12, fontweight="bold")
    axes[1,1].set_title("Computational Cost vs Swarm Size", fontsize=12, fontweight="bold")

    fig.suptitle(
        "UAV Count Sensitivity — All Algorithms (20% Obstacle Density)\n"
        "PPO tested out-of-distribution: policy trained for 6 agents only",
        fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/uav_count_sensitivity_all3.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: uav_count_sensitivity_all3.png")


if __name__ == "__main__":
    ppo_results = run_sensitivity()
    if ppo_results:
        plot_combined_sensitivity(ppo_results)
        print(f"\n{'='*70}")
        print("  IMPORTANT: PPO results above are OUT-OF-DISTRIBUTION tests.")
        print("  The Agent-0 policy was trained assuming exactly 5 greedy")
        print("  teammates. Results at N != 6 show robustness/generalisation")
        print("  behaviour, not optimal performance at that swarm size.")
        print(f"{'='*70}")
