"""
Renders swarm trajectories: where each agent actually flew.

The summary statistics say APF and the MARL policy score similarly at 20%
density. They do not say that the two get there by completely different
strategies, and that is what these pictures show.

APF follows the corridors between buildings -- its repulsion term pushes
agents away from walls, which funnels them along streets. The MARL policy
sweeps outward from the launch point and partitions the arena between agents,
with no communication doing it. Boids doubles back over open ground, which is
what the redundancy measure quantifies but does not make visible.

Same layout, same seed, same budget for all three, so the differences are in
behaviour rather than in the problem.

Figure 7 in the report comes from here. The equivalents at every density,
referred to in that caption, are also produced.

Run:  python3 -m visualise.visualise_trajectories      (from code/)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../phase2_boids"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../phase3_apf"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../phase4_ppo"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from phase2_boids.model_phase2 import SwarmCoveragePhase2
from phase3_apf.model_apf import APFModel

DENSITIES      = [0.0, 0.10, 0.20, 0.30]
MAX_STEPS      = 700
NUM_UAVS       = 6
SEED           = 15
GRID_SIZE      = 50
HOME           = (25, 25)
DENSITY_LABELS = ["0% obstacles","10% obstacles",
                  "20% obstacles","30% obstacles"]
UAV_COLORS     = ["#E63946","#457B9D","#2A9D8F",
                  "#E9C46A","#F4A261","#6A0572"]


def record_mesa_trajectories(ModelClass, obstacle_density, seed):
    """Run a Mesa controller, keeping every agent's full position history."""
    np.random.seed(seed)
    model        = ModelClass(n_agents=NUM_UAVS,
                              obstacle_density=obstacle_density,
                              obstacle_seed=seed)
    obstacle_map = model.obstacle_grid.copy()
    trajectories = {i: [] for i in range(NUM_UAVS)}

    for step in range(MAX_STEPS):
        uavs = [a for a in model.agents if hasattr(a, 'cells_visited')]
        for i, agent in enumerate(uavs[:NUM_UAVS]):
            trajectories[i].append(agent.pos)
        model.step()

    uavs = [a for a in model.agents if hasattr(a, 'cells_visited')]
    for i, agent in enumerate(uavs[:NUM_UAVS]):
        trajectories[i].append(agent.pos)

    data      = model.datacollector.get_model_vars_dataframe()
    final_cov = data["Coverage_%"].iloc[-1]
    return trajectories, obstacle_map, final_cov


def record_ppo_trajectories(obstacle_density, seed):
    """Same for the PPO hybrid, which has a different interface."""
    try:
        from stable_baselines3 import PPO
        from phase4_ppo.uav_coverage_env import UAVCoverageEnv, _build_obstacles, ACTIONS
    except ImportError:
        print("  PPO not available — skipping")
        return None, None, None

    pct        = int(obstacle_density * 100)
    model_path = os.path.join(
        os.path.dirname(__file__), f"../phase4_ppo/results/ppo_model_d{pct}.zip")

    if not os.path.exists(model_path):
        print(f"  PPO model not found: {model_path}")
        return None, None, None

    model = PPO.load(model_path)
    env   = UAVCoverageEnv(obstacle_density=obstacle_density, seed=seed)
    obs, _ = env.reset(seed=seed)

    trajectories = {i: [tuple(env.positions[i])] for i in range(NUM_UAVS)}
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        for i in range(NUM_UAVS):
            trajectories[i].append(tuple(env.positions[i]))
        done = terminated or truncated

    obstacle_map = env.obstacle_map.astype(int)
    final_cov    = env.get_coverage()
    return trajectories, obstacle_map, final_cov


def draw_trajectories(ax, trajectories, obstacle_map, final_cov, title):
    """One panel: obstacles, six coloured paths, HOME.

    Each agent gets its own colour so a reader can follow one through the
    episode and see whether it stayed in its own region or wandered across
    everyone else's.
    """
    for (ox, oy) in np.argwhere(obstacle_map == 1):
        ax.add_patch(plt.Rectangle(
            (oy-0.5, ox-0.5), 1, 1,
            color="#4A4A4A", zorder=1))

    ax.set_xlim(-0.5, GRID_SIZE-0.5)
    ax.set_ylim(-0.5, GRID_SIZE-0.5)
    ax.set_xticks(np.arange(-0.5, GRID_SIZE, 10))
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, 10))
    ax.tick_params(length=0, labelsize=7)
    ax.grid(True, color="#EEEEEE", linewidth=0.4, zorder=0)
    ax.set_aspect("equal")

    for i in range(NUM_UAVS):
        traj  = trajectories[i]
        color = UAV_COLORS[i]
        if len(traj) < 2:
            continue
        xs = [p[1] for p in traj]
        ys = [p[0] for p in traj]
        n  = len(xs) - 1

        for seg in range(n):
            alpha = 0.12 + 0.70*(seg/max(n,1))
            ax.plot(xs[seg:seg+2], ys[seg:seg+2],
                    color=color, linewidth=1.0,
                    alpha=alpha, zorder=2,
                    solid_capstyle="round")

        if len(xs) >= 4:
            ax.annotate("",
                xy=(xs[-1], ys[-1]),
                xytext=(xs[-4], ys[-4]),
                arrowprops=dict(arrowstyle="->",
                                color=color, lw=1.3),
                zorder=6)

    hx, hy = HOME[1], HOME[0]
    ax.plot(hx, hy, "*", color="#FFD700", markersize=20,
            markeredgecolor="#333333", markeredgewidth=1.0,
            zorder=10)
    ax.annotate("HOME", xy=(hx, hy), xytext=(hx+1.5, hy+1.5),
                fontsize=7, fontweight="bold",
                color="#333333", zorder=11)

    offsets = [(-0.8,-0.8),(-0.8,0),(-0.8,0.8),
               ( 0.8,-0.8),( 0.8,0),( 0.8,0.8)]
    for i in range(NUM_UAVS):
        sx = hx + offsets[i][1]
        sy = hy + offsets[i][0]
        ax.plot(sx, sy, "o", color=UAV_COLORS[i], markersize=6,
                markeredgecolor="white", markeredgewidth=1.2, zorder=9)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
    ax.set_xlabel("Column", fontsize=8)
    ax.set_ylabel("Row",    fontsize=8)


def make_legend():
    """Shared legend, so each panel does not repeat it."""
    handles = [mpatches.Patch(color=UAV_COLORS[i], label=f"UAV {i+1}")
               for i in range(NUM_UAVS)]
    handles += [
        plt.Line2D([0],[0], marker="*", color="#FFD700",
                   markersize=13, markeredgecolor="#333",
                   label="HOME (start & end)", linestyle="None"),
        plt.Line2D([0],[0], marker="o", color="grey",
                   markersize=8, label="UAV start",
                   linestyle="None", markeredgecolor="white"),
    ]
    return handles


def plot_algo_grid(all_data, algo_name, figures_path):
    """One controller at all four densities, as a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    for ax_idx, d in enumerate(DENSITIES):
        traj, obs, cov = all_data[d]
        if traj is None:
            axes.flatten()[ax_idx].set_title("No data", fontsize=11)
            continue
        title = f"{DENSITY_LABELS[ax_idx]}\nFinal Coverage: {cov:.1f}%"
        draw_trajectories(axes.flatten()[ax_idx], traj, obs, cov, title)

    fig.legend(handles=make_legend(), loc="lower center",
               ncol=8, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(
        f"{algo_name} — UAV Trajectories (Structured Urban Obstacles)\n"
        f"6 UAVs, 50×50 Grid, 700 Steps, HOME=(25,25), Seed={SEED}",
        fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    fname = os.path.join(figures_path,
        f"trajectories_{algo_name.lower()}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {fname}")


def plot_three_way(boids_data, apf_data, ppo_data, density, figures_path):
    d_int = int(density * 100)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    datasets = [("Boids", boids_data), ("APF", apf_data), ("PPO", ppo_data)]
    for ax, (algo_name, data) in zip(axes, datasets):
        traj, obs, cov = data[density]
        if traj is None:
            ax.set_title(f"{algo_name} — No data", fontsize=11)
            continue
        title = f"{algo_name} — {d_int}% Obstacles\nFinal Coverage: {cov:.1f}%"
        draw_trajectories(ax, traj, obs, cov, title)

    fig.legend(handles=make_legend(), loc="lower center",
               ncol=8, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"Boids vs APF vs PPO — {d_int}% Obstacle Density (Urban Structured)\n"
        f"All UAVs start and end at HOME (★), 700 Steps, Seed={SEED}",
        fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    fname = os.path.join(figures_path,
        f"trajectory_three_way_{d_int}pct.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {fname}")


def plot_boids_vs_apf(boids_data, apf_data, density, figures_path):
    d_int = int(density * 100)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    b_traj, b_obs, b_cov = boids_data[density]
    a_traj, a_obs, a_cov = apf_data[density]

    draw_trajectories(ax1, b_traj, b_obs, b_cov,
        f"Boids — {d_int}% Obstacles\nFinal Coverage: {b_cov:.1f}%")
    draw_trajectories(ax2, a_traj, a_obs, a_cov,
        f"APF — {d_int}% Obstacles\nFinal Coverage: {a_cov:.1f}%")

    fig.legend(handles=make_legend(), loc="lower center",
               ncol=8, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"Boids vs APF — {d_int}% Obstacle Density (Urban Structured)\n"
        f"All UAVs start and end at HOME (★), 700 Steps",
        fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    fname = os.path.join(figures_path,
        f"trajectory_comparison_{d_int}pct.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {fname}")


if __name__ == "__main__":
    figures_path = os.path.join(
        os.path.dirname(__file__), "../../figures")
    os.makedirs(figures_path, exist_ok=True)

    print("Recording Boids trajectories...")
    boids_data = {}
    for d in DENSITIES:
        print(f"  {int(d*100)}% obstacles...")
        traj, obs, cov = record_mesa_trajectories(
            SwarmCoveragePhase2, d, SEED)
        boids_data[d] = (traj, obs, cov)
        print(f"    Coverage: {cov:.1f}%")

    print("\nRecording APF trajectories...")
    apf_data = {}
    for d in DENSITIES:
        print(f"  {int(d*100)}% obstacles...")
        traj, obs, cov = record_mesa_trajectories(
            APFModel, d, SEED)
        apf_data[d] = (traj, obs, cov)
        print(f"    Coverage: {cov:.1f}%")

    print("\nRecording PPO trajectories...")
    ppo_data = {}
    for d in DENSITIES:
        print(f"  {int(d*100)}% obstacles...")
        traj, obs, cov = record_ppo_trajectories(d, SEED)
        ppo_data[d] = (traj, obs, cov)
        if cov is not None:
            print(f"    Coverage: {cov:.1f}%")

    print("\nSaving figures...")

    plot_algo_grid(boids_data, "Boids", figures_path)
    plot_algo_grid(apf_data,   "APF",   figures_path)
    if any(ppo_data[d][0] is not None for d in DENSITIES):
        plot_algo_grid(ppo_data, "PPO", figures_path)

    for d in DENSITIES:
        plot_boids_vs_apf(boids_data, apf_data, d, figures_path)

    for d in DENSITIES:
        if ppo_data[d][0] is not None:
            plot_three_way(boids_data, apf_data, ppo_data, d, figures_path)

    print("\nDone — figures saved to figures/ folder:")
    print("  trajectories_boids.png")
    print("  trajectories_apf.png")
    print("  trajectories_ppo.png")
    print("  trajectory_comparison_0pct.png  (Boids vs APF)")
    print("  trajectory_comparison_10pct.png")
    print("  trajectory_comparison_20pct.png")
    print("  trajectory_comparison_30pct.png")
    print("  trajectory_three_way_0pct.png   (Boids vs APF vs PPO)")
    print("  trajectory_three_way_10pct.png")
    print("  trajectory_three_way_20pct.png")
    print("  trajectory_three_way_30pct.png")
