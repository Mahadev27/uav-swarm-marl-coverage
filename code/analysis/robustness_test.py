"""
Two perturbation tests. One is reported; one is withdrawn, and the reason is
worth reading.

MOVING OBSTACLES (reported). A share of the obstacles relocate every 50
steps, simulating an environment that changes under the swarm. At 20% density
with half the obstacles moving, APF falls from 94.0% to 85.9% while Boids
drops only from 85.9% to 82.8%. APF needs a potential field that stays
consistent long enough to guide an agent through a corridor; when the field
keeps changing, that guidance is worth less. Boids recomputes from scratch
each step and barely notices.

The learned policies were not retrained under this perturbation, so this
covers the classical pair only.

SENSOR NOISE (WITHDRAWN -- see run_ppo_with_sensor_noise and
run_boids_apf_with_noise below). This experiment is not reported anywhere and
nothing depends on it. The implementations are kept so the flaw is inspectable
rather than merely asserted.

The flaw: the two noise models are not the same intervention. For Boids and
APF the perturbation corrupts the obstacle grid itself, once per episode,
marking roughly 398 of 1,991 free cells unreachable -- which imposes a
mechanical ceiling near 80% before the controller does anything at all. For
the hybrid it enters only the observation of the single learned agent, which
contributes 0.4% of coverage. So the comparison is between "the map is
broken" and "one agent out of six sees slightly wrong values". Those are not
the same experiment, and no conclusion survives the difference.

A valid version needs the same corruption, the same representation and the
same persistence for every controller. That is future work, and the report
says so rather than reporting the numbers this produced.

Run:  python3 -m analysis.robustness_test      (from code/)
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

from phase2_boids.model_phase2 import SwarmCoveragePhase2
from phase3_apf.model_apf import APFModel

DENSITY      = 0.20
MAX_STEPS    = 700
NUM_UAVS     = 6
N_SEEDS      = 20
G            = 50
HOME         = (25, 25)

NOISE_LEVELS       = [0.0, 0.05, 0.10, 0.20]
MOVING_OBS_FRACTIONS = [0.0, 0.10, 0.25, 0.50]
MOVE_INTERVAL       = 50

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "../../figures/extended_analysis")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(FIGURES_DIR, exist_ok=True)


def run_moving_obstacles(ModelClass, moving_fraction, seed):
    """Relocate a share of the obstacles every 50 steps.

    This is the reported perturbation. Both classical controllers get the
    identical treatment: same layout, same relocation schedule, same seed.
    """
    np.random.seed(seed)
    rng   = np.random.default_rng(seed)
    model = ModelClass(n_agents=NUM_UAVS, obstacle_density=DENSITY,
                       obstacle_seed=seed)

    obs_map = model.obstacle_grid
    obs_cells = list(zip(*np.where(obs_map == 1)))
    n_moving  = int(len(obs_cells) * moving_fraction)
    moving_idx = rng.choice(len(obs_cells), size=n_moving, replace=False) if n_moving > 0 else []

    for step in range(MAX_STEPS):
        if moving_fraction > 0 and step > 0 and step % MOVE_INTERVAL == 0:
            for idx in moving_idx:
                old_r, old_c = obs_cells[idx]
                if model.obstacle_grid[old_r][old_c] == 1:
                    model.obstacle_grid[old_r][old_c] = 0
                    if model.coverage_grid[old_r][old_c] == -1:
                        model.coverage_grid[old_r][old_c] = 0
                for _ in range(20):
                    nr, nc = int(rng.integers(0,G)), int(rng.integers(0,G))
                    if model.obstacle_grid[nr][nc] == 0 and \
                       (abs(nr-HOME[0])>5 or abs(nc-HOME[1])>5):
                        model.obstacle_grid[nr][nc] = 1
                        obs_cells[idx] = (nr, nc)
                        break
        model.step()

    data = model.datacollector.get_model_vars_dataframe()
    return float(data["Coverage_%"].iloc[-1]) if len(data) > 0 else 0.0


def run_ppo_with_sensor_noise(noise_level, density, seed):
    """WITHDRAWN. Noise enters only the learned agent's observation.

    Not comparable with run_boids_apf_with_noise below, which corrupts the
    map itself. Kept so the asymmetry can be inspected. Not reported.
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
    rng = np.random.default_rng(seed + 1000)

    done = False
    while not done:
        noisy_obs = obs.copy()
        if noise_level > 0:
            n_flip = int(49 * noise_level)
            flip_idx = rng.choice(49, size=n_flip, replace=False)
            for idx in flip_idx:
                val = noisy_obs[idx]
                if val == -1.0:
                    noisy_obs[idx] = 0.0
                elif val == 0.0:
                    noisy_obs[idx] = -1.0
        action, _ = model.predict(noisy_obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return info["coverage"]


def run_boids_apf_with_noise(ModelClass, noise_level, seed):
    """WITHDRAWN. Noise corrupts the obstacle grid, once, permanently.

    This marks free cells as blocked for the whole episode, capping
    achievable coverage near 80% before the controller acts. A fundamentally
    different intervention from the observation noise applied to the hybrid.
    Not reported.
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = ModelClass(n_agents=NUM_UAVS, obstacle_density=DENSITY,
                       obstacle_seed=seed)

    true_obs = model.obstacle_grid.copy()
    if noise_level > 0:
        flip_mask = rng.random(true_obs.shape) < noise_level
        perceived = true_obs.copy()
        perceived[flip_mask] = 1 - perceived[flip_mask]
        model.obstacle_grid = perceived

    for step in range(MAX_STEPS):
        model.step()

    data = model.datacollector.get_model_vars_dataframe()
    return float(data["Coverage_%"].iloc[-1]) if len(data) > 0 else 0.0


def run_all_robustness_tests():
    """Runs both. Only the moving-obstacle results are used anywhere."""
    results = {
        "moving_obstacles": {"Boids": {}, "APF": {}},
        "sensor_noise":     {"Boids": {}, "APF": {}, "PPO": {}},
    }

    print(f"{'='*70}")
    print("  TEST 1: MOVING OBSTACLES")
    print(f"{'='*70}")
    for algo_name, ModelClass in [("Boids", SwarmCoveragePhase2), ("APF", APFModel)]:
        print(f"\n{algo_name}:")
        for frac in MOVING_OBS_FRACTIONS:
            covs = [run_moving_obstacles(ModelClass, frac, s) for s in range(N_SEEDS)]
            results["moving_obstacles"][algo_name][frac] = covs
            print(f"  {int(frac*100):>3}% moving | coverage={np.mean(covs):.1f}% "
                  f"(+/- {np.std(covs):.1f})")

    print(f"\n{'='*70}")
    print("  TEST 2: SENSOR NOISE")
    print(f"{'='*70}")
    for algo_name, ModelClass in [("Boids", SwarmCoveragePhase2), ("APF", APFModel)]:
        print(f"\n{algo_name}:")
        for noise in NOISE_LEVELS:
            covs = [run_boids_apf_with_noise(ModelClass, noise, s) for s in range(N_SEEDS)]
            results["sensor_noise"][algo_name][noise] = covs
            print(f"  noise={noise:.2f} | coverage={np.mean(covs):.1f}% "
                  f"(+/- {np.std(covs):.1f})")

    print(f"\nPPO:")
    for noise in NOISE_LEVELS:
        covs = [run_ppo_with_sensor_noise(noise, DENSITY, s)
                for s in range(N_SEEDS)]
        covs = [c for c in covs if c is not None]
        if covs:
            results["sensor_noise"]["PPO"][noise] = covs
            print(f"  noise={noise:.2f} | coverage={np.mean(covs):.1f}% "
                  f"(+/- {np.std(covs):.1f})")

    np.save(os.path.join(RESULTS_DIR, "robustness_results.npy"), results)
    return results


def plot_robustness(results):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = {"Boids": "#457B9D", "APF": "#E63946", "PPO": "#2A9D8F"}

    ax1 = axes[0]
    for algo in ["Boids", "APF"]:
        means = [np.mean(results["moving_obstacles"][algo][f]) for f in MOVING_OBS_FRACTIONS]
        stds  = [np.std(results["moving_obstacles"][algo][f]) for f in MOVING_OBS_FRACTIONS]
        ax1.errorbar([f*100 for f in MOVING_OBS_FRACTIONS], means, yerr=stds,
                     marker="o", markersize=8, linewidth=2, capsize=5,
                     color=colors[algo], label=algo)
    ax1.set_xlabel("% of Obstacles That Move (every 50 steps)", fontsize=11)
    ax1.set_ylabel("Final Coverage (%)", fontsize=11)
    ax1.set_title("Robustness to Moving Obstacles", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for algo in ["Boids", "APF", "PPO"]:
        if algo not in results["sensor_noise"] or not results["sensor_noise"][algo]:
            continue
        means = [np.mean(results["sensor_noise"][algo][n]) for n in NOISE_LEVELS]
        stds  = [np.std(results["sensor_noise"][algo][n]) for n in NOISE_LEVELS]
        ax2.errorbar([n*100 for n in NOISE_LEVELS], means, yerr=stds,
                     marker="o", markersize=8, linewidth=2, capsize=5,
                     color=colors[algo], label=algo)
    ax2.set_xlabel("Sensor Noise Level (% of observation cells flipped)", fontsize=11)
    ax2.set_ylabel("Final Coverage (%)", fontsize=11)
    ax2.set_title("Robustness to Sensor Noise", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Robustness Testing — 20% Obstacle Density, {N_SEEDS} seeds",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/robustness_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: robustness_comparison.png")


if __name__ == "__main__":
    results = run_all_robustness_tests()
    plot_robustness(results)
