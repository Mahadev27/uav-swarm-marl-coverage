"""
Evaluates the trained hybrid policies and writes ppo_results_A.npy and
ppo_results_B.npy, which the statistical analysis reads.

Same densities, same seeds and same 700-step budget as every other
controller, so the records pair cleanly against theirs.

Note this evaluates deterministically, unlike the MARL evaluation. With a
single learned agent there is no symmetry to break: the other five are
scripted and already behave differently from each other, so a greedy argmax
does not collapse the swarm the way it does when six agents share one network.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from stable_baselines3 import PPO
except ImportError:
    print("Run: pip install stable-baselines3")
    sys.exit(1)

from phase4_ppo.uav_coverage_env import UAVCoverageEnv

DENSITIES   = [0.0, 0.1, 0.2, 0.3]
SEEDS_A     = list(range(0, 50))
SEEDS_B     = list(range(50, 100))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "../../figures")
MAX_STEPS   = 700


def test_one(model, density, seed):
    """One episode on one layout. The seed is passed explicitly rather than
    letting the environment cycle, so evaluation never touches the modulo
    defect described in uav_coverage_env.py."""
    env    = UAVCoverageEnv(obstacle_density=density, seed=seed)
    obs, _ = env.reset(seed=seed)
    done   = False
    curve  = []
    while not done:
        # Deterministic here, unlike the MARL rollout: only one agent is
        # under policy control, so there is no shared-network symmetry to
        # break by sampling.
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        curve.append(info["coverage"])
        done = terminated or truncated
    # Pad so every curve is the same length and they stack for the mean.
    while len(curve) < MAX_STEPS:
        curve.append(curve[-1] if curve else 0.0)
    return {
        "coverage":       env.get_coverage(),
        "redundancy":     env.get_redundancy(),
        "coverage_curve": curve,
        "timesteps":      env.timestep,
        "seed":           seed,
        "density":        density,
    }


def test_seed_set(seed_set, label):
    """All four densities for one seed set."""
    print(f"\n{'='*60}")
    print(f"  PPO | Seed Set {label} | {len(seed_set)} seeds x {len(DENSITIES)} densities")
    print(f"{'='*60}")
    all_results = {}
    for d in DENSITIES:
        pct  = int(d * 100)
        path = os.path.join(RESULTS_DIR, f"ppo_model_d{pct}.zip")
        if not os.path.exists(path):
            print(f"  {pct}% — model not found: {path}")
            continue
        model   = PPO.load(path)
        results = [test_one(model, d, s) for s in seed_set]
        all_results[d] = results
        covs    = [r["coverage"] for r in results]
        above80 = sum(1 for c in covs if c >= 80.0)
        above90 = sum(1 for c in covs if c >= 90.0)
        print(f"  {pct:>2}% | mean={np.mean(covs):.1f}% "
              f"std={np.std(covs):.1f}% "
              f"min={np.min(covs):.1f}% max={np.max(covs):.1f}% "
              f">=80%:{above80}/{len(results)} "
              f">=90%:{above90}/{len(results)}")
    return all_results


def save_curves(results_A):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for d, results in results_A.items():
        pct        = int(d * 100)
        curves     = np.array([r["coverage_curve"] for r in results])
        mean_curve = np.mean(curves, axis=0)
        fname      = os.path.join(RESULTS_DIR, f"ppo_Obstacles_{pct}pct.npy")
        np.save(fname, mean_curve)
        print(f"  Curve saved -> {fname}  (final={mean_curve[-1]:.1f}%)")


def plot_ppo_results(results_A):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    densities = [int(d*100) for d in DENSITIES if d in results_A]
    means     = [np.mean([r["coverage"] for r in results_A[d]])
                 for d in DENSITIES if d in results_A]
    stds      = [np.std([r["coverage"] for r in results_A[d]])
                 for d in DENSITIES if d in results_A]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = ["steelblue", "darkorange", "green", "red"]
    bars    = ax.bar([str(d)+"%" for d in densities], means,
                     yerr=stds, capsize=5,
                     color=colors[:len(densities)], alpha=0.8)
    ax.axhline(y=90, color="black", linestyle=":", linewidth=1.5,
               label="90% target")
    ax.set_ylabel("Final Coverage (%)", fontsize=12)
    ax.set_xlabel("Obstacle Density",   fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title("PPO Final Coverage — 50 Seeds, 700 Steps, 6 UAVs",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "ppo_final_coverage.png")
    plt.savefig(out, dpi=150)
    print(f"\nFigure saved -> {out}")


if __name__ == "__main__":
    results_A = test_seed_set(SEEDS_A, "A (known environments)")
    results_B = test_seed_set(SEEDS_B, "B (unseen environments)")

    np.save(os.path.join(RESULTS_DIR, "ppo_results_A.npy"), results_A)
    np.save(os.path.join(RESULTS_DIR, "ppo_results_B.npy"), results_B)
    print("\nSaved ppo_results_A.npy and ppo_results_B.npy")

    if results_A:
        print("\nSaving curves for compare.py...")
        save_curves(results_A)
        plot_ppo_results(results_A)

    if results_A and results_B:
        print(f"\n{'='*60}")
        print("  GENERALISATION: PPO Seed Set A vs B")
        print(f"{'='*60}")
        for d in DENSITIES:
            if d not in results_A or d not in results_B:
                continue
            a    = np.mean([r["coverage"] for r in results_A[d]])
            b    = np.mean([r["coverage"] for r in results_B[d]])
            flag = "<- GENERALISES" if abs(a-b) < 10 else "<- DEGRADES"
            print(f"  {int(d*100):>2}%  A={a:.1f}%  B={b:.1f}%  "
                  f"delta={a-b:+.1f}%  {flag}")
        print(f"{'='*60}\n")
