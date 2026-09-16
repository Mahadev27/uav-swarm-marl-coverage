"""
Runs the APF baseline over every density and both seed sets, and writes the
result files the statistical analysis reads.

Mirrors run_boids.py exactly -- same densities, same seeds, same 700-step
budget, same output format -- so the two can be compared run against run.
Output is apf_results_A.npy and apf_results_B.npy.

APF has no training, so the A/B split here is a control rather than a
generalisation test. It answers a narrower question: does this controller
behave the same on both halves of the seed space? If it did not, the learned
controllers' held-out result would be hard to interpret.

The default parameters come from the model, (k_rep=1.5, rep_radius=2). See
analysis/apf_parameter_sweep.py for the surface those sit on.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model_apf import APFModel

DENSITIES  = [0.0, 0.1, 0.2, 0.3]
SEEDS_A    = list(range(0, 50))
SEEDS_B    = list(range(50, 100))
MAX_STEPS  = 700         # same budget for every controller
NUM_UAVS   = 6


def run_one(density, seed):
    """One episode. The seed fixes both the layout and the agent randomness."""
    np.random.seed(seed)
    model = APFModel(n_agents=NUM_UAVS,
                     obstacle_density=density,
                     obstacle_seed=seed)
    for _ in range(MAX_STEPS):
        model.step()
    data  = model.datacollector.get_model_vars_dataframe()
    curve = data["Coverage_%"].values.tolist()
    # Pad short curves so all runs stack into one array for the mean.
    while len(curve) < MAX_STEPS:
        curve.append(curve[-1] if curve else 0.0)
    return {"coverage": float(data["Coverage_%"].iloc[-1]),
            "seed": seed, "density": density, "coverage_curve": curve}


def run_seed_set(seed_set, label):
    """All four densities for one seed set."""
    print(f"\n{'='*60}")
    print(f"  APF | Seed Set {label} | {len(seed_set)} seeds x {len(DENSITIES)} densities")
    print(f"{'='*60}")
    all_results = {}
    for d in DENSITIES:
        results = [run_one(d, s) for s in seed_set]
        all_results[d] = results
        covs    = [r["coverage"] for r in results]
        above80 = sum(1 for c in covs if c >= 80.0)
        above90 = sum(1 for c in covs if c >= 90.0)
        print(f"  {int(d*100):>2}% | mean={np.mean(covs):.1f}% std={np.std(covs):.1f}% "
              f"min={np.min(covs):.1f}% max={np.max(covs):.1f}% "
              f">=80%:{above80}/{len(results)} >=90%:{above90}/{len(results)}")
    return all_results


def save_curves(all_results, results_dir):
    """Mean coverage-over-time curve per density."""
    for d, results in all_results.items():
        d_int      = int(d*100)
        curves     = np.array([r["coverage_curve"] for r in results])
        mean_curve = np.mean(curves, axis=0)
        fname      = os.path.join(results_dir, f"apf_Obstacles_{d_int}pct.npy")
        np.save(fname, mean_curve)
        print(f"  Saved -> {fname}  (final={mean_curve[-1]:.1f}%)")


if __name__ == "__main__":
    results_dir  = os.path.join(os.path.dirname(__file__), "results")
    figures_path = os.path.join(os.path.dirname(__file__), "../../figures")
    os.makedirs(results_dir,  exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    results_A = run_seed_set(SEEDS_A, "A (known environments)")
    results_B = run_seed_set(SEEDS_B, "B (unseen environments)")

    np.save(os.path.join(results_dir, "apf_results_A.npy"), results_A)
    np.save(os.path.join(results_dir, "apf_results_B.npy"), results_B)
    print("\nSaved apf_results_A.npy and apf_results_B.npy")

    print("\nSaving mean coverage curves (Seed Set A)...")
    save_curves(results_A, results_dir)

    colors    = ["steelblue", "darkorange", "green", "red"]
    timesteps = np.arange(MAX_STEPS)
    fig, ax   = plt.subplots(figsize=(10, 6))
    for d, color in zip(DENSITIES, colors):
        curves     = np.array([r["coverage_curve"] for r in results_A[d]])
        mean_curve = np.mean(curves, axis=0)
        std_curve  = np.std(curves,  axis=0)
        ax.plot(timesteps, mean_curve, color=color, linewidth=2,
                label=f"Obstacles={int(d*100)}%  (final={mean_curve[-1]:.1f}%)")
        ax.fill_between(timesteps,
                        np.clip(mean_curve-std_curve, 0, 100),
                        np.clip(mean_curve+std_curve, 0, 100),
                        color=color, alpha=0.15)
    ax.axhline(y=90, color="black", linestyle=":", linewidth=1.5, label="90% target")
    ax.set_xlabel("Timestep", fontsize=13)
    ax.set_ylabel("Coverage (%)", fontsize=13)
    ax.set_ylim(0, 105)
    ax.set_title(f"Phase 3 APF: Mean Coverage +/- Std — {NUM_UAVS} UAVs, 50x50 Grid",
                 fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(figures_path, "phase3_apf_obstacle_density.png")
    plt.savefig(out, dpi=150)
    print(f"\nFigure saved -> {out}")
    plt.show()

    print(f"\n{'='*60}")
    print("  GENERALISATION: APF Seed Set A vs B")
    print(f"{'='*60}")
    for d in DENSITIES:
        mean_a = np.mean([r["coverage"] for r in results_A[d]])
        mean_b = np.mean([r["coverage"] for r in results_B[d]])
        delta  = mean_a - mean_b
        # Eyeball threshold for this printout only; the reported result uses
        # paired tests with Holm correction in analysis/statistical_tests.py.
        flag   = "<- GENERALISES" if abs(delta) < 5.0 else "<- DEGRADES"
        print(f"  {int(d*100):>2}%  A={mean_a:.1f}%  B={mean_b:.1f}%  "
              f"delta={delta:+.1f}%  {flag}")
