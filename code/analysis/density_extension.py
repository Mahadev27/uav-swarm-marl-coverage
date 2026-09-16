"""
Pushes obstacle density past 30% to find where the task stops being
well-posed, and checks how much of the arena is even reachable.

The main experiment stops at 30%. This asks what happens at 40% and 50%, and
the answer is less about the controllers than about the environment: past a
point, randomly placed buildings start walling off regions entirely, and
coverage becomes bounded by connectivity rather than by search behaviour.

reachable_fraction() is what makes that measurable. It floods outward from
HOME and counts how many free cells can actually be reached. If a controller
scores 70% on a layout where only 72% is reachable, it did nearly everything
possible -- and reporting that as a coverage failure would be wrong.

This is diagnostic work rather than a reported result. It informed the choice
to cap the main experiment at 30%, where connectivity is not the binding
constraint and the numbers still say something about the controllers.

Run:  python3 -m analysis.density_extension      (from code/)
"""

import sys, os
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../phase2_boids"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../phase3_apf"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase2_boids.model_phase2 import SwarmCoveragePhase2
from phase3_apf.model_apf import APFModel
from phase2_boids.obstacle_generator import generate_structured_obstacles

DENSITIES   = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
MAX_STEPS   = 700
NUM_UAVS    = 6
N_SEEDS     = 30
HOME        = (25, 25)
G           = 50

FIGURES_DIR = os.path.join(os.path.dirname(__file__),
                           "../../figures/extended_analysis")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def reachable_fraction(obstacle_map):
    """Flood fill from HOME. Returns (reachable, total_free, fraction).

    Eight-connected, matching the agents' movement, so a diagonal gap counts
    as passable exactly as it would for a UAV. A four-connected fill would
    understate what the swarm can actually reach.
    """
    free = ~obstacle_map
    total_free = int(free.sum())
    if not free[HOME]:
        return 0, total_free, 0.0

    seen = np.zeros_like(free, dtype=bool)
    seen[HOME] = True
    q = deque([HOME])
    moves = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    while q:
        r, c = q.popleft()
        for dr, dc in moves:
            nr, nc = r+dr, c+dc
            if 0 <= nr < G and 0 <= nc < G and free[nr,nc] and not seen[nr,nc]:
                seen[nr,nc] = True
                q.append((nr,nc))

    reached = int(seen.sum())
    return reached, total_free, reached / max(total_free, 1)


def connectivity_analysis():
    """How much of the free area is reachable, as density climbs."""
    print(f"{'='*74}")
    print("  CONNECTIVITY — how much of the free space can be reached")
    print(f"  Flood-fill from HOME, {N_SEEDS} seeds per density")
    print(f"{'='*74}")
    print(f"{'Requested':>10} {'Actual':>8} {'Free cells':>11} "
          f"{'Reachable':>10} {'Ceiling':>9} {'Isolated':>9}")
    print("-"*74)

    out = {}
    for d in DENSITIES:
        actuals, fractions, frees, reached_counts, isolated_counts = [], [], [], [], []
        for seed in range(N_SEEDS):
            obs = generate_structured_obstacles(
                grid_size=G, obstacle_density=d, seed=seed, style="urban")
            obs = obs.astype(bool)
            reached, total_free, frac = reachable_fraction(obs)
            actuals.append(obs.sum() / (G*G))
            frees.append(total_free)
            fractions.append(frac)
            reached_counts.append(reached)
            isolated_counts.append(total_free - reached)

        out[d] = {
            "actual_density": float(np.mean(actuals)),
            "free_cells":     float(np.mean(frees)),
            "ceiling":        float(np.mean(fractions)) * 100,
            "isolated":       float(np.mean(isolated_counts)),
        }
        print(f"{int(d*100):>9}% {np.mean(actuals)*100:>7.1f}% "
              f"{np.mean(frees):>11.0f} "
              f"{np.mean(reached_counts):>10.0f} "
              f"{np.mean(fractions)*100:>8.1f}% "
              f"{np.mean(isolated_counts):>9.1f}")

    print()
    print("  'Ceiling' is the maximum coverage physically attainable.")
    print("  'Isolated' is free cells with no path from HOME - these")
    print("  can never be visited, however good the algorithm is.")
    return out


def run_one(ModelClass, density, seed):
    """One episode, returning final coverage."""
    np.random.seed(seed)
    model = ModelClass(n_agents=NUM_UAVS, obstacle_density=density,
                       obstacle_seed=seed)
    for _ in range(MAX_STEPS):
        model.step()
    df = model.datacollector.get_model_vars_dataframe()
    return float(df["Coverage_%"].iloc[-1]) if len(df) else 0.0


def density_sweep():
    """Both classical controllers across the extended density range."""
    print(f"\n{'='*74}")
    print("  COVERAGE ACROSS THE EXTENDED DENSITY RANGE")
    print(f"  {N_SEEDS} seeds per condition, 700 steps, 6 UAVs")
    print(f"{'='*74}")

    results = {}
    for name, cls in [("Boids", SwarmCoveragePhase2), ("APF", APFModel)]:
        print(f"\n{name}:")
        results[name] = {}
        for d in DENSITIES:
            covs = [run_one(cls, d, s) for s in range(N_SEEDS)]
            results[name][d] = covs
            print(f"  {int(d*100):>3}% | mean={np.mean(covs):5.1f}% "
                  f"sd={np.std(covs):4.1f} "
                  f"min={np.min(covs):5.1f} max={np.max(covs):5.1f}")

    return results


def plot(results, conn):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = {"Boids": "#457B9D", "APF": "#E63946"}
    x = [d*100 for d in DENSITIES]

    for name in ["Boids", "APF"]:
        means = [np.mean(results[name][d]) for d in DENSITIES]
        stds  = [np.std(results[name][d]) for d in DENSITIES]
        ax1.errorbar(x, means, yerr=stds, marker="o", markersize=7,
                     linewidth=2, capsize=4, color=colors[name], label=name)

    ceiling = [conn[d]["ceiling"] for d in DENSITIES]
    ax1.plot(x, ceiling, linestyle="--", color="#555555", linewidth=1.5,
             label="Reachable ceiling")
    ax1.axhline(90, color="black", linestyle=":", alpha=0.5,
                label="90% target")
    ax1.axvspan(30, 50, alpha=0.07, color="grey")
    ax1.text(40, 30, "beyond the range\ntested in the main\nexperiments",
             ha="center", fontsize=8.5, color="#666666")
    ax1.set_xlabel("Obstacle density (%)", fontsize=11)
    ax1.set_ylabel("Final coverage (%)", fontsize=11)
    ax1.set_title("Coverage across the extended density range",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    req = x
    act = [conn[d]["actual_density"]*100 for d in DENSITIES]
    ax2.plot(req, req, linestyle=":", color="#999999", label="Requested")
    ax2.plot(req, act, marker="s", markersize=7, linewidth=2,
             color="#2A9D8F", label="Achieved")
    ax2.set_xlabel("Requested obstacle density (%)", fontsize=11)
    ax2.set_ylabel("Achieved obstacle density (%)", fontsize=11)
    ax2.set_title("Generator fidelity at high density",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_density_extension.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    conn = connectivity_analysis()
    results = density_sweep()
    plot(results, conn)

    np.save(os.path.join(RESULTS_DIR, "density_extension.npy"),
            {"coverage": results, "connectivity": conn})

    apf_means = [np.mean(results["APF"][d]) for d in DENSITIES]
    peak = DENSITIES[int(np.argmax(apf_means))]
    print(f"\n{'='*74}")
    print(f"  APF peaks at {int(peak*100)}% obstacle density "
          f"({max(apf_means):.1f}%)")
    if peak < 0.5:
        print("  Coverage falls beyond that point, so the corridor effect")
        print("  that helps APF at moderate density does not persist.")
    else:
        print("  No turnover within the range tested; the corridor effect")
        print("  still holds at 50%.")
    print(f"{'='*74}")
