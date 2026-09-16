import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
RESULTS = os.path.join(_HERE, "results")
FIGDIR = os.path.join(os.path.dirname(_CODE), "latex_final", "figures")

TEXTWIDTH = 7.0
COLWIDTH = 3.417

COLOUR = {"Boids": "#4878A8", "APF": "#D64550",
          "Hybrid": "#E8A33D", "MARL": "#1F9E89"}
MARKER = {"Boids": "o", "APF": "s", "Hybrid": "^", "MARL": "D"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.4,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.borderpad": 0.3,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.25,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})


def save(fig, relpath):
    """Write and report the physical size, which is the point of this file."""
    out = os.path.join(FIGDIR, relpath)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    w, h = fig.get_size_inches()
    print(f"  {relpath:<46} {w:.2f} x {h:.2f} in")


def read_descriptive():
    """Pull means and CIs from the stats CSV rather than recomputing them.

    Reading the same file the report tables come from means a figure can
    never quietly disagree with the table beside it.
    """
    path = os.path.join(_CODE, "..", "latex_final", "stats_descriptive.csv")
    if not os.path.exists(path):
        path = os.path.join(RESULTS, "stats_descriptive.csv")
    rows = list(csv.DictReader(open(path)))
    d = {}
    for r in rows:
        d.setdefault(r["seed_set"], {}).setdefault(r["method"], {})[
            int(r["density_pct"])] = (
                float(r["mean"]), float(r["ci95_low"]), float(r["ci95_high"]))
    return d


def fig_coverage():
    """Figure 6: coverage against obstacle density, both seed sets.

    The paper's central result. The crossover between the learned policy and
    APF is the thing to see, so both panels share a y-axis and the 90% target
    is drawn as a reference line rather than left to the reader.
    """
    d = read_descriptive()
    label = {"Boids": "Boids", "APF": "APF",
             "PPO": "PPO hybrid", "MARL": "MARL (ours)"}
    key = {"Boids": "Boids", "APF": "APF",
           "PPO": "Hybrid", "MARL": "MARL"}
    dens = [0, 10, 20, 30]

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.66), sharey=True)
    for ax, (sset, title) in zip(
            axes, [("A", "Seed Set A (development)"),
                   ("B", "Seed Set B (held out)")]):
        tgt = ax.axhline(90, color="0.35", ls=(0, (4, 3)), lw=0.9, zorder=1)
        for m in ["Boids", "APF", "PPO", "MARL"]:
            mu = np.array([d[sset][m][x][0] for x in dens])
            lo = np.array([d[sset][m][x][1] for x in dens])
            hi = np.array([d[sset][m][x][2] for x in dens])
            ax.errorbar(dens, mu, yerr=[mu - lo, hi - mu],
                        color=COLOUR[key[m]], marker=MARKER[key[m]],
                        label=label[m], capsize=2.2, capthick=0.8,
                        elinewidth=0.9, zorder=3,
                        markeredgecolor="white", markeredgewidth=0.4)
        ax.set_title(title, pad=4)
        ax.set_xlabel("Obstacle density (%)")
        ax.set_xticks(dens)
        ax.set_xlim(-3, 33)
        ax.grid(alpha=0.3, lw=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Final coverage (%)")
    axes[0].set_ylim(80, 99)
    h, lab = axes[0].get_legend_handles_labels()
    axes[0].legend(h + [tgt], lab + ["90% target"], loc="lower left",
                   ncol=3, columnspacing=0.9)
    fig.subplots_adjust(wspace=0.07)
    save(fig, "results/four_way_coverage_ci.png")


def fig_swarm_size():
    """Figure 8: coverage and near-misses against swarm size.

    Near-misses go on a log axis because the learned controllers sit an order
    of magnitude below the classical pair; on a linear axis their curves would
    be flat against zero and the gap would be invisible.

    MARL's near-misses are read from marl_swarm_size_classical.json, not from
    the environment's own counter, so this series is measured the same way as
    the three beside it.
    """
    sizes = [3, 6, 9, 12]
    cls = np.load(os.path.join(RESULTS, "uav_sensitivity.npy"),
                  allow_pickle=True).item()
    hyb = np.load(os.path.join(RESULTS, "ppo_uav_sensitivity.npy"),
                  allow_pickle=True).item()
    marl = np.load(os.path.join(RESULTS, "marl_swarm_size.npy"),
                   allow_pickle=True).item()

    cov, coll = {}, {}
    for name in ["Boids", "APF"]:
        cov[name] = [np.mean([r["final_coverage"] for r in cls[name][n]])
                     for n in sizes]
        coll[name] = [np.mean([r["collision_events"] for r in cls[name][n]])
                      for n in sizes]
    cov["Hybrid"] = [np.mean([r["final_coverage"] for r in hyb[n]])
                     for n in sizes]
    coll["Hybrid"] = [np.mean([r["collision_events"] for r in hyb[n]])
                      for n in sizes]
    cov["MARL"] = [float(np.mean(marl[n]["cov"])) for n in sizes]
    cls_path = os.path.join(_CODE, "phase5_marl", "results",
                            "marl_swarm_size_classical.json")
    marl_cl = json.load(open(cls_path))
    coll["MARL"] = [float(np.mean([r["near_classical"]
                                   for r in marl_cl[str(n)]]))
                    for n in sizes]

    label = {"Boids": "Boids", "APF": "APF",
             "Hybrid": "PPO hybrid", "MARL": "MARL (ours)"}
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.52))
    for name in ["Boids", "APF", "Hybrid", "MARL"]:
        style = dict(color=COLOUR[name], marker=MARKER[name],
                     label=label[name], markeredgecolor="white",
                     markeredgewidth=0.4,
                     ls="--" if name in ("Hybrid", "MARL") else "-")
        axes[0].plot(sizes, cov[name], **style)
        axes[1].plot(sizes, np.maximum(coll[name], 0.5), **style)

    axes[0].axhline(90, color="0.35", ls=(0, (4, 3)), lw=0.9, zorder=1)
    axes[0].set_ylabel("Final coverage (%)")
    axes[0].annotate("90% target", xy=(3.2, 90), xytext=(0, 2.5),
                     textcoords="offset points", fontsize=7, color="0.3")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Near-miss events (log)")
    for ax in axes:
        ax.set_xlabel("Swarm size (number of UAVs)")
        ax.set_xticks(sizes)
        ax.set_xlim(2.2, 12.8)
        ax.grid(alpha=0.3, lw=0.4, which="both")
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower right", ncol=2, columnspacing=1.0)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "extended/uav_count_sensitivity_all3.png")


def fig_reward():
    """Figure 4: what each reward term contributes over one episode.

    Symmetric log scale, and that is not decoration. The three terms span
    three orders of magnitude -- coverage reaches ~750, the time penalty
    -35, the team bonus 1.9 -- so on a linear axis two of the three sit flat
    on zero and the figure shows nothing. Symlog is what makes the comparison
    the figure exists for actually visible.
    """
    T = 700
    t = np.arange(T + 1)
    cov = 750.0 * (1 - np.exp(-t / 190.0))
    time_pen = -0.05 * t
    team = np.zeros_like(t, dtype=float)
    team[-1] = 2.0 * 93.6 / 100.0

    fig, ax = plt.subplots(figsize=(COLWIDTH, 1.50))
    ax.plot(t, cov, color=COLOUR["MARL"])
    ax.plot(t, time_pen, color=COLOUR["APF"])
    ax.plot(t, team, color=COLOUR["Hybrid"])
    ax.scatter([T], [team[-1]], s=14, color=COLOUR["Hybrid"], zorder=5,
               edgecolor="white", linewidth=0.5)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_yticks([-10, -1, 0, 1, 10, 100, 1000])
    ax.set_yticklabels(["$-10$", "$-1$", "0", "1", "10", "100", "1000"])
    ax.axhline(0, color="0.55", lw=0.6, zorder=1)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative reward (symlog)")
    ax.set_xlim(0, 830)
    ax.set_ylim(-60, 4000)
    ax.set_xticks([0, 175, 350, 525, 700])
    ax.grid(alpha=0.3, lw=0.4, which="major")
    ax.set_axisbelow(True)

    ax.annotate(r"coverage $\rightarrow 750$", xy=(300, 700),
                xytext=(210, 1700), fontsize=7, color=COLOUR["MARL"],
                fontweight="bold")
    ax.annotate(r"team bonus $+1.9$, paid once", xy=(T, team[-1]),
                xytext=(120, 4.5), fontsize=7, color="#B8801F",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#B8801F",
                                shrinkA=1, shrinkB=2))
    ax.annotate(r"time penalty $\rightarrow -35$", xy=(300, -12),
                xytext=(150, -4.0), fontsize=7, color=COLOUR["APF"],
                fontweight="bold")
    save(fig, "results/reward_structure.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("regenerating report figures at print size\n")
    if which in ("all", "coverage"):
        fig_coverage()
    if which in ("all", "swarm"):
        fig_swarm_size()
    if which in ("all", "reward"):
        fig_reward()
    print(f"\nwritten under {FIGDIR}")
