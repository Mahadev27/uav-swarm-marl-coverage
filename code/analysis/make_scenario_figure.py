import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
for _sub in ["", "phase2_boids", "phase5_marl"]:
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from stable_baselines3 import PPO
from phase5_marl.marl_env import (
    CoverageWorld, HOME, HALF, G, START_OFFSETS)

torch.set_num_threads(1)

FIGDIR = os.path.join(os.path.dirname(_CODE), "latex_final", "figures")
COLWIDTH = 3.417
CELL_M = 3.0

DENSITY = 0.2
SEED = 7
SNAPSHOT = 110
FOCUS = 2

TRAIL = ["#4878A8", "#D64550", "#1F9E89", "#E8A33D", "#8E6FB5", "#5C9E5C"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})


def run_to(step):
    """Advance the real policy, recording every agent's path."""
    world = CoverageWorld(DENSITY, SEED)
    model = PPO.load(os.path.join(_CODE, "phase5_marl", "results",
                                  f"marl_d{int(DENSITY*100)}.zip"),
                     device="cpu")
    model.set_random_seed(90000 + SEED)
    obs = world.observations()
    n = len(world.positions)
    paths = [[tuple(p)] for p in world.positions]
    while world.timestep < step:
        actions, _ = model.predict(obs, deterministic=False)
        _, done = world.step(np.atleast_1d(actions))
        obs = world.observations()
        for i in range(n):
            paths[i].append(tuple(world.positions[i]))
        if done:
            break
    return world, paths


def main():
    """Draw one real episode: obstacles, swept ground, paths, sensing window.

    Nothing here is illustrative. The layout is a seeded map the experiments
    ran on, the paths are the trained policy acting on its own observations,
    and the swept region is the same array the coverage metric counts.
    """
    world, paths = run_to(SNAPSHOT)
    obst = world.obstacle_map.astype(bool)
    seen = world.visited & ~obst

    fig, ax = plt.subplots(figsize=(COLWIDTH, 2.00))

    ax.imshow(np.where(seen, 1.0, np.nan), cmap="Blues", vmin=0, vmax=5.0,
              origin="upper", interpolation="nearest", zorder=1)
    ax.imshow(np.where(obst, 1.0, np.nan), cmap="Greys", vmin=0, vmax=1.7,
              origin="upper", interpolation="nearest", zorder=2)

    for dr, dc in START_OFFSETS:
        ax.plot(HOME[1] + dc, HOME[0] + dr, marker="o", ms=3.6,
                mfc="white", mec="0.3", mew=0.9, zorder=4)
    ax.plot(HOME[1], HOME[0], marker="*", ms=9, color="#111111",
            mec="white", mew=0.6, zorder=6)

    for i, p in enumerate(paths):
        r = np.array([q[0] for q in p], dtype=float)
        c = np.array([q[1] for q in p], dtype=float)
        ax.plot(c, r, lw=0.9, color=TRAIL[i], alpha=0.95, zorder=5,
                solid_capstyle="round")
        ax.plot(c[-1], r[-1], marker="o", ms=4.6, color=TRAIL[i],
                mec="white", mew=0.8, zorder=7)

    fr, fc = paths[FOCUS][-1]
    side = 2 * HALF + 1
    ax.add_patch(Rectangle((fc - HALF - 0.5, fr - HALF - 0.5), side, side,
                           fill=False, ec="#111111", lw=1.2, zorder=8))

    box = dict(boxstyle="round,pad=0.2", fc="white", ec="0.55", lw=0.5,
               alpha=0.92)
    ax.annotate(r"$7\times7$ window", xy=(fc + HALF + 0.5, fr + HALF + 0.5),
                xytext=(9, -11), textcoords="offset points", fontsize=6.8,
                ha="left", va="top", zorder=10, bbox=box,
                arrowprops=dict(arrowstyle="-", lw=0.7, color="0.2"))

    y0, x0, ln = G - 3.0, 3.0, 20.0
    ax.add_patch(Rectangle((x0 - 1.5, y0 - 4.6), ln + 3.0, 6.4,
                           fc="white", ec="0.6", lw=0.5, alpha=0.92,
                           zorder=9))
    ax.plot([x0, x0 + ln], [y0, y0], lw=1.8, color="#111111", zorder=10)
    for x in (x0, x0 + ln):
        ax.plot([x, x], [y0 - 1.0, y0 + 1.0], lw=1.2, color="#111111",
                zorder=10)
    ax.text(x0 + ln / 2, y0 - 1.2, f"{ln*CELL_M:.0f} m", fontsize=6.8,
            ha="center", va="bottom", zorder=11)

    ax.set_xlim(-0.5, G - 0.5)
    ax.set_ylim(G - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_linewidth(0.7)
        s.set_color("0.3")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    keys = [
        Patch(fc="#8C8C8C", ec="none", label="obstacle"),
        Patch(fc="#D3E3F3", ec="none", label="swept"),
        Line2D([], [], ls="none", marker="*", ms=8, color="#111111",
               label="HOME"),
        Line2D([], [], ls="none", marker="o", ms=4, mfc="white",
               mec="0.3", mew=0.9, label="launch position"),
        Line2D([], [], ls="none", marker="o", ms=4.4, color="#4878A8",
               mec="white", mew=0.8,
               label=f"UAV at $t={SNAPSHOT}$"),
    ]
    ax.legend(handles=keys, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, fontsize=6.8, frameon=False, handletextpad=0.5,
              columnspacing=1.1, labelspacing=0.35, borderpad=0.1)

    out = os.path.join(FIGDIR, "methodology", "scenario_arena.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"t = {world.timestep}, coverage {world.coverage():.1f}%, "
          f"free cells {world.free_total}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
