"""
make_swarm_animation.py - Side-by-side animation of APF and MARL sweeping
one identical layout, for the video presentation.

Both controllers are run on the same seed, so the obstacle map is the same
map cell for cell, exactly as in the reported experiments. Coverage is read
from each controller's own accounting, so the percentages ticking up in the
animation are the same quantity Table II reports.

The condition was not chosen for its outcome. Of the fifty seeds at 10%
density, seed 27 is the one whose stored result sits closest to the reported
means for BOTH controllers at once: MARL 93.84 against a mean of 93.73, APF
92.94 against 92.52. The animation therefore shows a typical run, not a
favourable one.

Output is an animated GIF. PowerPoint plays GIFs in slideshow view without
needing an embedded video codec, which keeps the deck portable.

Run:  python3 -m analysis.make_swarm_animation      (from code/)
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import Rectangle             # noqa: E402
import numpy as np                                   # noqa: E402
from PIL import Image                                # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
for _sub in ["", "phase2_boids", "phase3_apf", "phase5_marl"]:
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                          # noqa: E402
from stable_baselines3 import PPO                     # noqa: E402
from phase5_marl.marl_env import CoverageWorld, G     # noqa: E402
from phase3_apf.model_apf import APFModel             # noqa: E402

torch.set_num_threads(1)

DENSITY = 0.1
SEED = 27          # most representative of the 50, see docstring
MAX_STEPS = 700
EVERY = 12                       # sample every Nth step into a frame
OUT = os.path.join(os.path.dirname(_CODE), "latex_final", "figures",
                   "results", "swarm_animation.gif")

TRAIL = ["#4878A8", "#D64550", "#1F9E89", "#E8A33D", "#8E6FB5", "#5C9E5C"]
TEAL, CORAL, INK = "#1F9E89", "#D64550", "#22303C"


def run_marl():
    """Run the trained policy, snapshotting every EVERY-th step into a frame."""
    world = CoverageWorld(DENSITY, SEED)
    model = PPO.load(os.path.join(_CODE, "phase5_marl", "results",
                                  f"marl_d{int(DENSITY*100)}.zip"),
                     device="cpu")
    model.set_random_seed(90000 + SEED)
    obs = world.observations()
    frames = []
    while world.timestep < MAX_STEPS:
        if world.timestep % EVERY == 0:
            frames.append(([tuple(p) for p in world.positions],
                           world.visited.copy(), world.coverage()))
        a, _ = model.predict(obs, deterministic=False)
        _, done = world.step(np.atleast_1d(a))
        obs = world.observations()
        if done:
            break
    frames.append(([tuple(p) for p in world.positions],
                   world.visited.copy(), world.coverage()))
    return frames, world.obstacle_map.astype(bool)


def run_apf():
    """Same for APF, on the same seed so the obstacle map is identical."""
    m = APFModel(n_agents=6, obstacle_density=DENSITY, obstacle_seed=SEED)
    frames = []
    for t in range(MAX_STEPS):
        if t % EVERY == 0:
            ag = [a for a in m.agents if hasattr(a, "cells_visited")]
            vis = (m.coverage_grid == 1)
            free = (m.obstacle_grid == 0)
            cov = 100.0 * (vis & free).sum() / free.sum()
            frames.append(([a.pos for a in ag[:6]], vis.copy(), cov))
        m.step()
    ag = [a for a in m.agents if hasattr(a, "cells_visited")]
    vis = (m.coverage_grid == 1)
    free = (m.obstacle_grid == 0)
    frames.append(([a.pos for a in ag[:6]], vis.copy(),
                   100.0 * (vis & free).sum() / free.sum()))
    return frames


def panel(ax, title, colour, obst, visited, positions, cov):
    """Draw one side of one frame: swept ground, obstacles, agents, coverage."""
    ax.clear()
    ax.imshow(np.where(visited & ~obst, 1.0, np.nan), cmap="Blues",
              vmin=0, vmax=5.0, origin="upper", interpolation="nearest")
    ax.imshow(np.where(obst, 1.0, np.nan), cmap="Greys",
              vmin=0, vmax=1.7, origin="upper", interpolation="nearest")
    for i, (r, c) in enumerate(positions):
        ax.plot(c, r, marker="o", ms=5.5, color=TRAIL[i],
                mec="white", mew=0.9)
    ax.set_xlim(-0.5, G - 0.5)
    ax.set_ylim(G - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_linewidth(1.2); s.set_color(colour)
    ax.set_title(title, fontsize=13, color=colour, fontweight="bold", pad=6)
    ax.text(0.5, -0.06, f"{cov:5.1f}% covered", transform=ax.transAxes,
            ha="center", va="top", fontsize=15, color=INK,
            fontweight="bold", family="monospace")


def main():
    """Run both controllers, render the frames, write the GIF."""
    print("running MARL ...", flush=True)
    marl, obst = run_marl()
    print("running APF ...", flush=True)
    apf = run_apf()
    n = min(len(marl), len(apf))
    print(f"{n} frames", flush=True)

    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["DejaVu Serif"]})
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.35), dpi=100)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.10,
                        wspace=0.08)

    images = []
    for k in range(n):
        panel(axes[0], "APF  (classical)", CORAL, obst, apf[k][1],
              apf[k][0], apf[k][2])
        panel(axes[1], "MARL  (ours)", TEAL, obst, marl[k][1],
              marl[k][0], marl[k][2])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        images.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE,
                                                   colors=128))
    plt.close(fig)

    durations = [90] * (len(images) - 1) + [2200]      # hold the last frame
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    images[0].save(OUT, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=True)
    mb = os.path.getsize(OUT) / 1048576
    secs = sum(durations) / 1000
    print(f"\nwritten: {OUT}")
    print(f"  {len(images)} frames, {secs:.1f} s per loop, {mb:.2f} MB")
    print(f"  final coverage - APF {apf[n-1][2]:.1f}%  MARL {marl[n-1][2]:.1f}%")


if __name__ == "__main__":
    main()
