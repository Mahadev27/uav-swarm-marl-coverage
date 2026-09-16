"""
Builds the obstacle layouts every controller is tested on.

This module is the reason the whole comparison works. Boids, APF, the PPO
hybrid and the MARL policy all import this same generator and pass it the
same seed, so for any given seed all four face a byte-identical map. That is
what licenses paired statistics: the difference between two controllers on
seed 7 is a difference in behaviour, not in how hard seed 7 happened to be.

Layouts are built from rectangular blocks three to eight cells on a side
rather than scattered single cells. Early versions scattered individual
obstacles at random, which produced a kind of noise field that no real search
area resembles and that no controller could navigate meaningfully. Buildings
create corridors, dead ends and lines of sight, which is what makes the
difference between controllers visible at all.

Determinism matters here and is easy to lose. Everything is drawn from
np.random.default_rng(seed) and nothing touches global numpy state, so a seed
gives the same map on any machine and in any order. An earlier version of the
project drew layouts from the simulator's internal RNG instead, and one seed
produced three different maps across three runs -- 503, 510 and 512 obstacle
cells. Every paired test in the project would have been invalid. Check seed 7
at 20% density gives 1,991 free cells before trusting anything downstream.

A five-cell safe radius around HOME is always kept clear, so the swarm never
starts inside a building.
"""

import numpy as np

HOME = (25, 25)          # launch point, grid centre
SAFE_RADIUS = 5          # cells around HOME kept clear of obstacles

def generate_structured_obstacles(grid_size=50, obstacle_density=0.1,
                                   seed=0, style="urban"):
    """One layout as a boolean grid, True where an obstacle is.

    style="urban" is what every reported result uses: rectangular building
    blocks. "indoor" produces walls and L- and U-shaped structures; "mixed"
    is half of each. Both alternatives exist for the environment-design
    comparison and are not used in the main experiment.
    """
    rng = np.random.default_rng(seed)
    G   = grid_size
    obs = np.zeros((G, G), dtype=bool)

    # Keep a square around HOME clear, so no agent starts inside a wall.
    safe = set()
    for dr in range(-SAFE_RADIUS, SAFE_RADIUS+1):
        for dc in range(-SAFE_RADIUS, SAFE_RADIUS+1):
            r, c = HOME[0]+dr, HOME[1]+dc
            if 0 <= r < G and 0 <= c < G:
                safe.add((r, c))

    # Target cell count. The placers stop once they reach it, so the
    # realised density lands close to but not exactly on the requested one --
    # a block is placed whole or not at all.
    target = int(G * G * obstacle_density)

    if style == "urban":
        _place_urban(obs, rng, G, safe, target)
    elif style == "indoor":
        _place_indoor(obs, rng, G, safe, target)
    else:
        _place_urban(obs, rng, G, safe, target // 2)
        _place_indoor(obs, rng, G, safe, target // 2)

    return obs


def _place_urban(obs, rng, G, safe, target):
    """Drop non-overlapping rectangular buildings until the target is met.

    The attempt cap matters at high density: once the grid is crowded, most
    proposed rectangles collide with something and the loop would otherwise
    spin forever. Hitting the cap simply means the realised density lands a
    little under the requested one.
    """
    placed = 0
    attempts = 0

    while placed < target and attempts < 5000:
        attempts += 1

        h = int(rng.integers(3, 9))
        w = int(rng.integers(3, 9))

        r = int(rng.integers(1, G - h - 1))
        c = int(rng.integers(1, G - w - 1))

        cells = [(r+dr, c+dc)
                 for dr in range(h)
                 for dc in range(w)]

        if any(p in safe for p in cells):
            continue
        if any(obs[p] for p in cells):
            continue

        for (br, bc) in cells:
            obs[br, bc] = True
            placed += 1

        if placed >= target:
            break


def _place_indoor(obs, rng, G, safe, target):
    """Walls and L- and U-shaped structures, for the indoor style.

    Four shape types: horizontal wall, vertical wall, L, and U. Not used by
    any reported result; kept for the environment-design comparison.
    """
    placed = 0
    attempts = 0

    while placed < target and attempts < 5000:
        attempts += 1

        shape_type = rng.integers(0, 4)

        if shape_type == 0:
            length = int(rng.integers(6, 13))
            r = int(rng.integers(2, G-2))
            c = int(rng.integers(2, G-length-2))
            cells = [(r, c+dc) for dc in range(length)]

        elif shape_type == 1:
            length = int(rng.integers(6, 13))
            r = int(rng.integers(2, G-length-2))
            c = int(rng.integers(2, G-2))
            cells = [(r+dr, c) for dr in range(length)]

        elif shape_type == 2:
            length = int(rng.integers(5, 10))
            r = int(rng.integers(2, G-length-2))
            c = int(rng.integers(2, G-length-2))
            arm = int(rng.integers(4, 8))
            cells = ([(r+dr, c) for dr in range(length)] +
                     [(r+length-1, c+dc) for dc in range(1, arm)])

        else:
            h = int(rng.integers(4, 8))
            w = int(rng.integers(4, 8))
            r = int(rng.integers(2, G-h-2))
            c = int(rng.integers(2, G-w-2))
            cells = ([(r+dr, c)   for dr in range(h)] +
                     [(r+dr, c+w) for dr in range(h)] +
                     [(r, c+dc)   for dc in range(1, w)])

        # L and U shapes can run off the grid; clip before testing.
        cells = [(br, bc) for (br, bc) in cells
                 if 0 <= br < G and 0 <= bc < G]

        if any(p in safe for p in cells):
            continue
        if any(obs[p] for p in cells):
            continue

        for (br, bc) in cells:
            obs[br, bc] = True
            placed += 1

        if placed >= target:
            break


def visualise_obstacle_map(obs, title="Obstacle Map"):
    """Quick look at a single layout. Not used by any reported figure."""
    import matplotlib.pyplot as plt

    G   = obs.shape[0]
    fig, ax = plt.subplots(figsize=(7, 7))

    for r in range(G):
        for c in range(G):
            if obs[r, c]:
                ax.add_patch(plt.Rectangle(
                    (c-0.5, r-0.5), 1, 1,
                    color="#555555", zorder=1))

    ax.plot(HOME[1], HOME[0], "*", color="#FFD700",
            markersize=18, markeredgecolor="#333",
            zorder=5, label="HOME")

    ax.set_xlim(-0.5, G-0.5)
    ax.set_ylim(-0.5, G-0.5)
    ax.set_aspect("equal")
    ax.grid(True, color="#EEEEEE", linewidth=0.4, zorder=0)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    configs = [
        (0.1, "urban",  "10% Obstacles — Urban Style"),
        (0.2, "urban",  "20% Obstacles — Urban Style"),
        (0.2, "indoor", "20% Obstacles — Indoor Style"),
        (0.3, "mixed",  "30% Obstacles — Mixed Style"),
    ]

    for ax, (density, style, title) in zip(axes.flatten(), configs):
        obs = generate_structured_obstacles(
            grid_size=50, obstacle_density=density,
            seed=42, style=style)

        pct = np.sum(obs) / (50*50) * 100
        for r in range(50):
            for c in range(50):
                if obs[r, c]:
                    ax.add_patch(plt.Rectangle(
                        (c-0.5, r-0.5), 1, 1,
                        color="#555555", zorder=1))

        ax.plot(HOME[1], HOME[0], "*", color="#FFD700",
                markersize=16, markeredgecolor="#333", zorder=5)
        ax.set_xlim(-0.5, 49.5)
        ax.set_ylim(-0.5, 49.5)
        ax.set_aspect("equal")
        ax.grid(True, color="#EEEEEE", linewidth=0.3)
        ax.set_title(f"{title}\nActual density: {pct:.1f}%", fontsize=11)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    plt.suptitle("Structured Obstacle Environments — Building & Wall Layouts",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("structured_obstacles_preview.png", dpi=150, bbox_inches="tight")
    print("Saved structured_obstacles_preview.png")
    plt.show()
