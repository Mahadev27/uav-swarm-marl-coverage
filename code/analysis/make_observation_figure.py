"""
Draws the 7x7 observation-window diagram -- Figure 3 in the report.

Illustrative rather than derived from a run: it shows what the encoding means
rather than a particular agent's view at a particular moment. Values are the
same for every controller; only the learned policy additionally receives the
nine memory features, which are not shown here.

The encoding, matching marl_env.observation():

    -1.0   obstacle, or off-grid  (an agent cannot tell these apart)
     0.0   free and unswept
     1.0   swept by any agent
     2.0   another agent

The caption in the report notes the window spans 21 m at 3 m per cell, which
is the point the figure exists to make: 21 m of a 150 m block.

Run:  python3 -m analysis.make_observation_figure      (from code/)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.join(os.path.dirname(__file__),
                       "../../figures/methodology")
os.makedirs(OUT_DIR, exist_ok=True)

# Palette shared with the report figures, so diagram and plots agree.
C_OBSTACLE = "#3D4852"
C_VISITED  = "#B5D4ED"
C_FREE     = "#F0F2F4"
C_SELF     = "#E63946"
C_OTHER    = "#2A9D8F"

window = np.array([
    [ 0, -1, -1,  0,  0,  1,  1],
    [ 0,  0,  1,  1,  1,  0,  0],
    [ 1,  1,  1,  2,  0,  0, -1],
    [ 0,  0,  1,  9,  0,  0,  0],
    [ 0, -1, -1,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  1,  1,  0],
    [ 0,  0,  1,  1,  0,  0,  0],
])

fig, (ax, ax_legend) = plt.subplots(
    1, 2, figsize=(11, 5.5),
    gridspec_kw={"width_ratios": [1.15, 1]})
fig.patch.set_facecolor("white")

for r in range(7):
    for c in range(7):
        v = window[r, c]
        if   v == -1: fc, txt, tc = C_OBSTACLE, "-1",  "white"
        elif v ==  1: fc, txt, tc = C_VISITED,  "1.0", "#2C3440"
        elif v ==  2: fc, txt, tc = C_OTHER,    "2",   "white"
        elif v ==  9: fc, txt, tc = C_SELF,     "UAV", "white"
        else:         fc, txt, tc = C_FREE,     "0",   "#6B7280"

        ax.add_patch(plt.Rectangle(
            (c, 6-r), 1, 1,
            facecolor=fc, edgecolor="white", linewidth=1.5, zorder=2))
        ax.text(c+0.5, 6-r+0.5, txt, ha="center", va="center",
                fontsize=8.5, fontweight="bold" if v == 9 else "normal",
                color=tc, zorder=3)

ax.add_patch(plt.Rectangle(
    (0, 0), 7, 7, fill=False,
    edgecolor=C_SELF, linewidth=2.2, linestyle="--", zorder=4))

ax.set_xlim(-0.4, 7.4)
ax.set_ylim(-0.4, 7.4)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("$7\\times7$ Local Observation Window\n"
             "3-cell detection radius (9 m at 3 m/cell)",
             fontsize=12, fontweight="bold", pad=12)

ax_legend.axis("off")

handles = [
    mpatches.Patch(facecolor=C_SELF,     label="Observing UAV (window centre)"),
    mpatches.Patch(facecolor=C_OTHER,    label="Another UAV  $\\rightarrow$  encoded 2.0"),
    mpatches.Patch(facecolor=C_OBSTACLE, label="Obstacle  $\\rightarrow$  encoded $-1.0$"),
    mpatches.Patch(facecolor=C_VISITED,  edgecolor="#C8CDD2",
                   label="Already covered  $\\rightarrow$  encoded 1.0"),
    mpatches.Patch(facecolor=C_FREE, edgecolor="#C8CDD2",
                   label="Uncovered  $\\rightarrow$  encoded 0.0"),
    plt.Line2D([0], [0], color=C_SELF, linestyle="--", linewidth=2.2,
               label="Sensor detection limit"),
]
ax_legend.legend(handles=handles, loc="upper left", fontsize=10,
                 frameon=False, bbox_to_anchor=(0.0, 0.95))

ax_legend.text(
    0.0, 0.34,
    "Observation vector passed to the policy:\n\n"
    "    49 window values  +  2 position values\n"
    "    =  51 floats\n\n"
    "Position is normalised to $[0,1]$ by dividing\n"
    "row and column by the grid size.\n\n"
    "Cells outside the grid boundary are encoded\n"
    "as obstacles ($-1.0$), so the swarm treats the\n"
    "search-area edge the same way it treats a wall.",
    transform=ax_legend.transAxes, fontsize=9.5,
    va="top", ha="left", color="#374151", linespacing=1.5)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "fig_uav_observation_window.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved -> {out_path}")
