"""
The Artificial Potential Fields baseline, and the strongest of the three
comparators.

Khatib's method treats obstacles as repulsive charges and the goal as
attractive. Adapted for coverage, there is no single goal, so the attractive
term becomes a preference for unswept ground and the repulsive term keeps
agents off walls and away from each other.

Each agent scores the eight neighbouring cells on four terms:

  coverage    +5 nobody has swept it, +1 swept but not by me, -2 I was here
  repulsion   inverse-square from every obstacle within rep_radius
  separation  distance to the nearest other agent, saturating at 5 cells
  jitter      +/- 0.1 to break ties

APF turns out to be hard to beat, and the report treats that as a finding
rather than an inconvenience. It clears the 90% target at every density and
is ahead of the learned policy in the most cluttered conditions. Walls and
corridors give its repulsion term structure to work with: buildings channel
agents along streets and decorrelate their paths for free.

THE ESCAPE RULE MATTERS MORE THAN IT LOOKS. Potential-field methods are known
for local minima -- an agent can find itself in a pocket where every move
scores worse than staying put, and then it sits there for the rest of the
episode. The rule at the bottom of step() fires a random move after three
stalled steps. Papers routinely leave this unspecified, and which remedy you
pick changes measured performance a lot, so it is stated explicitly here and
the parameters are swept in analysis/apf_parameter_sweep.py. The reported
configuration sits on a broad plateau rather than a lucky peak, which is what
makes it a fair comparator.

Each agent keeps a private visited_memory, distinct from the shared coverage
grid: it can tell "I have been here" apart from "somebody has been here".
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import mesa
import numpy as np
from obstacle_generator import generate_structured_obstacles

HOME         = (25, 25)
RETURN_AFTER = 600       # step at which agents head home, of 700


class ObstacleAgent(mesa.Agent):
    def __init__(self, model): super().__init__(model)
    def step(self): pass


class APFAgentMesa(mesa.Agent):
    """One UAV under potential-field control.

    k_rep and rep_radius are the two swept parameters. (1.5, 2) is the
    reported configuration; see apf_parameter_sweep.py for the surface.
    """

    def __init__(self, model, k_rep=1.5, rep_radius=2):
        super().__init__(model)
        self.k_rep          = k_rep
        self.rep_radius     = rep_radius
        self.stall          = 0
        self.escape_count   = 0
        self.cells_visited  = 0
        self.returning      = False
        # Private to this agent, unlike the model-wide coverage grid.
        self.visited_memory = set()

    def step(self):
        neighbours = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False)
        free = [n for n in neighbours
                if self.model.obstacle_grid[n[0]][n[1]] == 0]
        if not free:
            return

        other_pos = [a.pos for a in self.model.agents
                     if a is not self and hasattr(a, 'cells_visited')]

        if self.returning:
            best = min(free, key=lambda n:
                abs(n[0]-HOME[0])+abs(n[1]-HOME[1]))
        else:
            r, c = self.pos
            G    = self.model.grid.width
            best_pos, best_score = None, -999

            for pos in free:
                nr, nc = pos
                # Three-level preference: unswept by anyone beats swept-by-
                # others beats somewhere I have personally been.
                if self.model.coverage_grid[nr][nc] == 0:
                    cov_score = 5.0
                elif pos not in self.visited_memory:
                    cov_score = 1.0
                else:
                    cov_score = -2.0

                # Inverse-square repulsion from every obstacle in radius.
                # The 1e-6 avoids a divide-by-zero when dr = dc = 0.
                rep_score = 0.0
                for dr in range(-self.rep_radius, self.rep_radius+1):
                    for dc in range(-self.rep_radius, self.rep_radius+1):
                        ar, ac = nr+dr, nc+dc
                        if 0<=ar<G and 0<=ac<G:
                            if self.model.obstacle_grid[ar][ac]==1:
                                dist = np.sqrt(dr**2+dc**2)+1e-6
                                if dist < self.rep_radius:
                                    rep_score -= self.k_rep/(dist**2)

                sep_score = 1.0
                if other_pos:
                    min_d = min(abs(nr-op[0])+abs(nc-op[1]) for op in other_pos)
                    sep_score = min(min_d, 5)/5.0

                # Repulsion is weighted at 0.3: enough to round corners,
                # not enough to override the drive towards new ground.
                score = cov_score + 0.3*rep_score + sep_score + \
                        self.random.uniform(-0.1, 0.1)
                if score > best_score:
                    best_score, best_pos = score, pos
            best = best_pos

        if best is None:
            return

        # Local-minimum escape. Without this an agent can sit in a pocket
        # for the rest of the episode. Three strikes, then jump to a random
        # free neighbour, preferring an unswept one.
        if best == self.pos:
            self.stall += 1
            if self.stall >= 3:
                unvisited_free = [n for n in free
                    if self.model.coverage_grid[n[0]][n[1]]==0]
                if unvisited_free:
                    best = unvisited_free[self.random.randrange(len(unvisited_free))]
                elif free:
                    best = free[self.random.randrange(len(free))]
                self.stall = 0
                self.escape_count += 1
        else:
            self.stall = 0

        self.visited_memory.add(best)
        self.model.grid.move_agent(self, best)
        if self.model.coverage_grid[best[0]][best[1]] == 0:
            self.model.coverage_grid[best[0]][best[1]] = 1
            self.cells_visited += 1


class APFModel(mesa.Model):
    def __init__(self, n_agents=6, grid_width=50, grid_height=50,
                 obstacle_density=0.0, obstacle_style="urban",
                 obstacle_seed=0):
        super().__init__(seed=obstacle_seed)
        self.grid          = mesa.space.MultiGrid(grid_width, grid_height, torus=False)
        self.coverage_grid = np.zeros((grid_width, grid_height), dtype=int)
        self.timestep      = 0

        obs_map = generate_structured_obstacles(
            grid_size=grid_width,
            obstacle_density=obstacle_density,
            seed=obstacle_seed,
            style=obstacle_style)

        self.obstacle_grid = obs_map.astype(int)

        for ox in range(grid_width):
            for oy in range(grid_height):
                if self.obstacle_grid[ox][oy] == 1:
                    a = ObstacleAgent(self)
                    self.grid.place_agent(a, (ox, oy))
                    self.coverage_grid[ox][oy] = -1

        self.coverage_grid[HOME[0]][HOME[1]] = 1
        for _ in range(n_agents):
            a = APFAgentMesa(self)
            self.grid.place_agent(a, HOME)

        self.datacollector = mesa.DataCollector(
            model_reporters={"Coverage_%": lambda m: (
                np.sum(m.coverage_grid==1) /
                max(1, np.sum(m.coverage_grid>=0)))*100})

    def step(self):
        self.datacollector.collect(self)
        self.timestep += 1
        if self.timestep >= RETURN_AFTER:
            for a in self.agents:
                if hasattr(a, 'returning'):
                    a.returning = True
        self.agents.do("step")
