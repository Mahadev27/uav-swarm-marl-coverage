"""
The Boids baseline: flocking rules adapted for area coverage.

Reynolds' original Boids has three rules -- separation, alignment, cohesion --
and produces flocking. Flocking is the wrong behaviour for coverage: a tight
flock sweeps the same cells repeatedly. Cohesion is therefore dropped and
separation kept, so agents push apart and spread out. Alignment has no
meaning on a grid with no velocity vector.

What remains is a local greedy rule. Each agent scores the eight cells around
it on two terms -- is this cell unswept, and how far is it from the nearest
other agent -- and moves to the best one. A small random tiebreak keeps two
agents in identical situations from making identical moves forever.

Agents step sequentially within a Mesa timestep, so each one sees where the
others have already moved this tick. That produces spatial separation without
communication, and it is a slightly weaker decentralisation claim than the
MARL environment makes, where all six move simultaneously.

The layout comes from the shared obstacle generator with the supplied seed,
so a given seed produces the same map here as it does for every other
controller. That is what makes the paired comparison valid.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import mesa
import numpy as np
from obstacle_generator import generate_structured_obstacles

HOME         = (25, 25)
RETURN_AFTER = 600       # step at which agents head home, of 700


# Obstacles are Mesa agents purely so they occupy grid cells and show up in
# neighbourhood queries. They never do anything.
class ObstacleAgent(mesa.Agent):
    def __init__(self, model): super().__init__(model)
    def step(self): pass


class UAVAgent(mesa.Agent):
    """One UAV. Moves one cell per step by local greedy scoring."""
    def __init__(self, model):
        super().__init__(model)
        self.cells_visited = 0
        self.returning     = False

    def step(self):
        neighbours = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False)
        free = [n for n in neighbours
                if self.model.obstacle_grid[n[0]][n[1]] == 0]
        if not free:
            return

        # Positions of the other UAVs, read live. Because Mesa steps agents
        # in sequence, agents that have already moved this tick are seen at
        # their new position -- which is what produces the separation.
        other_pos = [a.pos for a in self.model.agents
                     if a is not self and hasattr(a, 'cells_visited')]

        # After RETURN_AFTER, abandon coverage and head for HOME. Models a
        # real mission reserving fuel for the return leg.
        if self.returning:
            best = min(free, key=lambda n:
                abs(n[0]-HOME[0])+abs(n[1]-HOME[1]))
        else:
            best_pos, best_score = None, -999
            for pos in free:
                nr, nc    = pos
                # +2 for new ground, -1 for already swept: the coverage
                # drive. Note it reads the shared coverage grid, so an agent
                # can tell ground was swept but not by whom.
                cov_score = 2.0 if self.model.coverage_grid[nr][nc]==0 else -1.0
                # Separation, saturating at 5 cells: beyond that, being
                # further from the others stops being worth anything.
                sep_score = 1.0
                if other_pos:
                    min_d     = min(abs(nr-op[0])+abs(nc-op[1]) for op in other_pos)
                    sep_score = min(min_d, 5)/5.0
                # The jitter breaks ties. Without it two agents in mirror
                # situations make identical moves indefinitely.
                score = cov_score + sep_score + self.random.uniform(-0.1, 0.1)
                if score > best_score:
                    best_score, best_pos = score, pos
            best = best_pos

        if best is None:
            return

        self.model.grid.move_agent(self, best)
        if self.model.coverage_grid[best[0]][best[1]] == 0:
            self.model.coverage_grid[best[0]][best[1]] = 1
            self.cells_visited += 1


class SwarmCoveragePhase2(mesa.Model):
    """The Boids world. Same grid, same obstacles and same HOME as every
    other controller, so results are directly comparable."""
    def __init__(self, n_agents=6, grid_width=50, grid_height=50,
                 obstacle_density=0.0, obstacle_style="urban",
                 obstacle_seed=0):
        # Seeding the Mesa model from the same seed as the obstacle map is
        # what makes a run reproducible. An earlier version did not pass the
        # seed through here, and one seed produced three different layouts.
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
                    # -1 marks an obstacle, so it is excluded from both the
                    # numerator and denominator of the coverage percentage.
                    self.coverage_grid[ox][oy] = -1

        # HOME counts as swept from the start; all six launch from it.
        self.coverage_grid[HOME[0]][HOME[1]] = 1
        for _ in range(n_agents):
            a = UAVAgent(self)
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
