"""
The single-agent PPO hybrid: one learned agent alongside five hand-written
greedy ones.

This architecture is common enough in the literature to be worth testing, and
the report treats what it found as a finding rather than a failed attempt.
Two things are wrong with it, and both are measurable.

FIRST, IT CANNOT ANSWER THE QUESTION. Only agent 0 is under policy control;
agents 1-5 run a fixed greedy rule. Replacing the learned agent with an inert
one, a random one, a greedy one and the trained one gives swarm coverage of
82.21%, 82.37%, 84.07% and 84.83%. The trained agent is the best of the four,
so it had learned something -- but the entire range spanned by completely
different behaviours in that slot is 1.86 points, against a 9.37-point gap to
APF. No agent in that position could have closed it. The deficit belongs to
the five scripted agents, and no conclusion about PPO follows from it. See
analysis/greedy_ablation.py.

SECOND, THE REWARD IS SHARED. Look at the reward line in step(): it pays on
`new_cells`, the count of cells newly swept by the WHOLE swarm, not by the
learned agent. So the single agent is credited for ground the five scripted
ones covered, which is precisely the credit-assignment problem quantified in
phase5_marl/train_robustness.py -- and worth 30.6 percentage points there.

The fully decentralised architecture in phase5_marl removes both problems.
This environment is kept because the comparison is part of the result.

Observation is 51-dimensional: a 7x7 window plus normalised position. Note
there are no memory features here, unlike the MARL environment.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import gymnasium as gym
from gymnasium import spaces

HOME      = (25, 25)
G         = 50
N_UAVS    = 6
MAX_STEPS = 700
ACTIONS   = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]


def _build_obstacles(density, seed):
    """Layout from the shared generator, with a scattered-cell fallback.

    The fallback exists only so this file can be imported standalone. It
    produces a different kind of environment -- random single cells rather
    than buildings -- so any result generated through it would not be
    comparable. In practice the import always succeeds.
    """
    try:
        from phase2_boids.obstacle_generator import generate_structured_obstacles
        return generate_structured_obstacles(
            grid_size=G, obstacle_density=density,
            seed=seed, style="urban").astype(bool)
    except Exception:
        rng = np.random.default_rng(seed)
        obs = np.zeros((G,G), dtype=bool)
        safe = {(HOME[0]+dr, HOME[1]+dc)
                for dr in range(-5,6) for dc in range(-5,6)}
        n = int(G*G*density)
        placed = 0
        while placed < n:
            r,c = int(rng.integers(0,G)), int(rng.integers(0,G))
            if not obs[r,c] and (r,c) not in safe:
                obs[r,c]=True; placed+=1
        return obs


class UAVCoverageEnv(gym.Env):
    """Gymnasium env where the policy controls agent 0 and scripts the rest."""

    metadata = {"render_modes": []}

    def __init__(self, obstacle_density=0.0, seed=None):
        super().__init__()
        self.obstacle_density = obstacle_density
        self._ep_seed         = seed if seed is not None else 0

        # 49 window cells + 2 position. No memory features: those exist only
        # in the MARL environment, which is part of why the two are not
        # directly comparable.
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(51,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)

        self.obstacle_map = np.zeros((G,G), dtype=bool)
        self.visited      = np.zeros((G,G), dtype=bool)
        self.visit_count  = np.zeros((G,G), dtype=int)
        self.positions    = [[HOME[0],HOME[1]]]*N_UAVS
        self.timestep     = 0
        self._free        = G*G

        self._rng         = np.random.default_rng(0)

    def reset(self, seed=None, options=None):
        if seed is not None:
            s = int(seed)
        else:
            # NOTE the modulo. This was a defect: it mapped episode seeds onto
            # 0-49, so Seed Set B (50-99) resolved to Set A's fifty layouts and
            # the held-out test was not held out at all. Found during the audit
            # and repaired; results are regenerated. Kept here as the cycling
            # behaviour training relies on, but evaluation always passes an
            # explicit seed rather than relying on it.
            s = self._ep_seed % 50
            self._ep_seed += 1
        self._rng = np.random.default_rng(s)

        self.obstacle_map = _build_obstacles(self.obstacle_density, s)
        self._free        = max(1, int(np.sum(~self.obstacle_map)))
        self.visited      = np.zeros((G,G), dtype=bool)
        self.visit_count  = np.zeros((G,G), dtype=int)
        self.visited[HOME] = True

        # Same launch geometry as the MARL environment, so the two start
        # from identical positions on identical maps.
        offsets = [(-8,0),(8,0),(0,-8),(0,8),(-5,-5),(5,5)]
        self.positions = []
        for dr,dc in offsets:
            r = max(1,min(G-2,HOME[0]+dr))
            c = max(1,min(G-2,HOME[1]+dc))
            if self.obstacle_map[r,c]: r,c = HOME
            self.positions.append([r,c])
            self.visited[r,c] = True
            self.visit_count[r,c] += 1

        self.timestep = 0
        return self._get_obs(0), {}

    def step(self, action):
        """Agent 0 takes the policy action; agents 1-5 act greedily."""
        prev = int(np.sum(self.visited))

        self._move(0, int(action))

        # The five scripted agents: greedy on unswept ground plus a
        # separation term. This is close to the Boids rule, and it is these
        # five that determine the hybrid's ceiling.
        for i in range(1, N_UAVS):
            r,c = self.positions[i]
            best_a, best_s = 0, -999
            for a,(dr,dc) in enumerate(ACTIONS):
                nr,nc = r+dr, c+dc
                if 0<=nr<G and 0<=nc<G and not self.obstacle_map[nr,nc]:
                    unvisited = not self.visited[nr,nc]
                    dists = [abs(nr-self.positions[j][0])+
                               abs(nc-self.positions[j][1])
                               for j in range(N_UAVS) if j!=i]
                    sep = min(dists+[5])/5.0
                    s = (5.0 if unvisited else -2.0) + sep + \
                        self._rng.uniform(-0.05, 0.05)
                    if s > best_s:
                        best_s, best_a = s, a
            self._move(i, best_a)

        new_cells     = int(np.sum(self.visited)) - prev
        self.timestep += 1
        cov    = self.get_coverage()
        # new_cells is swarm-wide, so the learned agent is paid for ground
        # the five scripted agents covered. This is the shared-reward credit-
        # assignment problem, and it is measured directly in
        # phase5_marl/train_robustness.py.
        reward = float(new_cells)*3.0 - 0.05 + (cov/100.0)*0.1
        done   = self.timestep >= MAX_STEPS or cov >= 99.0

        return self._get_obs(0), reward, done, False, {"coverage": cov}

    def _move(self, i, action):
        r,c   = self.positions[i]
        dr,dc = ACTIONS[action]
        nr,nc = r+dr, c+dc
        if 0<=nr<G and 0<=nc<G and not self.obstacle_map[nr,nc]:
            self.positions[i] = [nr,nc]
            self.visited[nr,nc] = True
            self.visit_count[nr,nc] += 1

    def _get_obs(self, i):
        """7x7 window plus normalised position, 51 values.

        Encoding matches the MARL environment: -1 obstacle or off-grid, 0
        free and unswept, 1 swept, 2 another agent.
        """
        r,c  = self.positions[i]
        half = 3
        obs  = np.full(49, -1.0, dtype=np.float32)
        idx  = 0
        for dr in range(-half, half+1):
            for dc in range(-half, half+1):
                nr,nc = r+dr, c+dc
                if 0<=nr<G and 0<=nc<G:
                    obs[idx] = (-1.0 if self.obstacle_map[nr,nc]
                                else (1.0 if self.visited[nr,nc] else 0.0))
                idx += 1
        for j,pos in enumerate(self.positions):
            if j != i:
                adr,adc = pos[0]-r, pos[1]-c
                if -half<=adr<=half and -half<=adc<=half:
                    obs[(adr+half)*7+(adc+half)] = 2.0
        return np.append(obs,[r/G,c/G]).astype(np.float32)

    def get_coverage(self):
        return float(np.sum(self.visited))/self._free*100.0

    def get_redundancy(self):
        """Share of visits that were to already-swept cells."""
        v = self.visit_count[self.visit_count > 0]
        if v.size == 0:
            return 0.0
        return float((v.sum() - v.size) / v.sum())
