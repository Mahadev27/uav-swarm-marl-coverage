"""
The multi-agent coverage environment, and the vectorised wrapper that lets
Stable-Baselines3 train a single shared policy on it.

The task: six UAVs sweep a 50x50 grid of unknown ground, 3 m per cell, so
150 m by 150 m. They start from one launch point, get 700 timesteps, and are
scored on the fraction of obstacle-free cells at least one of them has
entered.

The constraint that makes it interesting: no agent sends or receives a
message, and no agent sees the whole map. Each one gets a 7x7 window around
itself, which is 21 m of a 150 m block. Any coordination has to fall out of
six agents independently reading their own patch of ground.

There are two classes here and they do different jobs:

  CoverageWorld              one simulated world holding all N agents. Steps
                             only when an action has been supplied for every
                             agent, so they all move within the same timestep.
  MultiAgentCoverageVecEnv   presents W worlds x N agents as W*N independent
                             "environments" to SB3. That is the whole trick
                             behind parameter sharing: SB3 thinks it is
                             training 48 separate agents, so one gradient
                             update draws on all of their experience, and
                             what comes out is one network every agent runs.

Read alongside train_marl.py, which drives this.
"""

import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

# The obstacle generator lives in the Boids package. Importing it rather than
# copying it is deliberate: every controller has to face byte-identical
# layouts for the paired statistics to be valid.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
for _sub in ["", "phase2_boids"]:
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phase2_boids.obstacle_generator import generate_structured_obstacles

G = 50                                 # grid side, 3 m per cell -> 150 m
N_UAVS = 6
HOME = (25, 25)                        # single launch point, grid centre
MAX_STEPS = 700                        # ~12 min of flight at 3 m/s
HALF = 3                               # sensor reaches 3 cells -> 7x7 window
N_WINDOW = (2 * HALF + 1) ** 2         # 49 sensed cells
N_EXTRA = 9                            # onboard-memory features, see below
OBS_DIM = N_WINDOW + 2 + N_EXTRA       # 49 + own position + memory = 60

# Eight-connected movement, one cell per timestep. No inertia, no turn
# radius: a first-order kinematic model and nothing more.
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1),
           (-1, -1), (-1, 1), (1, -1), (1, 1)]

# Where the six agents start, as offsets from HOME. Four sit eight cells out
# on the cardinals and two sit five-and-five on a diagonal. They deploy from
# one launch site rather than being scattered, because that is what a real
# team does.
START_OFFSETS = [(-8, 0), (8, 0), (0, -8), (0, 8), (-5, -5), (5, 5)]


def start_offsets(n):
    """Launch offsets for n agents.

    For n = 6 this returns START_OFFSETS unchanged, so the six-agent results
    are exactly reproducible. Other swarm sizes get placed on the same
    eight-cell ring at equal angular spacing, which keeps the deployment
    geometry comparable when the swarm-size sensitivity study varies n.
    """
    if n == len(START_OFFSETS):
        return list(START_OFFSETS)
    return [(int(round(8 * np.cos(2 * np.pi * k / n))),
             int(round(8 * np.sin(2 * np.pi * k / n)))) for k in range(n)]

# Reward weights. The ratio between them matters more than the values.
#
# W_COV is per cell this agent is FIRST to visit, not per cell it enters.
# That distinction is the single most consequential decision in the project:
# rewarding the shared team total instead costs 30.6 percentage points and
# no run of fifty reaches the target. See train_robustness.py.
#
# W_TIME makes one new cell worth 60 steps of loitering. Large enough that
# stalling is expensive, finite enough that a policy will still take a longer
# path to reach new ground.
#
# W_TEAM is paid once at termination and is deliberately tiny: at most 2.0
# against an accumulated ~750, about 0.3% of return. It breaks ties between
# otherwise equivalent policies without reintroducing the credit-assignment
# problem the per-agent term exists to solve.
W_COV = 3.0
W_TIME = 0.05
W_TEAM = 2.0


def build_obstacles(density, seed):
    """One layout. Deterministic given the seed, on any machine."""
    return generate_structured_obstacles(
        grid_size=G, obstacle_density=density,
        seed=int(seed), style="urban").astype(bool)


class CoverageWorld:
    """One world holding all N agents, the obstacle map and the swept record.

    Advances only when an action has been supplied for every agent, so all N
    move simultaneously within a timestep rather than in sequence. That
    matters: if agents moved one at a time, later agents would see earlier
    agents' moves within the same tick, which is a weak form of communication
    and would undercut the decentralisation claim.
    """

    def __init__(self, density, seed, n_agents=N_UAVS, use_memory=True):
        self.density = density
        self.n_agents = int(n_agents)
        # use_memory=False zeroes the nine memory features while keeping the
        # observation vector the same width. That is how the ablation in
        # train_ablation.py isolates what those features are worth.
        self.use_memory = bool(use_memory)
        self.reset(seed)

    def reset(self, seed):
        self.obstacle_map = build_obstacles(self.density, seed)
        self.free_total = max(1, int(np.sum(~self.obstacle_map)))

        # visited is the swarm's shared record: a cell is True once ANY agent
        # has entered it. visit_count tracks repeat entries, for redundancy.
        # first_visitor records who got there first, which is what the
        # per-agent reward pays on.
        self.visited = np.zeros((G, G), dtype=bool)
        self.visit_count = np.zeros((G, G), dtype=int)
        self.first_visitor = np.full((G, G), -1, dtype=int)

        # Per-agent belief about which cells it has had in sensor range.
        # Private to each agent, unlike self.visited.
        self.known = np.zeros((self.n_agents, G, G), dtype=bool)

        self.positions = []
        for dr, dc in start_offsets(self.n_agents):
            # Clamp inside the border, and fall back to HOME if the intended
            # start happens to be inside a building on this layout.
            r = max(1, min(G - 2, HOME[0] + dr))
            c = max(1, min(G - 2, HOME[1] + dc))
            if self.obstacle_map[r, c]:
                r, c = HOME
            self.positions.append([r, c])
            self._mark(r, c, len(self.positions) - 1)

        for i in range(self.n_agents):
            self._sense(i)

        self.timestep = 0
        self.path_len = np.zeros(self.n_agents, dtype=float)
        self.near_misses = 0
        return self.observations()

    def _sense(self, i):
        """Mark everything inside agent i's 7x7 window as known to i."""
        r, c = self.positions[i]
        r0, r1 = max(0, r - HALF), min(G, r + HALF + 1)
        c0, c1 = max(0, c - HALF), min(G, c + HALF + 1)
        self.known[i, r0:r1, c0:c1] = True

    def _mark(self, r, c, agent):
        """Record a visit. Returns True only if this agent got there first."""
        self.visit_count[r, c] += 1
        if not self.visited[r, c]:
            self.visited[r, c] = True
            self.first_visitor[r, c] = agent
            return True
        return False

    def step(self, actions):
        """Apply one action per agent. Returns per-agent rewards and done."""
        own_new = np.zeros(self.n_agents, dtype=float)

        for i, a in enumerate(actions):
            r, c = self.positions[i]
            dr, dc = ACTIONS[int(a)]
            nr, nc = r + dr, c + dc
            # Obstacle avoidance happens here, at the planning layer: a move
            # into a building or off the grid is simply not executed. The
            # policy never has to learn not to fly into walls, which is the
            # same deal the classical controllers get.
            if 0 <= nr < G and 0 <= nc < G and not self.obstacle_map[nr, nc]:
                self.positions[i] = [nr, nc]
                self.path_len[i] += np.hypot(dr, dc)
                if self._mark(nr, nc, i):
                    own_new[i] += 1.0
            self._sense(i)

        # Near-misses: pairs within one cell by Chebyshev distance, counted
        # after moving. NOTE this is not the definition used in the reported
        # tables. Those use Euclidean radius 1 sampled before the step, to
        # match how the classical controllers are measured. See
        # marl_secondary_metrics.py and marl_swarm_size_metrics.py, which
        # recompute this properly. This counter is kept for continuity.
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if (abs(self.positions[i][0] - self.positions[j][0]) <= 1 and
                        abs(self.positions[i][1] - self.positions[j][1]) <= 1):
                    self.near_misses += 1

        self.timestep += 1
        # 99.5% rather than 100%: a handful of cells can be walled off and
        # unreachable, and waiting for them would waste the whole budget.
        done = self.timestep >= MAX_STEPS or self.coverage() >= 99.5

        rewards = W_COV * own_new - W_TIME
        if done:
            rewards = rewards + W_TEAM * (self.coverage() / 100.0)
        return rewards, done

    def coverage(self):
        """Percentage of obstacle-free cells entered by at least one agent."""
        return 100.0 * float(np.sum(self.visited)) / self.free_total

    def redundancy(self):
        """Share of all visits that were to already-swept cells.

        0.0 means nobody ever retraced; higher means more wasted flying.
        """
        v = self.visit_count[self.visit_count > 0]
        if v.size == 0:
            return 0.0
        return float((v.sum() - v.size) / v.sum())

    def cells_per_agent(self):
        """First-visit count per agent, for the load-balance measure."""
        return [int(np.sum(self.first_visitor == i))
                for i in range(self.n_agents)]

    def _memory_features(self, i):
        """Nine numbers derived from what agent i has already sensed.

        A bare 7x7 window gives no gradient towards unexplored ground once
        the immediate neighbourhood is swept, and a policy without something
        like this performs no better than a random walk. These features stand
        in for the onboard map memory any real UAV carries.

        They are also the reason the headline comparison is not one of equal
        information: Boids and APF do not have them. The ablation measures
        exactly what they are worth.
        """
        r, c = self.positions[i]
        # Worth chasing: cells this agent has never sensed, plus cells it has
        # sensed but nobody has swept. Obstacles excluded.
        target = (~self.known[i]) | (self.known[i] & ~self.visited)
        target &= ~self.obstacle_map

        # Four scores: how much of that target area lies above, below, left
        # and right. Normalised, so they say "which way" not "how much".
        quad = np.zeros(4, dtype=np.float32)
        quad[0] = target[:r, :].sum()
        quad[1] = target[r + 1:, :].sum()
        quad[2] = target[:, :c].sum()
        quad[3] = target[:, c + 1:].sum()
        tot = max(1.0, float(target.sum()))
        quad = quad / tot

        # Direction and distance to the nearest target cell, by Manhattan
        # distance. Clipped and scaled so the network sees bounded inputs.
        rows, cols = np.nonzero(target)
        if rows.size:
            d = np.abs(rows - r) + np.abs(cols - c)
            k = int(np.argmin(d))
            dr_n = float(np.clip((rows[k] - r) / 10.0, -1.0, 1.0))
            dc_n = float(np.clip((cols[k] - c) / 10.0, -1.0, 1.0))
            dist = float(min(d[k], 50) / 50.0)
        else:
            dr_n = dc_n = 0.0
            dist = 1.0

        # The last two are internal to the agent: how much of the arena it
        # has sensed, and how much of the episode remains.
        sensed = float(self.known[i].sum()) / (G * G)
        t_left = 1.0 - self.timestep / MAX_STEPS
        return np.array([*quad, dr_n, dc_n, dist, sensed, t_left],
                        dtype=np.float32)

    def observation(self, i):
        """The 60-dimensional vector agent i acts on.

        Layout: 49 window cells, then own position normalised, then the nine
        memory features (or zeros when use_memory is False).

        Encoding: -1 obstacle or off-grid, 0 free and unswept, 1 swept,
        2 another agent. Off-grid reads the same as an obstacle, so an agent
        cannot tell the world boundary from a building.

        The swept channel is the swarm's shared record, not this agent's
        private one. An agent can tell the ground ahead was already covered
        but not by whom or when. That is indirect coordination through a
        shared trace, and the same channel is available to Boids and APF, so
        it advantages nobody. It is not message passing.
        """
        r, c = self.positions[i]
        obs = np.full(N_WINDOW, -1.0, dtype=np.float32)
        idx = 0
        for dr in range(-HALF, HALF + 1):
            for dc in range(-HALF, HALF + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < G and 0 <= nc < G:
                    obs[idx] = (-1.0 if self.obstacle_map[nr, nc]
                                else (1.0 if self.visited[nr, nc] else 0.0))
                idx += 1
        # Other agents overwrite the cell they occupy, so a neighbour is
        # visible as a 2.0 rather than as swept ground.
        for j, pos in enumerate(self.positions):
            if j != i:
                adr, adc = pos[0] - r, pos[1] - c
                if -HALF <= adr <= HALF and -HALF <= adc <= HALF:
                    obs[(adr + HALF) * (2 * HALF + 1) + (adc + HALF)] = 2.0
        tail = (self._memory_features(i) if self.use_memory
                else np.zeros(N_EXTRA, dtype=np.float32))
        return np.concatenate([obs,
                               np.array([r / G, c / G], dtype=np.float32),
                               tail]).astype(np.float32)

    def observations(self):
        return np.stack([self.observation(i) for i in range(self.n_agents)])


class MultiAgentCoverageVecEnv(VecEnv):
    """W worlds x N agents presented to SB3 as W*N independent environments.

    This is what makes parameter sharing work. SB3's PPO expects a vector of
    single-agent environments; it has no notion of a multi-agent one. By
    flattening every agent in every world into that vector, each agent's
    experience becomes one more sample for the same policy, and a single
    gradient update consumes all of them.

    The result is centralised training with decentralised execution: pooling
    experience across agents could not be done onboard, but the policy that
    comes out runs independently on each agent's own observation with nothing
    transmitted between them.

    Running W worlds in parallel rather than one is about sample diversity —
    the agents see W different layouts at once, which stops the policy
    overfitting to a single map.
    """

    metadata = {"render_modes": []}

    def __init__(self, density, n_worlds=4, seed_pool=None, use_memory=True):
        self.density = density
        self.n_worlds = n_worlds
        self.use_memory = bool(use_memory)
        # Training layouts come from Seed Set A only. Set B (50-99) is never
        # drawn here, which is what makes the held-out test genuine.
        self.seed_pool = list(seed_pool) if seed_pool is not None else list(range(50))
        self._rng = np.random.default_rng(12345)

        obs_space = spaces.Box(low=-1.0, high=2.0, shape=(OBS_DIM,),
                               dtype=np.float32)
        act_space = spaces.Discrete(len(ACTIONS))
        # num_envs = n_worlds * N_UAVS is the line that does the work.
        super().__init__(n_worlds * N_UAVS, obs_space, act_space)

        self.worlds = [CoverageWorld(density, self._draw_seed(),
                                     use_memory=self.use_memory)
                       for _ in range(n_worlds)]
        self._actions = None

    def _draw_seed(self):
        return int(self._rng.choice(self.seed_pool))

    def reset(self):
        obs = [w.reset(self._draw_seed()) for w in self.worlds]
        return np.concatenate(obs, axis=0)

    def step_async(self, actions):
        # SB3 hands back a flat vector of W*N actions; fold it into one row
        # per world so each world can step all its agents together.
        self._actions = np.asarray(actions).reshape(self.n_worlds, N_UAVS)

    def step_wait(self):
        obs_all, rew_all, done_all, info_all = [], [], [], []
        for w, acts in zip(self.worlds, self._actions):
            rewards, done = w.step(acts)
            if done:
                # SB3 auto-resets on done, so the final observation has to be
                # stashed in info or the value bootstrap uses the wrong state.
                info = {"coverage": w.coverage(),
                        "redundancy": w.redundancy()}
                terminal_obs = w.observations()
                obs = w.reset(self._draw_seed())
                infos = [{"coverage": info["coverage"],
                          "redundancy": info["redundancy"],
                          "terminal_observation": terminal_obs[i]}
                         for i in range(N_UAVS)]
            else:
                obs = w.observations()
                infos = [{} for _ in range(N_UAVS)]
            obs_all.append(obs)
            rew_all.append(rewards)
            # A world ends for all its agents at once, so the done flag is
            # broadcast across the N slots that world owns.
            done_all.append(np.full(N_UAVS, done, dtype=bool))
            info_all.extend(infos)
        return (np.concatenate(obs_all, axis=0),
                np.concatenate(rew_all, axis=0),
                np.concatenate(done_all, axis=0),
                info_all)

    def close(self):
        pass

    # The rest of the VecEnv interface. SB3 requires these to exist; nothing
    # in this project calls them, so the ones that would need real semantics
    # raise rather than returning something misleading.

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)
        return [seed] * self.num_envs


def rollout(policy, density, seed, deterministic=False, n_agents=N_UAVS,
            use_memory=True):
    """One evaluation episode. Returns every measure the analysis needs.

    deterministic=False is the default on purpose. Six agents share one
    network, so a greedy argmax makes every agent in an identical local
    situation pick the identical action, which collapses the swarm into one
    repeated move and costs nearly ninety points of coverage. Sampling is
    what breaks the symmetry between otherwise indistinguishable agents, and
    is a property of the deployed controller rather than exploration noise
    left switched on.

    Pass policy=None for a random-action baseline.
    """
    world = CoverageWorld(density, seed, n_agents=n_agents,
                          use_memory=use_memory)
    # Seeding from the episode seed means a given seed reproduces the
    # identical episode, so the reported means are exactly reproducible.
    rng = np.random.default_rng(90000 + int(seed))
    if policy is not None:
        policy.set_random_seed(int(90000 + seed))
    obs = world.observations()
    done = False
    while not done:
        if policy is None:
            actions = rng.integers(len(ACTIONS), size=world.n_agents)
        else:
            actions, _ = policy.predict(obs, deterministic=deterministic)
        _, done = world.step(np.atleast_1d(actions))
        obs = world.observations()
    return {
        "coverage": world.coverage(),
        "redundancy": world.redundancy(),
        "cells_per_uav": world.cells_per_agent(),
        "total_path_length": float(world.path_len.sum()),
        "collision_events": int(world.near_misses),
        "timesteps": world.timestep,
        "seed": seed,
        "density": density,
    }


if __name__ == "__main__":
    # Self-test. The first check is the important one: if this environment
    # and the classical controllers ever disagreed about what a seed means,
    # every paired statistic in the project would be invalid.
    print("Self-test: layout must match the Boids and APF generator")
    sys.path.insert(0, os.path.join(_CODE, "phase3_apf"))
    from phase3_apf.model_apf import APFModel
    ok = True
    for d in [0.0, 0.1, 0.2, 0.3]:
        for s in range(5):
            a = APFModel(n_agents=6, obstacle_density=d,
                         obstacle_seed=s).obstacle_grid.astype(bool)
            b = build_obstacles(d, s)
            ok = ok and np.array_equal(a, b)
    print(f"  layouts identical to APF/Boids: {ok}")

    # A random policy should land near 50%. Anything much higher would mean
    # the task is too easy to distinguish controllers.
    r = rollout(None, 0.2, 0)
    print(f"  random-policy rollout: coverage {r['coverage']:.2f}%, "
          f"redundancy {r['redundancy']:.3f}, "
          f"cells/agent {r['cells_per_uav']}")

    env = MultiAgentCoverageVecEnv(0.2, n_worlds=2)
    o = env.reset()
    print(f"  vec env: num_envs {env.num_envs}, obs {o.shape}")
    o, rw, dn, inf = env.step(np.random.randint(8, size=env.num_envs))
    print(f"  after one step: obs {o.shape}, rewards {np.round(rw, 3)}")
