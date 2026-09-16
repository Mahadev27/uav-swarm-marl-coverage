"""
The memory ablation: the same policy trained with its nine onboard-memory
features held at zero.

The learned controller carries nine features derived from what each agent has
already sensed — which direction holds unexplored ground, how far the nearest
unswept cell is, and so on. Boids and APF do not have them. Rather than list
that as a caveat and move on, this trains an otherwise identical policy
without them and measures the difference.

What it leaves the agent is exactly the information the classical controllers
get: a 7x7 sensed window and its own position.

Everything else is held fixed. The network, the hyperparameters, the seed
pool, the world count and the reward scheme all come straight from
train_marl.py, so the only variable is `use_memory=False`. The observation
vector stays 60-dimensional; the nine slots are simply zeroed, which keeps
the network architecture identical rather than changing its input width.

The result: at a matched one-million-step budget the features are worth
+4.00 percentage points, and without them the policy reaches 89.6% against
APF's 94.2%. Trained on to 1.85M the memory-free arm gains nothing further
(+0.43 pp, p=0.83) while the memory arm keeps climbing, so the gap widens
rather than closing. The features raise the ceiling, not the speed of
reaching it.

Saves to marl_nomem_v2_d{pct}.zip. The `_v2` suffix is not cosmetic: an
earlier no-memory checkpoint produced a figure that could not be reproduced
and was retracted. This is the arm that survived.

Usage:
    python3 train_ablation.py [density_pct] [steps]    default 20, 250000
"""

import os
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
for _sub in ["", "phase2_boids", "phase5_marl"]:
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phase5_marl.marl_env import MultiAgentCoverageVecEnv, rollout
# Imported rather than restated, so the ablation cannot drift away from the
# configuration it is being compared against.
from phase5_marl.train_marl import PPO_KWARGS, TRAIN_SEEDS, N_WORLDS

RESULTS = os.path.join(_HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

torch.set_num_threads(4)


def train(density, steps):
    """Train the memory-free arm. Resumes and overwrites, as train_marl does."""
    pct = int(round(density * 100))
    path = os.path.join(RESULTS, f"marl_nomem_v2_d{pct}.zip")

    # use_memory=False is the entire experiment.
    env = MultiAgentCoverageVecEnv(density, n_worlds=N_WORLDS,
                                   seed_pool=TRAIN_SEEDS,
                                   use_memory=False)

    if os.path.exists(path):
        model = PPO.load(path, env=env, device="cpu")
        print(f"  resuming no-memory policy, density {pct}%", flush=True)
    else:
        model = PPO("MlpPolicy", env, device="cpu", **PPO_KWARGS)
        print(f"  new no-memory policy, density {pct}%", flush=True)

    t0 = time.time()
    model.learn(total_timesteps=steps, reset_num_timesteps=True,
                progress_bar=False)
    model.save(path)

    # Evaluate with memory off too. Evaluating a memory-free policy in a
    # memory-on world would feed it nine features it never trained against.
    cov = [rollout(model, density, s, use_memory=False)["coverage"]
           for s in range(10)]
    print(f"  density {pct}%: +{steps:,} steps ({time.time()-t0:.0f}s) "
          f"| no-memory policy {np.mean(cov):.2f}%", flush=True)
    return model


if __name__ == "__main__":
    pct = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 250_000
    train(pct / 100.0, steps)
