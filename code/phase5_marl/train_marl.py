"""
Trains the parameter-shared multi-agent PPO policy, one policy per obstacle
density.

Nothing exotic happens here: it is Stable-Baselines3 PPO pointed at
MultiAgentCoverageVecEnv, which flattens six agents across eight worlds into
48 environment slots. The sharing is a property of that wrapper, not of
anything in this file. One network comes out and every agent runs its own
copy of it.

Two details are worth reading before you change anything:

  * n_steps is short on purpose, and the reason is in the comment on
    PPO_KWARGS. It was the difference between a policy that learns and one
    that stalls at the random-walk level.

  * train() RESUMES from an existing checkpoint and overwrites it. Running
    this on a finished project will quietly replace the models the reported
    results came from. See the warning on the function.

Usage:
    python3 train_marl.py [density_pct] [steps]      default 20, 500000
"""

import os
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
for _sub in ["", "phase2_boids", "phase5_marl"]:
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phase5_marl.marl_env import MultiAgentCoverageVecEnv, rollout

RESULTS = os.path.join(_HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# Seeds 0-49 only. Set B (50-99) is never seen during training, which is what
# makes the held-out generalisation test meaningful.
TRAIN_SEEDS = list(range(50))
N_WORLDS = 8                           # 8 worlds x 6 agents = 48 agent-slots

PPO_KWARGS = dict(
    # n_steps is the decisive hyperparameter here, and it is short for a
    # reason that is easy to get wrong. With 48 agent-slots, n_steps=256
    # gives 12,288 samples per update but only about forty gradient updates
    # per 500k steps. PPO does not converge on forty updates: coverage
    # stalls near the random-walk level of 50% and stays there however long
    # you train. Dropping to 64 gives roughly 160 updates over the same
    # sample budget, and that is what makes the task learnable.
    #
    # The general lesson: parameter sharing multiplies your slot count, and
    # the effective number of updates falls as it rises. Compensate.
    n_steps=64,                        # x48 slots = 3,072 samples per update
    batch_size=512,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,                    # mild; sampling already spreads agents
    learning_rate=5e-4,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=0,
    seed=0,                            # the seed the headline results use
)

torch.set_num_threads(4)


class CoverageLogger(BaseCallback):
    """Prints mean training coverage every 100k steps.

    Reads the coverage figure the vec env puts into info on episode end, so
    this reports what the swarm actually achieved rather than mean reward,
    which is hard to interpret across reward schemes.

    offset exists so a resumed run continues the step count from where the
    previous one stopped, instead of restarting at zero.
    """

    def __init__(self, tag, offset=0, verbose=0):
        super().__init__(verbose)
        self.tag = tag
        self.offset = offset
        self.history = []
        self._recent = []
        self._next = 100_000

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "coverage" in info:
                self._recent.append(info["coverage"])
        if self._recent and self.num_timesteps >= self._next:
            # Average the last 400 finished episodes rather than everything
            # since the start, so the number tracks current performance.
            m = float(np.mean(self._recent[-400:]))
            self.history.append({"step": int(self.num_timesteps + self.offset),
                                 "mean_cov": m})
            print(f"    [{self.tag}] {self.num_timesteps + self.offset:>9,}"
                  f"  train coverage {m:.2f}%", flush=True)
            self._next += 100_000
            self._recent = self._recent[-400:]
        return True


def train(density, steps):
    """Train (or continue training) the policy for one obstacle density.

    WARNING: this resumes. If a checkpoint already exists at the target path
    it is loaded, trained for `steps` MORE timesteps, and saved back over
    itself. On a finished project that destroys the model the published
    numbers came from. The resume behaviour exists because training was done
    in chunks on a laptop; it is not what you want when reproducing results.
    """
    pct = int(round(density * 100))
    path = os.path.join(RESULTS, f"marl_d{pct}.zip")
    log_path = os.path.join(RESULTS, f"marl_log_d{pct}.npy")

    env = MultiAgentCoverageVecEnv(density, n_worlds=N_WORLDS,
                                   seed_pool=TRAIN_SEEDS)

    history, offset = [], 0
    if os.path.exists(path):
        model = PPO.load(path, env=env, device="cpu")
        if os.path.exists(log_path):
            history = list(np.load(log_path, allow_pickle=True))
            offset = history[-1]["step"] if history else 0
        print(f"  resuming density {pct}% from {offset:,} steps", flush=True)
    else:
        model = PPO("MlpPolicy", env, device="cpu", **PPO_KWARGS)
        print(f"  new policy for density {pct}%", flush=True)

    cb = CoverageLogger(f"d{pct}", offset=offset)
    t0 = time.time()
    # reset_num_timesteps=True keeps each chunk's internal counter starting
    # at zero; the true cumulative total is tracked through `offset` instead.
    model.learn(total_timesteps=steps, callback=cb,
                reset_num_timesteps=True, progress_bar=False)
    dt = time.time() - t0

    model.save(path)
    np.save(log_path, np.array(history + cb.history, dtype=object))

    # Quick sanity read on eight seeds, against a random-action baseline on
    # the same eight. The gap matters more than the absolute number: if the
    # policy is not clearly above random, something is broken.
    tr = [rollout(model, density, s, deterministic=False)["coverage"]
          for s in range(8)]
    rd = [rollout(None, density, s)["coverage"] for s in range(8)]
    total = offset + steps
    print(f"  density {pct}%: +{steps:,} steps ({dt:.0f}s), total {total:,}"
          f" | policy {np.mean(tr):.2f}%  random {np.mean(rd):.2f}%"
          f"  gain {np.mean(tr)-np.mean(rd):+.2f} pp", flush=True)
    return model


if __name__ == "__main__":
    pct = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 500_000
    train(pct / 100.0, steps)
