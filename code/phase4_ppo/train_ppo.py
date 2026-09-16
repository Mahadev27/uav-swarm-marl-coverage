"""
Trains the single-agent PPO hybrid, one policy per obstacle density.

Only agent 0 learns; agents 1-5 are scripted inside the environment. See the
header of uav_coverage_env.py for why that architecture cannot answer the
research question, and what it cost.

The hyperparameters differ from the MARL trainer and the difference is
instructive. Here n_steps=1024 across 8 environments gives ~8k samples per
update, which is fine for a single agent. The MARL trainer has 48 agent-slots
and had to drop n_steps to 64 to get enough gradient updates out of the same
sample budget. Parameter sharing changes what a sensible rollout length is.

Entropy is higher here (0.03 against 0.005) because one agent exploring alone
needs more push than six agents whose sampled actions already spread them out.
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print("pip install stable-baselines3 gymnasium")
    sys.exit(1)

from phase4_ppo.uav_coverage_env import UAVCoverageEnv

DENSITIES       = [0.0, 0.1, 0.2, 0.3]
TOTAL_TIMESTEPS = 1_000_000
N_ENVS          = 4
RESULTS_DIR     = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


class EpisodeCallback(BaseCallback):
    def __init__(self, log_path, density, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.density  = density
        self.ep_covs  = []
        self.log_data = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done and "coverage" in info:
                self.ep_covs.append(info["coverage"])

        # Report the last 100 episodes rather than everything since the
        # start, so the number tracks current performance.
        if self.n_calls % 5000 == 0 and self.ep_covs:
            mean_cov = np.mean(self.ep_covs[-100:])
            self.log_data.append({"step": self.n_calls,
                                  "mean_cov": mean_cov})
            np.save(self.log_path, self.log_data)
            pct  = int(self.density*100)
            prog = self.n_calls/TOTAL_TIMESTEPS*100
            print(f"  [{pct}% obs] Step {self.n_calls:>8} ({prog:.0f}%) "
                  f"| Episodes: {len(self.ep_covs):>4} "
                  f"| Mean Coverage: {mean_cov:.1f}%")
        return True


def train_density(density, total_timesteps=TOTAL_TIMESTEPS, n_envs=N_ENVS):
    pct = int(density*100)
    print(f"\n{'='*60}")
    print(f"  PPO — {pct}% obstacle density")
    print(f"  Timesteps: {total_timesteps:,} | Parallel envs: {n_envs}")
    print(f"{'='*60}")

    # Stagger the starting seeds across parallel envs so they are not all
    # cycling through the same layouts in lockstep.
    env = DummyVecEnv([
        lambda d=density, s=i*13: UAVCoverageEnv(obstacle_density=d, seed=s)
        for i in range(n_envs)
    ])

    model = PPO(
        policy        = "MlpPolicy",
        env           = env,
        n_steps       = 1024,   # ~8k samples per update across 8 envs
        batch_size    = 128,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        learning_rate = 2e-4,
        ent_coef      = 0.03,   # higher than MARL: one agent explores alone
        clip_range    = 0.2,
        vf_coef       = 0.5,
        policy_kwargs = dict(net_arch=[256, 256]),
        verbose       = 0,
    )

    log_path = os.path.join(RESULTS_DIR, f"ppo_log_d{pct}.npy")
    model.learn(total_timesteps=total_timesteps,
                callback=EpisodeCallback(log_path, density))

    save_path = os.path.join(RESULTS_DIR, f"ppo_model_d{pct}")
    model.save(save_path)
    print(f"\n  Saved -> {save_path}.zip")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--density", type=int, default=None)
    parser.add_argument("--test",    action="store_true")
    args = parser.parse_args()

    if args.test:
        print("Quick test — 30k steps, 4 envs, 0% obstacles")
        train_density(0.0, total_timesteps=30_000, n_envs=4)
        print("\nTarget: coverage 85%+ by step 30k")
    elif args.density is not None:
        train_density(args.density/100.0)
    else:
        for d in DENSITIES:
            train_density(d)
        print("\nAll 4 models trained.")
