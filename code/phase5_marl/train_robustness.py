"""
The three experiments that turned out to matter most, all at 20% obstacle
density on the same fifty evaluation layouts.

1. CREDIT ASSIGNMENT. The headline policy rewards each agent for the cells it
   is first to reach. The alternative the multi-agent literature warns about
   is the naive shared reward, where every agent receives the global new-cell
   count. TeamRewardWorld below implements exactly that and nothing else: the
   network, budget, seed, observations and dynamics are all held fixed, so
   the reward scheme is the only variable.

   Result: 63.0% against 93.6%, a gap of 30.6 percentage points, and not one
   of fifty runs reaches the target. That is larger than the distance between
   any two controllers in the whole study.

2. TRAINING-SEED VARIANCE. Every MARL figure in the report comes from one
   training run at PPO seed 0. Fifty evaluation seeds control for layout
   variance; they say nothing about variance across training runs. Seeds 1,
   2 and 3 are trained identically here to find out.

   Result: 7.8 points between best and worst seed at 1.00M steps, and seed 0
   is the best of the four. The headline numbers sit at the top of their
   distribution rather than the middle of it.

3. BUDGET. Running every arm on to 1.85M separates two explanations the
   single-budget comparison cannot. If the memory-free policy were merely
   slower, extra training would close the gap. It does not: the memory-free
   arm gains nothing (+0.43 pp, p=0.83) while the memory arm keeps climbing.
   Meanwhile the across-seed spread halves and every seed overtakes APF, so
   at this density the training budget, not the method, set the result.

HOW IT RUNS. Training nine policies takes hours, so this is built to run in
short chunks and survive being killed. `stage_chunk` trains whichever job is
furthest behind for a fixed number of seconds, records progress in a JSON
ledger and exits. `stage_eval` does the same for evaluation, checkpointing
per seed. Call either repeatedly until it reports complete.

Usage:
    python3 train_robustness.py status          # what is done, what is not
    python3 train_robustness.py chunk [secs]    # train for a while, then stop
    python3 train_robustness.py eval  [secs]    # evaluate for a while
    python3 train_robustness.py stats           # write the tables
"""

import json
import os
import shutil
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
for _sub in ["", "phase2_boids", "phase5_marl"]:
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from phase5_marl.marl_env import (CoverageWorld,
                                  MultiAgentCoverageVecEnv,
                                  rollout, W_COV, W_TIME, W_TEAM)
from phase5_marl.train_marl import (PPO_KWARGS, TRAIN_SEEDS,
                                    N_WORLDS)

torch.set_num_threads(4)

RESULTS = os.path.join(_HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

DENSITY = 0.20
EVAL_SEEDS = list(range(50))
BASE_STEPS = 1_000_000
EXTENDED_STEPS = 850_000

LOG = os.path.join(RESULTS, "robustness_progress.log")


def say(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


class TeamRewardWorld(CoverageWorld):
    """CoverageWorld with the reward replaced by the shared team total.

    Rather than reimplement the dynamics, this calls the parent step and then
    reverses its reward arithmetic to recover how many new cells each agent
    personally found, sums them, and pays every agent that same total. The
    world, the movement and the coverage accounting are untouched, so the
    only thing that differs from the headline configuration is who gets
    credited for what.
    """

    def step(self, actions):
        rewards, done = super().step(actions)
        # Undo the parent's reward to recover own_new: subtract the terminal
        # bonus if this was the last step, add back the time penalty, divide
        # by the coverage weight.
        base = rewards + W_TIME
        bonus = W_TEAM * (self.coverage() / 100.0) if done else 0.0
        own_new = (base - bonus) / W_COV
        total_new = float(np.sum(own_new))
        # Everyone gets the swarm's total, so an agent that did nothing is
        # paid the same as one that swept a corridor. This is the credit-
        # assignment problem, implemented deliberately.
        team = np.full(self.n_agents, W_COV * total_new - W_TIME,
                       dtype=float)
        if done:
            team = team + bonus
        return team, done


class TeamRewardVecEnv(MultiAgentCoverageVecEnv):
    def __init__(self, density, n_worlds=N_WORLDS, seed_pool=None):
        super().__init__(density, n_worlds=n_worlds, seed_pool=seed_pool)
        self.worlds = [TeamRewardWorld(density, self._draw_seed())
                       for _ in range(n_worlds)]


class Budget(BaseCallback):
    """Stops learn() after a wall-clock deadline.

    Returning False from _on_step is SB3's signal to halt. This is what lets
    training proceed in short chunks on a laptop without losing progress.
    """

    def __init__(self, seconds):
        super().__init__(0)
        self.deadline = time.time() + seconds

    def _on_step(self):
        return time.time() < self.deadline


# The nine training jobs. Fields:
#   key, output file, target steps, env kind, PPO seed, resume-from
#
# The _185 jobs and mem185/nomem185 start from a finished 1.00M checkpoint
# and add 850k more, rather than retraining from scratch. `resume` names the
# file to copy from; the original is never modified.
JOBS = [
    ("mem185", "marl_mem185_d20.zip", 851_000, "std", 0, "marl_d20.zip"),
    ("s1",     "marl_s1_d20.zip",   BASE_STEPS, "std", 1, None),
    ("s2",     "marl_s2_d20.zip",   BASE_STEPS, "std", 2, None),
    ("s3",     "marl_s3_d20.zip",   BASE_STEPS, "std", 3, None),
    ("team",   "marl_team_d20.zip", BASE_STEPS, "team", 0, None),
    ("nomem185", "marl_nomem185_d20.zip", 851_000, "nomem", 0,
     "marl_nomem_v2_d20.zip"),
    ("s1_185", "marl_s1_185_d20.zip", 851_000, "std", 1, "marl_s1_d20.zip"),
    ("s2_185", "marl_s2_185_d20.zip", 851_000, "std", 2, "marl_s2_d20.zip"),
    ("s3_185", "marl_s3_185_d20.zip", 851_000, "std", 3, "marl_s3_d20.zip"),
]

LEDGER = os.path.join(RESULTS, "robustness_ledger.json")


def _ledger():
    if os.path.exists(LEDGER):
        with open(LEDGER) as fh:
            return json.load(fh)
    return {}


def _write_ledger(d):
    with open(LEDGER, "w") as fh:
        json.dump(d, fh, indent=1)


def _make_env(kind):
    """std = the reported configuration, team = shared reward,
    nomem = memory features zeroed."""
    if kind == "team":
        return TeamRewardVecEnv(DENSITY, n_worlds=N_WORLDS,
                                seed_pool=TRAIN_SEEDS)
    return MultiAgentCoverageVecEnv(DENSITY, n_worlds=N_WORLDS,
                                    seed_pool=TRAIN_SEEDS,
                                    use_memory=(kind != "nomem"))


def stage_chunk(seconds=35):
    """Train the first unfinished job for `seconds`, then return.

    Returns True only when every job has reached its target. Deliberately
    handles one job per call so a crash costs at most one chunk.
    """
    led = _ledger()
    for key, fname, want, kind, seed, resume in JOBS:
        done = led.get(key, 0)
        if done >= want:
            continue
        path = os.path.join(RESULTS, fname)
        env = _make_env(kind)
        if not os.path.exists(path):
            if resume:
                # Copy rather than train in place: the 1.00M checkpoints
                # are the ones the reported results come from and must not
                # be advanced past the budget they represent.
                shutil.copy2(os.path.join(RESULTS, resume), path)
                say(f"{key}: copied {resume} -> {fname} (original untouched)")
                model = PPO.load(path, env=env, device="cpu")
            else:
                kw = dict(PPO_KWARGS)
                kw["seed"] = int(seed)
                model = PPO("MlpPolicy", env, device="cpu", **kw)
                say(f"{key}: new policy, PPO seed {seed}, env {kind}")
        else:
            model = PPO.load(path, env=env, device="cpu")
        t0 = time.time()
        model.learn(total_timesteps=min(want - done, 10_000_000),
                    callback=Budget(seconds), reset_num_timesteps=True,
                    progress_bar=False)
        model.save(path)
        # num_timesteps counts this chunk only, because reset_num_timesteps
        # is True; the ledger accumulates the real total across chunks.
        led[key] = done + int(model.num_timesteps)
        _write_ledger(led)
        say(f"{key}: {led[key]:,}/{want:,} steps "
            f"(+{model.num_timesteps:,} in {time.time()-t0:.0f}s)")
        return False
    say("all training jobs complete")
    return True


def stage_status():
    led = _ledger()
    for key, fname, want, kind, seed, _ in JOBS:
        d = led.get(key, 0)
        print(f"  {key:8s} {d:>9,}/{want:,}  "
              f"{'DONE' if d >= want else 'pending'}")
    if os.path.exists(EVAL_OUT):
        store = dict(np.load(EVAL_OUT))
        print("  evaluated:", ", ".join(sorted(store)))


# What gets evaluated, and with which memory setting. A memory-free policy
# must be evaluated memory-free, or it would be handed nine features it never
# trained against.
CONFIGS = [
    ("mem_1.00M_s0",  "marl_d20.zip",             True,
     "Memory, 1.00M, seed 0 (reported)"),
    ("mem_1.85M_s0",  "marl_mem185_d20.zip",      True,
     "Memory, 1.85M, seed 0"),
    ("nomem_1.00M",   "marl_nomem_v2_d20.zip",    False,
     "No memory, 1.00M"),
    ("nomem_1.85M",   "marl_nomem185_d20.zip",    False,
     "No memory, 1.85M"),
    ("mem_1.00M_s1",  "marl_s1_d20.zip",          True,
     "Memory, 1.00M, seed 1"),
    ("mem_1.00M_s2",  "marl_s2_d20.zip",          True,
     "Memory, 1.00M, seed 2"),
    ("mem_1.00M_s3",  "marl_s3_d20.zip",          True,
     "Memory, 1.00M, seed 3"),
    ("team_1.00M_s0", "marl_team_d20.zip",        True,
     "Shared team reward, 1.00M, seed 0"),
    ("mem_1.85M_s1",  "marl_s1_185_d20.zip",      True,
     "Memory, 1.85M, seed 1"),
    ("mem_1.85M_s2",  "marl_s2_185_d20.zip",      True,
     "Memory, 1.85M, seed 2"),
    ("mem_1.85M_s3",  "marl_s3_185_d20.zip",      True,
     "Memory, 1.85M, seed 3"),
]

EVAL_OUT = os.path.join(RESULTS, "robustness_eval.npz")


PARTIAL = os.path.join(RESULTS, "robustness_eval_partial.json")


def stage_eval(seconds=35):
    """Evaluate finished policies on all fifty seeds, checkpointing per seed.

    Partial progress goes to a JSON file so an interrupted run resumes at the
    seed it stopped on. A config is only promoted into the .npz once all
    fifty seeds are in, so nothing downstream can read a half-finished arm.
    """
    torch.set_num_threads(1)
    store = dict(np.load(EVAL_OUT)) if os.path.exists(EVAL_OUT) else {}
    part = {}
    if os.path.exists(PARTIAL):
        with open(PARTIAL) as fh:
            part = json.load(fh)

    deadline = time.time() + seconds
    for key, fname, use_mem, label in CONFIGS:
        if key in store:
            continue
        path = os.path.join(RESULTS, fname)
        if not os.path.exists(path):
            say(f"eval: {fname} not trained yet, stopping here")
            return False
        model = PPO.load(path, device="cpu")
        done = part.get(key, [])
        for s in EVAL_SEEDS[len(done):]:
            done.append(float(rollout(model, DENSITY, s,
                                      deterministic=False,
                                      use_memory=use_mem)["coverage"]))
            if time.time() > deadline:
                break
        part[key] = done
        with open(PARTIAL, "w") as fh:
            json.dump(part, fh)
        if len(done) == len(EVAL_SEEDS):
            cov = np.array(done)
            store[key] = cov
            np.savez(EVAL_OUT, **store)
            say(f"eval: {key} DONE mean {cov.mean():.2f}% "
                f"sd {cov.std(ddof=1):.2f} n>=90 {(cov >= 90).sum()}/50")
        else:
            say(f"eval: {key} {len(done)}/{len(EVAL_SEEDS)} seeds")
        if time.time() > deadline:
            return False
    say("all evaluations complete")
    return True


def stage_stats():
    """Descriptives, paired tests and the seed-variance blocks.

    APF at the same density on the same fifty layouts is pulled in as the
    reference point, with an assert that the seeds really do match. Without
    that check the paired tests would silently compare different layouts.
    """
    from scipy import stats as st
    import pandas as pd

    store = dict(np.load(EVAL_OUT))
    apf = np.load(os.path.join(_CODE, "phase3_apf/results/apf_results_A.npy"),
                  allow_pickle=True).item()
    runs = sorted(apf[DENSITY], key=lambda r: r["seed"])
    assert [r["seed"] for r in runs] == EVAL_SEEDS, "APF seeds not matched"
    apf20 = np.array([r["coverage"] for r in runs])
    store["APF"] = apf20

    # Fixed bootstrap seed, so the confidence intervals are reproducible.
    rng = np.random.default_rng(20260729)

    def boot_ci(x, n=10000):
        idx = rng.integers(0, len(x), size=(n, len(x)))
        m = x[idx].mean(axis=1)
        return np.percentile(m, 2.5), np.percentile(m, 97.5)

    rows = []
    for key, _, _, label in CONFIGS:
        if key not in store:
            continue
        x = store[key]
        lo, hi = boot_ci(x)
        rows.append(dict(config=key, label=label, n=len(x),
                         mean=round(float(x.mean()), 2),
                         sd=round(float(x.std(ddof=1)), 2),
                         ci95_low=round(float(lo), 2),
                         ci95_high=round(float(hi), 2),
                         n_at_or_above_90=int((x >= 90).sum())))
    lo, hi = boot_ci(apf20)
    rows.append(dict(config="APF", label="APF, tuned (no training)",
                     n=len(apf20), mean=round(float(apf20.mean()), 2),
                     sd=round(float(apf20.std(ddof=1)), 2),
                     ci95_low=round(float(lo), 2),
                     ci95_high=round(float(hi), 2),
                     n_at_or_above_90=int((apf20 >= 90).sum())))
    desc = pd.DataFrame(rows)

    def paired(a, b):
        """Wilcoxon signed-rank plus rank-biserial and a bootstrap CI.

        Paired rather than independent because both arms ran on the same
        fifty layouts, which is a much tighter comparison: it removes
        layout difficulty as a source of variance entirely.
        """
        x, y = store[a], store[b]
        d = x - y
        W, p = st.wilcoxon(x, y, alternative="two-sided")
        n = len(d)
        pos = st.rankdata(np.abs(d))[d > 0].sum()
        neg = st.rankdata(np.abs(d))[d < 0].sum()
        rrb = (pos - neg) / (pos + neg) if (pos + neg) else 0.0
        idx = rng.integers(0, n, size=(10000, n))
        bm = d[idx].mean(axis=1)
        return dict(comparison=f"{a} vs {b}",
                    mean_a=round(float(x.mean()), 2),
                    mean_b=round(float(y.mean()), 2),
                    diff_pp=round(float(d.mean()), 2),
                    ci95_low=round(float(np.percentile(bm, 2.5)), 2),
                    ci95_high=round(float(np.percentile(bm, 97.5)), 2),
                    wilcoxon_W=float(W), wilcoxon_p=float(p),
                    rank_biserial=round(float(rrb), 3))

    comps = []
    for a, b in [("mem_1.00M_s0", "nomem_1.00M"),
                 ("mem_1.85M_s0", "nomem_1.85M"),
                 ("mem_1.85M_s0", "mem_1.00M_s0"),
                 ("nomem_1.85M", "nomem_1.00M"),
                 ("mem_1.00M_s0", "team_1.00M_s0"),
                 ("mem_1.00M_s0", "APF"),
                 ("mem_1.85M_s0", "APF"),
                 ("nomem_1.85M", "APF"),
                 ("team_1.00M_s0", "APF")]:
        if a in store and b in store:
            comps.append(paired(a, b))
    pair = pd.DataFrame(comps)

    def seed_block(keys, title):
        """Summarise across training seeds: the spread is the finding.

        Kruskal-Wallis is used rather than ANOVA because the coverage
        distributions are not normal (Shapiro-Wilk rejects for most cells).
        """
        keys = [k for k in keys if k in store]
        if len(keys) < 2:
            return []
        means = np.array([store[k].mean() for k in keys])
        pooled = np.concatenate([store[k] for k in keys])
        out = [f"   {title}",
               f"     per-seed means : {np.round(means, 2).tolist()}",
               f"     mean of means  : {means.mean():.2f}%",
               f"     SD across seeds: {means.std(ddof=1):.2f} pp",
               f"     range          : {means.max()-means.min():.2f} pp "
               f"({means.min():.2f} to {means.max():.2f})",
               f"     pooled {len(pooled)} runs: {pooled.mean():.2f} "
               f"+/- {pooled.std(ddof=1):.2f}, "
               f"{int((pooled >= 90).sum())}/{len(pooled)} at or above 90%"]
        if len(keys) > 2:
            H, p = st.kruskal(*[store[k] for k in keys])
            out.append(f"     Kruskal-Wallis : H = {H:.2f}, p = {p:.2e}")
        return out + [""]

    seed_keys = [k for k in ("mem_1.00M_s0", "mem_1.00M_s1",
                             "mem_1.00M_s2", "mem_1.00M_s3") if k in store]
    seed_means = np.array([store[k].mean() for k in seed_keys])
    pooled = np.concatenate([store[k] for k in seed_keys])

    txt = [
        "MARL ROBUSTNESS EXPERIMENTS - 20% obstacle density, Seed Set A",
        "=" * 68, "",
        "1. DESCRIPTIVE, 50 matched evaluation seeds each", "",
        desc.to_string(index=False), "",
        "2. PAIRED COMPARISONS (Wilcoxon signed-rank, matched layouts)", "",
        pair.to_string(index=False), "",
        "3. TRAINING-SEED VARIANCE", "",
    ]
    txt += seed_block(["mem_1.00M_s0", "mem_1.00M_s1",
                       "mem_1.00M_s2", "mem_1.00M_s3"],
                      "At 1.00M agent-steps (the budget used in the report)")
    txt += seed_block(["mem_1.85M_s0", "mem_1.85M_s1",
                       "mem_1.85M_s2", "mem_1.85M_s3"],
                      "At 1.85M agent-steps")

    out_txt = os.path.join(RESULTS, "robustness_stats.txt")
    with open(out_txt, "w") as fh:
        fh.write("\n".join(txt))
    desc.to_csv(os.path.join(RESULTS, "robustness_descriptive.csv"),
                index=False)
    pair.to_csv(os.path.join(RESULTS, "robustness_pairwise.csv"),
                index=False)
    print("\n".join(txt))
    say("stats written")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "status"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 35.0
    if stage == "chunk":
        stage_chunk(secs)
    elif stage == "eval":
        stage_eval(secs)
    elif stage == "stats":
        stage_stats()
    else:
        stage_status()
