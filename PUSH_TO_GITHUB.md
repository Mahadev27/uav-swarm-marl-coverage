# Push to GitHub

I could connect to your account (**Mahadev27**, 7 public repos) but the token is
read-only — it can't create repositories or write files. So this runs from your
Mac. It's the better route anyway: your 43 MB of model checkpoints would be slow
and unreliable through the API, and instant over git.

Everything in this folder is ready. Three steps, about two minutes.

---

## 1 — Create the empty repository on github.com

Go to **https://github.com/new** and set:

| Field | Value |
|---|---|
| Repository name | `uav-swarm-marl-coverage` |
| Description | Six UAVs cover an unknown area with no central controller and no communication. Multi-agent PPO vs classical control across 1,600 matched runs. MSc Robotics dissertation. |
| Visibility | **Public** |
| Initialize with README | **leave unticked** — you already have one |
| .gitignore / licence | **None** — you already have a .gitignore |

Click **Create repository**. Don't run the setup commands GitHub shows you; use
the ones below instead.

---

## 2 — Push from Terminal

```bash
cd ~/Documents/"Dissertation docs"

# Start a clean history. The existing .git points at the old 87-character repo
# and is two commits old, missing nearly all of this work.
rm -rf .git
git init -b main

git add -A

# Check before committing: should be roughly 130 files, no venv, no _archive,
# no .mov, no "research papers"
git status --short | wc -l
git status --short | grep -cE "venv|_archive|\.mov|research papers|Supplementary"   # expect 0

git commit -m "Decentralised UAV swarm coverage with multi-agent reinforcement learning

Four controllers compared across 1,600 matched runs: Boids, Artificial
Potential Fields, a single-agent PPO hybrid, and a parameter-shared
multi-agent PPO policy executing locally with no message passing.

- MARL environment, training loop and evaluation (Stable-Baselines3 PPO)
- credit-assignment, memory-ablation and training-seed experiments
- paired Wilcoxon with Holm-Bonferroni, effect sizes, bootstrap CIs
- figure and animation generation from stored results
- 14 trained checkpoints, all result files, LaTeX source and report"

git remote add origin https://github.com/Mahadev27/uav-swarm-marl-coverage.git
git push -u origin main
```

If it asks for a password, use a **personal access token**, not your GitHub
password — github.com → Settings → Developer settings → Personal access tokens.

---

## 3 — Finish the repo page

Once it's up, on the repo page:

- Click the **gear** beside *About*, add topics:
  `reinforcement-learning` `multi-agent` `uav` `swarm-robotics` `ppo` `robotics` `python`
- Check the animation plays at the top of the README. GitHub renders GIFs inline;
  if it doesn't appear, the path is `docs/demo.gif`.
- **Pin it** on your profile: github.com/Mahadev27 → Customize your pins.

---

## What gets pushed

**In** — roughly 130 files, ~67 MB:

- `code/` — 32 Python files, 14 trained checkpoints, 62 result files, both
  requirements files
- `latex_final/` — LaTeX source, figures, compiled dissertation
- `docs/` — the animation and coverage plot the README displays
- `README.md`, `.gitignore`

**Out**, via `.gitignore`:

- the three virtual environments
- `_archive/`, `code/compare/` (the stale 500-step generation)
- `research papers/` and `new papers/` — 86 MB of other people's PDFs
- `figures/` — 23 MB duplicating `latex_final/figures/`
- video recordings, submission zips, LaTeX build artefacts, `.DS_Store`

Largest single file is 6.4 MB, inside GitHub's 100 MB per-file limit.

---

## Before you link this anywhere

The Python files have **zero comments and zero docstrings**. You had me strip them
for the academic submission, where it was harmless. On a repository you are
showing employers it is not.

An engineer who opens `marl_env.py` and finds 300 lines with no explanation draws
one of two conclusions — careless, or not your code. Neither helps you, and it's
the first thing a technical reviewer checks after the README.

The backup holding the originals is gone, so restoring means rewriting. Ask me
and I'll document the files that matter: the environment, the training loop, the
statistics and the figure generation. It is the highest-value hour you could
spend on this repo, and worth doing before recruiters see it.
