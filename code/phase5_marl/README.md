# Phase 5 — Fully decentralised multi-agent PPO

All six UAVs are driven by one shared policy (parameter sharing). No
agent is hand-coded and there is no inter-agent communication. This is
the system reported as "MARL" in the dissertation, and it replaces the
Phase 4 hybrid in which only one agent learned.

| File | What it is |
|---|---|
| `marl_env.py` | The world, the VecEnv wrapper, and `rollout()` for evaluation |
| `train_marl.py` | Trains one shared policy per obstacle density |
| `test_marl.py` | Evaluates on Seed Set A (0–49) and Seed Set B (50–99) |
| `results/marl_d*.zip` | Trained policies, one per density (already included) |
| `results/marl_results_A.npy`, `_B.npy` | Evaluation output read by the statistics script |

---

## Setup (once)

From the `code/` directory:

```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
pip install -r requirements.txt
```

On an Apple Silicon Mac the default `pip install torch` gives a CPU
build, which is what these scripts use. Everything runs on CPU; no GPU
is needed.

---

## Important: run everything from `code/`

The scripts resolve imports relative to the repository root, so run
them as shown below and not from inside `phase5_marl/`.

```bash
cd code
```

---

## 1. Check the environment is wired up correctly

```bash
python3 phase5_marl/marl_env.py
```

This self-test confirms that the obstacle layouts generated here are
byte-identical to those used by the Boids and APF models, which is
what makes the paired statistical tests valid. Expect:

```
layouts identical to APF/Boids: True
random-policy rollout: coverage ~50%, ...
vec env: num_envs 12, obs (12, 60)
```

If the first line says `False`, the comparison is no longer matched and
nothing downstream should be trusted.

---

## 2. Training (optional — trained policies are already included)

```bash
# one density at a time: <density_pct> <steps>
python3 phase5_marl/train_marl.py 0  250000
python3 phase5_marl/train_marl.py 10 250000
python3 phase5_marl/train_marl.py 20 250000
python3 phase5_marl/train_marl.py 30 250000
```

Each call **resumes from the saved checkpoint** if one exists, so a long
run can be built up in short passes. Repeat the four commands until you
reach the budget you want; the reported results used roughly one
million agent-steps per density.

Roughly 8,000 agent-steps per second on an M1, so 250k steps takes
about 30 seconds and a full 1M-step policy about two minutes per
density.

To train from scratch rather than resume, delete the checkpoint first:

```bash
rm phase5_marl/results/marl_d20.zip phase5_marl/results/marl_log_d20.npy
```

Progress prints every 100k steps, and each pass ends with a sanity
check against a random policy:

```
density 20%: +250,000 steps (29s), total 1,000,112 | policy 92.68%  random 47.47%  gain +45.20 pp
```

---

## 3. Evaluation

Everything at once:

```bash
python3 phase5_marl/test_marl.py
```

Or one condition at a time, which is useful if you want to watch it or
are on a slow machine:

```bash
python3 phase5_marl/test_marl.py 20 A     # density 20%, Seed Set A
python3 phase5_marl/test_marl.py 30 B
```

Each condition runs 50 episodes and prints:

```
A 20% | mean  92.84%  sd  3.72  >=90%: 42/50
```

Part files are written as it goes, and once all eight conditions exist
they are merged automatically into `marl_results_A.npy` and
`marl_results_B.npy`. Allow about 35 seconds per condition, so roughly
five minutes for all eight.

---

## 4. Fold the results into the statistics

```bash
python3 analysis/statistical_tests.py
```

This reads all four controllers, verifies at run time that the layouts
match across them, then writes `analysis/results/statistical_tests.txt`
and four CSV files containing every number quoted in the report.

---

## Two things worth knowing

**Evaluation samples actions rather than taking the argmax.** A
deterministic argmax over a shared policy makes every agent in the same
local situation choose the same move, which collapses the swarm and
costs around ninety points of coverage. `rollout()` therefore defaults
to `deterministic=False` during evaluation. This is a property of
parameter sharing, not a bug.

**Rollout length is the hyperparameter that matters.** With 48
agent-slots, `n_steps=256` yields only about forty gradient updates per
500k steps and coverage stalls at the random-walk level.
`n_steps=64` gives roughly 160 updates over the same budget and is what
makes the task learnable. If you change `N_WORLDS`, adjust `n_steps` to
keep the update count high.

---

## If something breaks

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: phase5_marl` | You are not in `code/`. `cd code` first. |
| `ModuleNotFoundError: mesa` | The self-test imports the APF model for the layout check. `pip install mesa==3.0.3`. |
| `layouts identical to APF/Boids: False` | The obstacle generator or its seeding has changed. Fix before using any result. |
| Observation-shape error on load | A saved policy predates a change to the observation vector. Delete the `.zip` and retrain. |
| Training coverage flat near 50% | Rollout too long for the slot count. See the note above. |
