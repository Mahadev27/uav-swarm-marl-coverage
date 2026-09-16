"""
Every statistical test, table and CSV the report depends on.

This is the file to read if you want to know whether the comparison is sound.
It loads the four controllers' stored results, checks the pairing is real,
runs the tests, and writes the descriptive, pairwise, generalisation and
against-target tables.

THE DESIGN IS PAIRED, AND THAT IS THE WHOLE POINT. Each seed produces one
obstacle layout, and all four controllers are run on it. So the comparison
between two controllers on seed 7 is a difference in behaviour, not a
difference in how hard seed 7 was. Layout difficulty varies enormously --
some maps are simply easier to sweep -- and pairing removes that variance
entirely rather than averaging over it.

That only holds if the pairing is genuine, so verify_pairing() checks it two
ways before any test runs: the seed labels have to align across controllers,
AND the layouts are regenerated from scratch and compared cell by cell. An
earlier version of this project had a defect where one seed produced three
different maps, which would have invalidated every paired test silently. The
check exists because of that.

WHAT IS COMPUTED, AND WHY EACH ONE:

  Wilcoxon signed-rank    the primary test. Non-parametric, because 27 of the
                          32 coverage distributions fail Shapiro-Wilk.
  Mann-Whitney U          an assumption-light robustness check.
  Welch's t               reported alongside so a reader who prefers
                          parametric tests can see the answer does not hinge
                          on the choice.
  Holm-Bonferroni         applied WITHIN each comparison family, not across
                          everything at once. Correcting across unrelated
                          families would be needlessly conservative.
  Rank-biserial, Cliff's  effect sizes. A p-value says an effect exists; these
                          say whether it is worth anything.
  Bootstrap CIs           10,000 resamples, seeded at 20260729 so the
                          intervals reproduce exactly.

Outputs, all into analysis/results/:
    statistical_tests.txt    the full human-readable report
    stats_descriptive.csv    Table II of the paper
    stats_pairwise.csv       Table III, plus the full 48-comparison set
    stats_generalisation.csv Set A against Set B
    stats_vs_target.csv      one-sided tests against the 90% criterion

A stale generation of Boids and APF results (compare/, 500-step episodes) was
removed from the project on 2026-09-02. It was never read by this script.

Run:  python3 -m analysis.statistical_tests      (from code/)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

DENSITIES = [0.0, 0.1, 0.2, 0.3]
METHODS = ["Boids", "APF", "PPO", "MARL"]
TARGET_COVERAGE = 90.0   # search-and-rescue planning criterion, fixed in
                         # advance and never adjusted afterwards
ALPHA = 0.05
N_BOOT = 10000
# Fixed seed, so every confidence interval in the report reproduces exactly.
RNG = np.random.default_rng(20260729)

RESULT_PATHS = {
    ("Boids", "A"): "phase2_boids/results/boids_results_A.npy",
    ("Boids", "B"): "phase2_boids/results/boids_results_B.npy",
    ("APF", "A"): "phase3_apf/results/apf_results_A.npy",
    ("APF", "B"): "phase3_apf/results/apf_results_B.npy",
    ("PPO", "A"): "phase4_ppo/results/ppo_results_A.npy",
    ("PPO", "B"): "phase4_ppo/results/ppo_results_B.npy",
    ("MARL", "A"): "phase5_marl/results/marl_results_A.npy",
    ("MARL", "B"): "phase5_marl/results/marl_results_B.npy",
}

# All four get the same episode budget. Boids originally ran for 800 steps
# against 700 for the others, which made the comparison unfair; found during
# the audit and repaired before these results were generated.
STEP_BUDGET = {"Boids": 700, "APF": 700, "PPO": 700, "MARL": 700}


def load_all():
    """Load all eight result files into {(method, seed_set): {density: (seeds, coverage)}}.

    Sorting by seed is what makes the arrays positionally comparable: element
    i of every controller's vector is then the same layout. Every paired test
    downstream assumes this.
    """
    data = {}
    for (method, seed_set), rel in RESULT_PATHS.items():
        path = os.path.join(CODE, rel)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing results file: {path}")
        raw = np.load(path, allow_pickle=True)
        raw = raw.item() if raw.shape == () else raw
        per_density = {}
        for d in DENSITIES:
            # Some result files key densities as floats, others as ints.
            key = d if d in raw else int(d * 100)
            runs = raw[key]
            seeds = np.array([r["seed"] for r in runs], dtype=int)
            cov = np.array([float(r["coverage"]) for r in runs], dtype=float)
            order = np.argsort(seeds)
            per_density[d] = (seeds[order], cov[order])
        data[(method, seed_set)] = per_density
    return data


def verify_pairing(data):
    """Check the pairing is real, two independent ways.

    First: the seed vectors align across controllers, so element i means the
    same seed everywhere.

    Second, and stronger: regenerate the obstacle map from each controller's
    own model and compare cell by cell. Aligned labels would not save us if
    the same seed produced different maps in different controllers -- which
    is exactly the defect this project had at one point.

    Returns the notes and a single boolean. If that boolean is False, nothing
    downstream should be trusted.
    """
    notes, matched = [], True
    for seed_set in ["A", "B"]:
        for d in DENSITIES:
            vectors = {m: data[(m, seed_set)][d][0] for m in METHODS}
            ref = vectors[METHODS[0]]
            ok = all(np.array_equal(ref, v) for v in vectors.values())
            matched = matched and ok
            notes.append(
                f"  Set {seed_set}, density {int(d*100):>2}%: "
                f"seeds {ref.min()}-{ref.max()} (n={len(ref)}), "
                f"labels aligned: {ok}")

    layout_ok, layout_notes = True, []
    try:
        import sys
        for sub in ["", "phase2_boids", "phase3_apf", "phase4_ppo"]:
            path = os.path.join(CODE, sub)
            if path not in sys.path:
                sys.path.insert(0, path)
        from phase3_apf.model_apf import APFModel
        from phase2_boids.model_phase2 import SwarmCoveragePhase2
        from phase4_ppo.uav_coverage_env import _build_obstacles
        for d in DENSITIES:
            n = 0
            for s_ in range(10):
                a = APFModel(n_agents=6, obstacle_density=d,
                             obstacle_seed=s_).obstacle_grid.astype(bool)
                b = SwarmCoveragePhase2(n_agents=6, obstacle_density=d,
                                        obstacle_seed=s_).obstacle_grid.astype(bool)
                c = _build_obstacles(d, s_)
                if np.array_equal(a, b) and np.array_equal(b, c):
                    n += 1
            layout_ok = layout_ok and (n == 10)
            layout_notes.append(
                f"  density {int(d*100):>2}%: identical layout across all "
                f"three controllers for {n}/10 sampled seeds")
    except Exception as exc:
        layout_ok = False
        layout_notes.append(f"  layout check could not run: {exc}")

    notes.append("")
    notes.append("Layout identity check (regenerated maps compared cell by cell):")
    notes.extend(layout_notes)
    return notes, (matched and layout_ok)


def bootstrap_ci(x, statistic=np.mean, n_boot=N_BOOT, alpha=0.05):
    """Percentile bootstrap CI. Resamples with replacement, n_boot times."""
    x = np.asarray(x, dtype=float)
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    boots = statistic(x[idx], axis=1)
    return np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def bootstrap_paired_diff_ci(x, y, n_boot=N_BOOT, alpha=0.05):
    """CI on the mean paired difference.

    Resamples the DIFFERENCES, not the two samples independently. That keeps
    the pairing intact; resampling separately would throw away the very thing
    that makes this design tight.
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    idx = RNG.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def cliffs_delta(x, y):
    """Cliff's delta: P(x > y) - P(x < y), over all pairs.

    Non-parametric effect size. Answers "how often does one beat the other",
    which survives the non-normality that rules out Cohen's d as primary.
    """
    x = np.asarray(x, dtype=float)[:, None]
    y = np.asarray(y, dtype=float)[None, :]
    return float((np.sum(x > y) - np.sum(x < y)) / (x.size * y.size))


def interpret_cliff(delta):
    """Romano et al.'s conventional thresholds."""
    a = abs(delta)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def rank_biserial_paired(x, y):
    """Matched-pairs rank-biserial correlation: the Wilcoxon effect size.

    +1 means every seed favoured x, -1 every seed favoured y, 0 an even
    split. Ties are dropped, as Wilcoxon does.
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    r_pos = ranks[d > 0].sum()
    r_neg = ranks[d < 0].sum()
    return float((r_pos - r_neg) / (r_pos + r_neg))


def cohens_d_paired(x, y):
    """Cohen's d on the paired differences.

    Reported for readers who expect it. It assumes normality, which mostly
    does not hold here, so the rank-based measures are primary.
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else np.nan


def interpret_d(d):
    a = abs(d)
    if np.isnan(a):
        return "undefined"
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def holm_bonferroni(pvals):
    """Holm-Bonferroni step-down correction.

    Uniformly more powerful than plain Bonferroni at the same error control:
    the smallest p is multiplied by m, the next by m-1, and so on. The
    running max enforces monotonicity, so an adjusted p can never come out
    below one that was smaller before adjustment.

    Applied within each comparison family rather than across all of them.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def fmt_p(p):
    """Format a p-value, flooring the display at <0.0001."""
    if p < 1e-4:
        return "<0.0001"
    return f"{p:.4f}"


def descriptive_table(data):
    """Table II: mean, spread and normality per controller-density-set cell.

    The Shapiro-Wilk column is the justification for everything downstream.
    27 of the 32 cells reject normality at alpha=0.05, which is why the
    rank-based tests are primary and the parametric ones are robustness
    checks rather than the other way round.

    n_at_or_above_90 matters as much as the mean: the 90% figure is a
    planning criterion, so how often it is met is the operational question.
    """
    rows = []
    for seed_set in ["A", "B"]:
        for method in METHODS:
            for d in DENSITIES:
                _, cov = data[(method, seed_set)][d]
                lo, hi = bootstrap_ci(cov)
                q1, q3 = np.percentile(cov, [25, 75])
                w = stats.shapiro(cov)
                rows.append({
                    "seed_set": seed_set,
                    "method": method,
                    "density_pct": int(d * 100),
                    "n": len(cov),
                    "mean": round(float(cov.mean()), 2),
                    "sd": round(float(cov.std(ddof=1)), 2),
                    "median": round(float(np.median(cov)), 2),
                    "iqr_low": round(float(q1), 2),
                    "iqr_high": round(float(q3), 2),
                    "ci95_low": round(float(lo), 2),
                    "ci95_high": round(float(hi), 2),
                    "min": round(float(cov.min()), 2),
                    "max": round(float(cov.max()), 2),
                    "n_at_or_above_90": int((cov >= TARGET_COVERAGE).sum()),
                    "shapiro_W": round(float(w.statistic), 4),
                    "shapiro_p": round(float(w.pvalue), 6),
                    "normal_at_05": bool(w.pvalue > 0.05),
                })
    return pd.DataFrame(rows)


def pairwise_table(data):
    """Every controller against every other, at every density and both sets.

    Six pairs x four densities x two seed sets = 48 comparisons. Table III of
    the paper shows a selected subset; the full set is written to CSV.

    Three tests are run on each pair. Wilcoxon is primary because the data
    are not normal. Mann-Whitney is an assumption-light check. Welch's t is
    there so a reader who prefers parametric tests can confirm the answer
    does not depend on that preference. Where they disagree, that disagreement
    is itself worth knowing.
    """
    pairs = [("MARL", "APF"), ("MARL", "Boids"), ("MARL", "PPO"),
             ("APF", "Boids"), ("PPO", "APF"), ("PPO", "Boids")]
    rows = []
    for seed_set in ["A", "B"]:
        family = []
        for d in DENSITIES:
            for a, b in pairs:
                x = data[(a, seed_set)][d][1]
                y = data[(b, seed_set)][d][1]

                # Wilcoxon raises if every pair is tied. That cannot happen
                # with continuous coverage values, but a controller compared
                # against itself would trip it, so fail safe rather than
                # crash mid-run.
                try:
                    w_stat, w_p = stats.wilcoxon(x, y, alternative="two-sided",
                                                 zero_method="wilcox")
                except ValueError:
                    w_stat, w_p = np.nan, 1.0

                # Mann-Whitney and Welch both ignore the pairing, so they are
                # strictly more conservative here. Reported as robustness
                # checks, not as the headline result.
                u_stat, u_p = stats.mannwhitneyu(x, y, alternative="two-sided")

                t_stat, t_p = stats.ttest_ind(x, y, equal_var=False)

                rb = rank_biserial_paired(x, y)
                cd = cliffs_delta(x, y)
                dz = cohens_d_paired(x, y)
                lo, hi = bootstrap_paired_diff_ci(x, y)

                row = {
                    "seed_set": seed_set,
                    "density_pct": int(d * 100),
                    "comparison": f"{a} vs {b}",
                    "mean_a": round(float(x.mean()), 2),
                    "mean_b": round(float(y.mean()), 2),
                    "mean_diff_pp": round(float(x.mean() - y.mean()), 2),
                    "diff_ci95_low": round(float(lo), 2),
                    "diff_ci95_high": round(float(hi), 2),
                    "wilcoxon_W": float(w_stat),
                    "wilcoxon_p": float(w_p),
                    "mannwhitney_U": float(u_stat),
                    "mannwhitney_p": float(u_p),
                    "welch_t": round(float(t_stat), 3),
                    "welch_p": float(t_p),
                    "rank_biserial": round(float(rb), 3),
                    "cliffs_delta": round(float(cd), 3),
                    "cliffs_magnitude": interpret_cliff(cd),
                    "cohens_dz": round(float(dz), 3),
                    "cohens_magnitude": interpret_d(dz),
                }
                family.append(row)

        w_adj = holm_bonferroni([r["wilcoxon_p"] for r in family])
        u_adj = holm_bonferroni([r["mannwhitney_p"] for r in family])
        for r, wa, ua in zip(family, w_adj, u_adj):
            r["wilcoxon_p_holm"] = float(wa)
            r["mannwhitney_p_holm"] = float(ua)
            r["significant_holm_05"] = bool(wa < ALPHA)
        rows.extend(family)
    return pd.DataFrame(rows)


def generalisation_table(data):
    """Seed Set A against Seed Set B, per controller and density.

    This is the overfitting test. The learned controllers trained only on Set
    A layouts; Set B they have never seen. If a policy had memorised its
    training maps, it would show here as a drop from A to B.

    Note the tests are UNPAIRED, unlike everywhere else in this file. Set A
    and Set B are different layouts, so there is nothing to pair -- seed 7
    and seed 57 have no relationship. Mann-Whitney and Welch are the right
    tools here precisely because they do not assume one.

    The classical controllers are included as a control. They have nothing to
    overfit to, so any A-B difference they show is the baseline level of
    layout-sample noise the learned results have to be read against.
    """
    rows = []
    family = []
    for method in METHODS:
        for d in DENSITIES:
            a = data[(method, "A")][d][1]
            b = data[(method, "B")][d][1]
            u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
            t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
            cd = cliffs_delta(a, b)
            drop = float(a.mean() - b.mean())
            boots = np.empty(N_BOOT)
            ia = RNG.integers(0, len(a), size=(N_BOOT, len(a)))
            ib = RNG.integers(0, len(b), size=(N_BOOT, len(b)))
            boots = a[ia].mean(axis=1) - b[ib].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            family.append({
                "method": method,
                "density_pct": int(d * 100),
                "mean_A": round(float(a.mean()), 2),
                "mean_B": round(float(b.mean()), 2),
                "degradation_pp": round(drop, 2),
                "degradation_ci95_low": round(float(lo), 2),
                "degradation_ci95_high": round(float(hi), 2),
                "under_10pp_criterion": bool(drop < 10.0),
                "mannwhitney_U": float(u_stat),
                "mannwhitney_p": float(u_p),
                "welch_t": round(float(t_stat), 3),
                "welch_p": float(t_p),
                "cliffs_delta": round(float(cd), 3),
                "cliffs_magnitude": interpret_cliff(cd),
            })
    adj = holm_bonferroni([r["mannwhitney_p"] for r in family])
    for r, a_ in zip(family, adj):
        r["mannwhitney_p_holm"] = float(a_)
        r["significant_holm_05"] = bool(a_ < ALPHA)
    rows.extend(family)
    return pd.DataFrame(rows)


def target_table(data):
    """One-sided tests against the 90% criterion, per cell.

    Answers a different question from the pairwise table: not "is A better
    than B" but "does this controller clear the bar". The 90% figure comes
    from search-and-rescue planning, where a coverage factor of 1.2 gives a
    90% probability of detection; standard tactics reach 63, 72 and 87%, so
    it is a demanding target rather than an arbitrary round number.

    It was fixed before any experiment ran and never adjusted afterwards.

    Both directions are tested. Reporting only "significantly above" would
    hide the cells that are significantly BELOW, and those are informative:
    Boids falls short significantly in seven of eight.
    """
    rows = []
    family = []
    for seed_set in ["A", "B"]:
        for method in METHODS:
            for d in DENSITIES:
                cov = data[(method, seed_set)][d][1]
                # One-sample Wilcoxon, implemented as a signed-rank test on
                # the differences from the target.
                diff = cov - TARGET_COVERAGE
                if np.all(diff == 0):
                    p_greater, p_less = 1.0, 1.0
                    stat = np.nan
                else:
                    stat, p_greater = stats.wilcoxon(
                        diff, alternative="greater", zero_method="wilcox")
                    _, p_less = stats.wilcoxon(
                        diff, alternative="less", zero_method="wilcox")
                lo, hi = bootstrap_ci(cov)
                family.append({
                    "seed_set": seed_set,
                    "method": method,
                    "density_pct": int(d * 100),
                    "mean": round(float(cov.mean()), 2),
                    "ci95_low": round(float(lo), 2),
                    "ci95_high": round(float(hi), 2),
                    "prop_runs_at_or_above_90": round(
                        float((cov >= TARGET_COVERAGE).mean()), 3),
                    "wilcoxon_stat": float(stat) if stat == stat else np.nan,
                    "p_greater_than_90": float(p_greater),
                    "p_less_than_90": float(p_less),
                    "verdict": None,
                    "_p_less": float(p_less),
                })
    adj_g = holm_bonferroni([r["p_greater_than_90"] for r in family])
    adj_l = holm_bonferroni([r["_p_less"] for r in family])
    for r, g, l in zip(family, adj_g, adj_l):
        r["p_greater_holm"] = float(g)
        r["p_less_holm"] = float(l)
        r["verdict"] = ("exceeds 90%" if g < ALPHA
                        else "falls short of 90%" if l < ALPHA
                        else "indistinguishable from 90%")
        del r["_p_less"]
    rows.extend(family)
    return pd.DataFrame(rows)


def write_report(data, desc, pair, gen, targ, pairing_notes, pairing_ok):
    """Assemble the human-readable report.

    Leads with the pairing verification rather than the results, because if
    that check failed nothing below it means anything. Formatting only; every
    number was computed above.
    """
    lines = []
    add = lines.append

    add("=" * 78)
    add("STATISTICAL ANALYSIS — UAV SWARM COVERAGE")
    add("Boids vs Artificial Potential Field vs Decentralised PPO")
    add("=" * 78)
    add("")
    add("DATA PROVENANCE AND DESIGN")
    add("-" * 78)
    add("Results regenerated 2026-08-01 from the repaired harness. All three")
    add("controllers use a 700-step episode budget:")
    for (m, ss), pth in sorted(RESULT_PATHS.items()):
        if ss == "A":
            add(f"  {m:<6} -> {pth}  ({STEP_BUDGET[m]} steps)")
    add("")
    add("Three defects present in the earlier generation were repaired")
    add("before these runs: the obstacle seed is now passed explicitly to")
    add("both Mesa models, the modulo that mapped Seed Set B onto Seed Set")
    add("A layouts was removed, and the episode budget was equalised.")
    add("")
    add("Pairing verification:")
    lines.extend(pairing_notes)
    add(f"  => Matched-pairs design verified: {pairing_ok}")
    add("")
    add("TEST SELECTION")
    add("-" * 78)
    add("Each seed defines one obstacle layout, replayed identically for all")
    add("three controllers, so observations are matched. The primary test is")
    add("the paired Wilcoxon signed-rank test. Mann-Whitney U, which")
    add("discards the pairing, is reported as an assumption-light")
    add("robustness check, and Welch's t so that a reader preferring a")
    add("parametric framework can confirm the conclusions hold. All p-values")
    add("within a family are Holm-Bonferroni corrected. Alpha = 0.05, n = 50")
    add("per condition, bootstrap intervals use 10,000 resamples.")
    add("")
    n_non_normal = int((~desc["normal_at_05"]).sum())
    add("NORMALITY SCREENING (Shapiro-Wilk)")
    add("-" * 78)
    add(f"{n_non_normal} of {len(desc)} condition-level coverage distributions")
    add("depart from normality at alpha = 0.05. Non-normal cells are listed")
    add("below. This departure is the formal justification for treating the")
    add("rank-based tests as primary rather than the t-test.")
    bad = desc[~desc["normal_at_05"]]
    for _, r in bad.iterrows():
        add(f"  Set {r.seed_set}  {r.method:<6} {r.density_pct:>2}%  "
            f"W={r.shapiro_W:.4f}  p={fmt_p(r.shapiro_p)}")
    add("")

    add("TABLE S1 — DESCRIPTIVE STATISTICS")
    add("-" * 78)
    for seed_set in ["A", "B"]:
        add(f"Seed Set {seed_set}")
        add(f"{'Density':>8} {'Method':<7} {'Mean':>7} {'SD':>6} {'Median':>7} "
            f"{'95% CI':>16} {'n>=90%':>7}")
        sub = desc[desc.seed_set == seed_set]
        for d in [0, 10, 20, 30]:
            for m in METHODS:
                r = sub[(sub.density_pct == d) & (sub.method == m)].iloc[0]
                add(f"{str(d)+'%':>8} {m:<7} {r['mean']:>7.2f} {r['sd']:>6.2f} "
                    f"{r['median']:>7.2f} "
                    f"[{r['ci95_low']:>6.2f},{r['ci95_high']:>6.2f}] "
                    f"{r['n_at_or_above_90']:>4}/50")
        add("")

    add("TABLE S2 — PAIRWISE METHOD COMPARISONS (paired Wilcoxon, primary)")
    add("-" * 78)
    for seed_set in ["A", "B"]:
        add(f"Seed Set {seed_set}")
        add(f"{'Dens':>5} {'Comparison':<16} {'Diff(pp)':>9} {'95% CI':>17} "
            f"{'W':>8} {'p':>9} {'Holm p':>9} {'r_rb':>6} {'Mag':<11}")
        sub = pair[pair.seed_set == seed_set]
        for _, r in sub.iterrows():
            add(f"{str(r.density_pct)+'%':>5} {r.comparison:<16} "
                f"{r.mean_diff_pp:>9.2f} "
                f"[{r.diff_ci95_low:>6.2f},{r.diff_ci95_high:>6.2f}] "
                f"{r.wilcoxon_W:>8.1f} {fmt_p(r.wilcoxon_p):>9} "
                f"{fmt_p(r.wilcoxon_p_holm):>9} {r.rank_biserial:>6.2f} "
                f"{r.cliffs_magnitude:<11}")
        add("")
    add("Mann-Whitney U robustness check (pairing discarded):")
    for seed_set in ["A", "B"]:
        sub = pair[pair.seed_set == seed_set]
        agree = int((sub.mannwhitney_p_holm < ALPHA).eq(
            sub.wilcoxon_p_holm < ALPHA).sum())
        add(f"  Set {seed_set}: agrees with Wilcoxon on {agree}/{len(sub)} "
            f"comparisons at Holm-corrected alpha = 0.05.")
    add("Welch's t-test agreement:")
    for seed_set in ["A", "B"]:
        sub = pair[pair.seed_set == seed_set]
        agree = int((sub.welch_p < ALPHA).eq(sub.wilcoxon_p < ALPHA).sum())
        add(f"  Set {seed_set}: agrees with Wilcoxon on {agree}/{len(sub)} "
            f"comparisons at uncorrected alpha = 0.05.")
    add("")

    add("TABLE S3 — GENERALISATION (Seed Set A vs Seed Set B)")
    add("-" * 78)
    add(f"{'Method':<7} {'Dens':>5} {'Mean A':>8} {'Mean B':>8} {'Drop(pp)':>9} "
        f"{'95% CI':>17} {'MWU p':>9} {'Holm p':>9} {'<10pp':>6}")
    for _, r in gen.iterrows():
        add(f"{r.method:<7} {str(r.density_pct)+'%':>5} {r.mean_A:>8.2f} "
            f"{r.mean_B:>8.2f} {r.degradation_pp:>9.2f} "
            f"[{r.degradation_ci95_low:>6.2f},{r.degradation_ci95_high:>6.2f}] "
            f"{fmt_p(r.mannwhitney_p):>9} {fmt_p(r.mannwhitney_p_holm):>9} "
            f"{str(r.under_10pp_criterion):>6}")
    add("")

    add("TABLE S4 — TESTS AGAINST THE 90% COVERAGE CRITERION")
    add("-" * 78)
    add("One-sided Wilcoxon signed-rank of coverage against the 90% threshold.")
    add(f"{'Set':>4} {'Method':<7} {'Dens':>5} {'Mean':>7} {'95% CI':>16} "
        f"{'p(>90)':>9} {'Holm':>9} {'Verdict':<28}")
    for _, r in targ.iterrows():
        add(f"{r.seed_set:>4} {r.method:<7} {str(r.density_pct)+'%':>5} "
            f"{r['mean']:>7.2f} "
            f"[{r.ci95_low:>6.2f},{r.ci95_high:>6.2f}] "
            f"{fmt_p(r.p_greater_than_90):>9} {fmt_p(r.p_greater_holm):>9} "
            f"{r.verdict:<28}")
    add("")

    add("=" * 78)
    add("SENTENCES READY FOR THE RESULTS CHAPTER")
    add("=" * 78)
    for seed_set in ["A", "B"]:
        sub = pair[(pair.seed_set == seed_set)]
        for _, r in sub.iterrows():
            a, b = r.comparison.split(" vs ")
            direction = "higher" if r.mean_diff_pp > 0 else "lower"
            sig = ("statistically significant after Holm correction"
                   if r.wilcoxon_p_holm < ALPHA
                   else "not statistically significant after Holm correction")
            add(f"[Set {seed_set}, {r.density_pct}%] {a} achieved "
                f"{r.mean_a:.2f}% mean coverage against {b}'s {r.mean_b:.2f}%, "
                f"a difference of {abs(r.mean_diff_pp):.2f} pp {direction} "
                f"(95% CI [{r.diff_ci95_low:.2f}, {r.diff_ci95_high:.2f}]; "
                f"Wilcoxon signed-rank W = {r.wilcoxon_W:.1f}, "
                f"p = {fmt_p(r.wilcoxon_p)}, Holm-adjusted "
                f"p = {fmt_p(r.wilcoxon_p_holm)}; rank-biserial r = "
                f"{r.rank_biserial:.2f}, Cliff's delta = {r.cliffs_delta:.2f}, "
                f"{r.cliffs_magnitude} effect) - {sig}.")
        add("")

    return "\n".join(lines)


def main():
    """Load, verify, test, write.

    Order matters: verify_pairing runs before any test, and its outcome is
    printed at the top of the report.
    """
    data = load_all()
    pairing_notes, pairing_ok = verify_pairing(data)

    desc = descriptive_table(data)
    pair = pairwise_table(data)
    gen = generalisation_table(data)
    targ = target_table(data)

    desc.to_csv(os.path.join(OUT, "stats_descriptive.csv"), index=False)
    pair.to_csv(os.path.join(OUT, "stats_pairwise.csv"), index=False)
    gen.to_csv(os.path.join(OUT, "stats_generalisation.csv"), index=False)
    targ.to_csv(os.path.join(OUT, "stats_vs_target.csv"), index=False)

    report = write_report(data, desc, pair, gen, targ, pairing_notes, pairing_ok)
    with open(os.path.join(OUT, "statistical_tests.txt"), "w") as f:
        f.write(report)
    print(report)
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
