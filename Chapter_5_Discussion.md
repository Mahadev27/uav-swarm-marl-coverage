# Chapter 5 — Discussion

Chapter 4 reported what was measured. This chapter explains what it means, why the pattern arose, how it sits against the literature reviewed in Chapter 2, and what it does and does not license anyone to conclude. No new numbers are introduced; every figure quoted here appears in Chapter 4.

---

## 5.1 Answering the research question

The research question asked:

> Does a fully decentralised multi-agent PPO achieve ≥90% area coverage across obstacle densities of 0–30% in a 50×50 grid, and does its coverage degrade by less than 10 percentage points on layouts unseen during training, compared to a standalone APF controller and a Boids swarm baseline?

The answer has three parts.

**Does decentralised PPO achieve ≥90% coverage? No.** PPO's mean coverage ranged from 82.11% to 85.98% across the eight method-density-seed-set cells. Every cell falls significantly short of 90% under a one-sided Wilcoxon test after Holm correction. At most 15 of 50 runs in any cell reached the threshold. The shortfall is not marginal: even the best cell sits four percentage points below the criterion, with a confidence interval that does not approach it.

**Does it degrade by less than 10 pp on unseen layouts? Unknown.** The question cannot be answered from these experiments. Seed Set B, intended as the held-out set, uses the same fifty obstacle layouts as Seed Set A. The measured difference of 0.23 to 2.04 pp is policy stochasticity on identical environments. Reporting it as a generalisation result would be wrong, and the fix — removing a modulo operator and retraining — is cheap enough that it should be done rather than caveated.

**Does it compare favourably to the baselines? No, in both directions.** PPO was significantly worse than APF at every density, by 4.51 to 11.38 percentage points, with large effect sizes throughout (Cliff's δ −0.53 to −0.87). It was statistically indistinguishable from Boids at every density, with the largest absolute effect size across all eight comparisons being 0.21.

The hypothesis motivating this study is not supported. The finding is that a fully decentralised learned policy with a 5×5 local observation window and no inter-agent communication does not match a well-tuned reactive controller on this task, and does not beat a simple flocking heuristic either.

---

## 5.2 Why APF won

Three properties of the task favour APF, and they compound as obstacle density rises.

**The environment matches APF's assumptions almost exactly.** APF was designed for reactive navigation in continuous space with local obstacle sensing. A 50×50 grid with an unexplored-cell attractor and obstacle repulsors is close to the canonical setting for the method. The controller has no learning to do because the correct behaviour is already encoded in the force field.

**Obstacles help APF more than they hurt it.** This is the most instructive result in Chapter 4 and it runs against intuition. APF coverage *increased* with obstacle density, from 90.55% at 0% obstacles to 93.52% at 20% on Seed Set A, and the same pattern appears on Seed Set B. The likely mechanism is that obstacles act as repulsive sources that break the symmetry of the swarm. In an empty grid, six agents starting from the same corner under mutual repulsion disperse into a fairly regular pattern and then have little to distinguish one direction from another; coverage suffers from agents re-sweeping similar regions. Obstacles inject structure, channelling agents down corridors and around walls, which decorrelates their trajectories. The load-balance figures support this: APF's coefficient of variation is lowest of the three methods at every density.

An alternative explanation must be acknowledged. The coverage metric is the fraction of *free* cells entered. As obstacle density rises, the number of free cells falls — at 30% density roughly 750 of 2,500 cells are blocked. The same absolute number of cells visited therefore yields a higher percentage. Chapter 4's data cannot separate these two mechanisms, and doing so would require reporting absolute cells covered alongside the percentage. This should be added.

**The local-minimum escape mechanism is doing real work.** APF's classical failure mode is trapping in concave geometry, and the structured obstacle generator produces exactly such geometry — walls, L-shapes and 2×2 blocks. That APF does not collapse at 30% density indicates the escape behaviour is effective. This is worth stating because Chapter 2 noted that classical baselines in the MARL literature are frequently included without a stated escape mechanism, which makes them easy to beat. The APF controller here is a hard baseline, and beating it would have been a meaningful result. Failing to beat it is also meaningful.

---

## 5.3 Why PPO underperformed

The result is consistent with theory. Fusco et al. (2025) state the mechanism directly: in a fully decentralised structure "no information is exchanged between the agents and each agent would act based on their local observations", with the consequence that "the actions of the agents would be suboptimal due to their limited information about the environment" and that "RL algorithms might have a hard time converging when the training is fully independent due to high partial observability."

This study's design maximises exactly that penalty. Each agent observes a 5×5 window — 25 of 2,500 cells, or 1% of the environment — and receives no communication from the other five. Four specific consequences follow, and each is visible in the data.

**Coverage is a global objective pursued with local information.** An agent cannot tell whether a neighbouring region is unexplored or already swept by a teammate it cannot see. The optimal joint policy requires spatial partitioning of the arena, and partitioning cannot be represented in a 5×5 window. The classical controllers face the same observational limit but sidestep it: APF's mutual repulsion produces implicit partitioning through the force field, without needing to represent it.

**The load-balance figures show the failure directly.** PPO's coefficient of variation of cells-per-UAV is 0.417 to 0.482, against 0.057 to 0.166 for APF. The raw spread is starker than the coefficient suggests: within a single PPO run the busiest agent covers on average 11.6 times as many cells as the least busy at 0% obstacles, rising to 41.8 times at 30% obstacles, with individual runs reaching a ratio of 120. Several agents are contributing almost nothing. This is the signature of failed spatial coordination — the swarm is not dividing the arena, it is letting one or two agents do the work while the rest idle or mill. It is the single clearest explanation for the coverage gap, and it is a coordination failure, not a navigation failure.

**The near-zero collision count is the same failure seen from the other side.** PPO records 0.8 to 1.1 near-collision events per run against 574 to 1,243 for the classical controllers. This is not a safety achievement; it is evidence that the collision penalty in the reward function dominated the exploration bonus. A policy that learns "stay far from other agents" satisfies the penalty term cheaply and reliably, whereas learning to cover the arena efficiently requires a much harder credit assignment over long horizons. The agents took the easy gradient. The shorter total path length — 8 to 15% below the classical controllers — points the same way: the policy is moving less, not moving smarter.

**Training breadth was insufficient for the state space.** The policy was trained for 500,000 steps per density condition on fifty layouts. Because of the modulo defect, those fifty layouts were also the evaluation layouts, so the reported coverage is *training-set* performance. The policy failed to reach 90% even on maps it had effectively memorised. That is a stronger negative result than a generalisation gap would have been, and it should be stated that way: the ceiling here is representational and architectural, not a matter of overfitting.

---

## 5.4 Why PPO tied with Boids rather than beating it

PPO matched a controller with three hand-written rules and no learning at all. Two readings are available and the data does not fully separate them.

The **charitable reading** is that Boids is a stronger baseline than it appears. The dispersion-modified variant used here is effectively a reactive controller optimised for spread, which is most of what coverage requires. Beating it needs genuine long-horizon planning, which a 5×5 window cannot support.

The **less charitable reading** is that the PPO policy did not learn much beyond obstacle avoidance and mutual separation — which is what Boids implements by construction. On this reading the two methods tie because they converged on the same behaviour, one by design and one by gradient descent, and the learned version paid 500,000 training steps for it.

The load-balance figures favour the second reading. If PPO had learned genuine coordination beyond Boids' rules, its load distribution should be more even than Boids', not four times less even. It is the least balanced of the three methods.

One caveat cuts the other way and must be stated. Boids ran with an 800-step episode budget against PPO's 700 — a 14% advantage. Equalising the budgets would lower Boids' coverage and could turn the PPO-versus-Boids ties into small PPO wins. The honest position is that the tie result is unresolved until the budgets are equalised, and the re-run should be done. It does not affect the APF comparison, which was already on equal budgets.

---

## 5.5 What PPO was actually better at

Reporting only the coverage deficit would misrepresent the results. PPO was better on three secondary measures, and the pattern is coherent.

**Collision avoidance.** Three orders of magnitude fewer near-collision events. For an application where airframe loss is expensive and coverage shortfall is recoverable — inspection of infrastructure, operations in confined airspace — this may matter more than four percentage points of coverage. It is also the one property that came from learning rather than from design.

**Computational cost is flat in environment complexity.** PPO's per-step inference cost stayed at 0.186–0.192 ms regardless of obstacle density, because a forward pass through a fixed network costs the same whatever the map contains. APF's cost rose from 0.267 to 0.638 ms and Boids' from 0.075 to 0.279 ms, because both recompute pairwise forces or neighbour sets whose size grows with the number of obstacles. At 30% density PPO is 3.3× cheaper per step than APF. On an embedded flight controller in a cluttered environment, that predictability has real value — and it is the standard argument for learned policies, supported here by measurement rather than assertion.

**Path efficiency.** 8 to 15% shorter total travel distance. Read alongside the lower coverage this is ambiguous — it may reflect efficiency or simply less movement — but as an energy proxy it is the metric Fusco et al. (2025) treat as primary, and on their chosen axis PPO looks better than it does on coverage.

The defensible summary is that decentralised PPO trades coverage for safety and computational predictability. That is a real engineering trade-off and a legitimate contribution, provided it is presented as a trade-off rather than dressed up as a win.

---

## 5.6 Relation to the literature

**The result confirms a prediction the empirical literature mostly avoids testing.** Chapter 2 established that Fusco et al. (2025) set out the theoretical penalty for full decentralisation, and that almost every empirical coverage paper avoids paying it — Yu et al. (2025) through event-triggered hierarchy, He et al. through a hierarchical swarm framework, Leong et al. (2024) through viewpoint assignment, Xie et al. (2025) through centralised cell decomposition. This study did not avoid it, and observed the predicted penalty. That is a modest but genuine contribution: it supplies the empirical data point the field's own theory implies but does not measure.

**It suggests published MARL-versus-classical comparisons should be read carefully.** APF beat a learned policy by up to 11.4 pp here, with a properly configured escape mechanism. Chapter 2 noted that classical baselines in this literature are routinely under-specified. The reasonable inference is not that all such comparisons are wrong, but that the strength of the classical baseline is a first-order determinant of the result, and papers that do not state their baseline's parameters cannot be assessed on this point.

**The evidentiary standard is the most transferable part of this work.** None of the papers reviewed in Chapter 2 reports a significance test, an effect size, or a replication count above ten. This study reports 50 replications per cell, Mann-Whitney U tests with Holm–Bonferroni correction, Cliff's delta and bootstrap confidence intervals. If the results chapter had shown PPO winning by 2 pp, that standard would have been what determined whether the claim was believable. It is worth arguing in the viva that this, rather than any individual number, is the methodological contribution.

**Direct numerical comparison with prior work is not available, and that is a finding.** Fusco et al. report no coverage percentage; He et al.'s ">90%" is communication coverage of ground users, not swept area; papers that terminate on full coverage report time-to-completion rather than coverage-at-budget. The field has no shared benchmark for this task. Chapter 2's comparison table therefore compares experimental design rather than headline numbers, which is the only honest form the comparison can take.

---

## 5.7 Limitations

These are ordered by how much they constrain the conclusions.

**Generalisation is untested.** The single most important limitation. The held-out set was not held out. Nothing in this work supports or refutes a claim about generalisation to unseen layouts. The claim should be removed from the abstract and conclusion until the experiment is re-run.

**Results are not reproducible as the code currently stands.** The Boids and APF models draw their obstacle seed from an unseeded internal generator, so identical seeds produce different maps. Re-running the experiment today would not reproduce the tables in Chapter 4. This is a defect of the harness rather than of the methods, and it is fixable in a few lines, but it must be fixed before the numbers can be defended.

**Episode budgets were unequal.** Boids received 800 steps against 700 for APF and PPO. The APF-versus-PPO conclusions are unaffected. The PPO-versus-Boids tie is unresolved.

**The environment is a substantial simplification.** A 2D grid with discrete movement, perfect single-cell sensing, no motion uncertainty, no battery limit and no communication constraints. Real UAVs face all five. The absolute coverage numbers should not be read as predictions of field performance; only the relative ordering of the methods transfers, and even that is conditional on the simplifications not favouring one method.

**Swarm size was fixed at six for the main experiment.** The sensitivity analysis in Table 4.9 shows all three methods exceed 90% at nine UAVs and above, and all three fall below 71% at three. Six sits in the steep part of that curve, where small differences in coordination quality are amplified. The result "PPO fails to reach 90%" is specific to six UAVs on a 50×50 grid; at nine agents PPO reaches 94.87%. Fixing the criterion but not the agent count is arguably the weakest design decision in the study, and a fairer framing of the finding is that *PPO needs more agents than APF to reach the same coverage*.

**PPO hyperparameters were not tuned.** A single configuration was trained. The comparison is between a tuned classical controller and an untuned learned one. This is a real asymmetry and it should be stated in the conclusion rather than left for an examiner to raise.

**Two metrics are not usable.** PPO coverage curves are synthetic and PPO redundancy is a stub returning zero. Neither can appear in the submitted document without being fixed.

---

## 5.8 Future work

In descending order of value per unit of effort.

1. **Repair the seeding and re-run.** Removing the modulo in the PPO environment and plumbing the seed through the Mesa models converts an unreproducible study into a reproducible one. Roughly a day of compute. Nothing else on this list is worth doing first.
2. **Equalise episode budgets and re-run all three methods at a common value.** Resolves the PPO-versus-Boids question.
3. **Report absolute cells covered alongside percentage coverage.** Separates the two competing explanations for APF's rising coverage with density, given in Section 5.2.
4. **Add minimal communication to the PPO agents.** Sharing only each agent's position with its nearest neighbours would test whether the deficit is caused by partial observability or by the algorithm. Given that the load-balance figures point squarely at coordination failure, this is the highest-value scientific extension.
5. **Compare against a hierarchical MARL variant**, following Yu et al. (2025), to quantify how much of the gap decentralisation costs.
6. **Tune PPO hyperparameters and increase training steps**, so that the comparison is between two tuned systems.
7. **Vary swarm size in the main experiment** rather than in a 15-seed side analysis, and report the agent count each method needs to reach 90%. On the evidence of Table 4.9 this may be the more informative framing of the whole study.

---

## 5.9 Conclusion of the discussion

A fully decentralised PPO policy with a 5×5 local observation window and no inter-agent communication reached 82–86% area coverage on a 50×50 grid with six UAVs. It did not reach the 90% criterion at any obstacle density. It performed significantly worse than a tuned artificial potential field controller at every density, with large effect sizes, and the gap widened as obstacles increased. It did not perform significantly better than a Boids flocking baseline at any density.

The learned policy was better at collision avoidance by three orders of magnitude, and its per-step computational cost was constant in environment complexity where both classical controllers' costs grew. The trade-off it offers is coverage for safety and predictability.

The mechanism behind the coverage deficit is visible in the data: PPO's distribution of work across the swarm was 2.9 to 7.3 times less even than APF's. The failure is one of coordination, not navigation, and it is the failure that theory predicts for a fully decentralised policy under high partial observability. Whether communication would close the gap is the obvious next experiment.
