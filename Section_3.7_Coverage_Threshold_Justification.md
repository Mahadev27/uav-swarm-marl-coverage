# Section 3.7 — Justification of the 90% Coverage Criterion

*Insert into Chapter 3 (Methodology), immediately after the definition of the coverage metric and before the description of the seed sets.*

---

## 3.7.1 Why the threshold needs justifying

The research question sets a success threshold of ≥90% area coverage. A threshold used as a pass/fail criterion has to come from somewhere. If it comes from an operational standard, it should be cited. If it is a modelling choice, it should be declared as one. Leaving it unstated invites the obvious question of whether it was chosen after seeing the results.

This section states the origin of the figure, sets out the limits of that origin, and records the criterion as a pre-specified decision rule.

## 3.7.2 Operational origin: probability of detection in search and rescue

The 90% figure is taken from search-and-rescue planning practice, where the governing quantity is not area swept but **probability of detection (POD)** — the probability that a searcher or sensor detects a target given that the target is inside the searched region.

In formal search theory, POD is a function of the **coverage factor** *C*:

> C = (number of searchers × distance travelled × sweep width) / (size of the area being searched)

where sweep width is the effective lateral detection range of the searcher (Perkins, 2013).

Perkins (2013), writing for the Centre for Search Research, gives the operational figure directly. Working through an area-search planning example, he states:

> "From the PoD curve for searching an area we can see that to achieve a PoD of 90% we need coverage of 1.2."

The same document records the POD values that correspond to standard search tactics: a two-searcher route search at *C* = 1 yields POD 63%; a four-searcher line at Critical Separation yields *C* = ½ and POD 72%; searchers at half Critical Separation reach *C* = 1 and POD 87%; purposeful wandering at Critical Separation also reaches POD 87%.

Two things follow. First, **90% POD is a recognised planning target in land SAR**, not an arbitrary round number — it is the value a search manager plans searcher spacing to achieve. Second, it is a *demanding* target: none of the standard tactics reaches it, and achieving it requires deliberately tightening searcher spacing to roughly 42% of Critical Separation.

The same POD-versus-coverage framework underpins the United States National Search and Rescue Supplement to the IAMSAR Manual and its Land SAR Addendum, which are the governing planning documents for US SAR operations and which express incident objectives in POD terms.

A comparable figure appears in the UAV literature. He et al. report that their swarm framework achieves an average coverage rate "evidently above 90%" across ground-user counts from 30 to 50, treating 90% as the level at which a deployment is considered acceptable.

## 3.7.3 The construct mismatch, stated plainly

The 90% used in this dissertation is **not** the same quantity as SAR probability of detection, and the difference must be acknowledged rather than glossed over.

| | SAR probability of detection | This study's coverage metric |
|---|---|---|
| What is counted | Probability the target is detected | Fraction of free grid cells entered at least once |
| Sensor model | Probabilistic, with sweep width and lateral-range curve | Deterministic — entering a cell reveals it with certainty |
| Can exceed by searching harder? | Yes, POD rises with coverage factor above 1 | No, capped at 100% |
| Depends on target properties | Yes (size, colour, terrain, vegetation) | No |

Because this study's UAVs have a deterministic, perfect single-cell sensor, its coverage metric is an **upper bound** on what a real system with an imperfect sensor would achieve. A controller reaching 90% cell coverage here would achieve less than 90% POD in the field, because a real sensor sometimes fails to detect a target in a cell it passes over.

The threshold is therefore best understood as follows:

> **The 90% criterion is a decision rule adopted for this study, motivated by the 90% POD planning target used in land search and rescue (Perkins, 2013), but operationalised as deterministic cell coverage rather than probabilistic detection. It is a necessary condition for field-relevant performance, not a sufficient one.**

That is the honest formulation and it is defensible under examination. Claiming the metric *is* an operational standard would not be.

## 3.7.4 Status as a pre-specified criterion

The threshold was fixed in the project proposal before any experiment was run, and is not adjusted in light of the results. This matters because the outcome is unfavourable to the study's original hypothesis: the PPO controller does not reach 90% at any obstacle density, on either seed set. Retaining the criterion unchanged is what makes that a finding rather than a moved goalpost.

Three secondary criteria are reported alongside the threshold so that a reader who considers 90% too strict or too lenient can still interpret the results:

1. **Mean coverage with a 95% bootstrap confidence interval** at every method × density × seed-set cell, so the distance from any threshold can be read directly.
2. **The proportion of runs at or above 90%**, which shows the spread rather than only the centre. This distinguishes a controller that is consistently at 88% from one that averages 88% by mixing 95% and 80% runs.
3. **A one-sided Wilcoxon signed-rank test of coverage against the 90% value**, Holm-corrected, which converts the threshold from an eyeball comparison into a decision with a stated error rate. Results are reported in Table S4.

## 3.7.5 Sensitivity of the conclusions to the threshold

Because the threshold is a choice, the sensitivity of the conclusions to it should be stated. Counting the eight method × density × seed-set cells per controller, and judging by mean coverage:

| Threshold | APF | Boids | PPO |
|---|---|---|---|
| **90%** | 8/8 cells pass | 0/8 | 0/8 |
| **85%** | 8/8 | 6/8 (fails at 30% density, both sets) | 5/8 (fails at 20% and 30% on Set A, 30% on Set B) |
| **80%** | 8/8 | 8/8 | 8/8 |

Two clarifications on the 90% row. Judged by mean alone APF clears 90% in all eight cells, but the one-sided Wilcoxon test in Table S4 is stricter: it finds APF *significantly* above 90% in six cells and statistically indistinguishable from 90% in the two 0%-density cells, where the mean sits just above the line at 90.55% and 90.19%. The test result, not the raw mean, is what should be reported.

The ranking of the three controllers — APF > Boids ≥ PPO — is unchanged at every threshold. The threshold determines how many methods are labelled successful; it does not determine which method is best. This matters because it shows the central finding does not depend on the choice of criterion. Had the threshold been set at 80%, all three methods would have passed and the study would have produced no discrimination at all; had it been set at 95%, none would have passed. The 90% value happens to sit at the point of maximum discrimination between the three controllers, which is convenient but was not the reason for choosing it, and the coincidence should be acknowledged rather than presented as design.

---

## References for this section

Perkins, D. (2013) *Probability of Detection for the Search Manager*. Ashington: The Centre for Search Research. Available at: https://tcsr.org.uk/media/fsldirix/2013-probability-of-detection-for-the-search-manager.pdf

Perkins, D. and Lovelock, D. (2008) *Lateral Range Curves, Search Probabilities, and Grid Searching*. The Centre for Search Research.

United States Coast Guard (2018) *United States National Search and Rescue Supplement to the International Aeronautical and Maritime Search and Rescue Manual*. Washington, DC: National Search and Rescue Committee.

National Search and Rescue Committee, *Land Search and Rescue Addendum to the National Search and Rescue Supplement*. Available at: https://www.dco.uscg.mil/Portals/9/CG-5R/nsarc/Land_SAR_Addendum/

United States Coast Guard, *The Theory of Search — A Simplified Explanation*. Available at: https://navcen.uscg.gov/sites/default/files/pdf/Theory_of_Search.pdf

He, J., Jia, Z., Dong, C., Liu, J., Wu, Q. and Liu, J. 'UAV swarm deployment and trajectory for 3D area coverage via reinforcement learning', Nanjing University of Aeronautics and Astronautics.
