# Phase 3 — does disconnection predict outcome better than location?

The go/no-go question for the whole project, run on **ARC**: 228 subjects with both a lesion
mask and a WAB Aphasia Quotient, registered to MNI by `src/lesionconn/registration.py`.

`src/lesionconn/modeling.py`, `scripts/19_train.py`. Registration: [REGISTRATION.md](REGISTRATION.md).

## The answer

**B ≈ A.** Disconnection is not distinguishable from the location baseline. That is the
roadmap's middle branch — *dig into representation before declaring anything* — and it is
neither the win nor the failure.

**This replicates.** Two larger studies using the same method class — a lesion mask embedded
in a normative connectome — report the same ordering: Salvalaggio 2020 (n=132) finds
structural disconnection predicts "to a level comparable to lesion information", and
Khalilian 2025 (n=340) finds lesion masks outright superior to unthresholded disconnection
maps. The absolute R² values here are in the normal published range too. See
[LITERATURE.md](LITERATURE.md), which also records the contradicting results and what
separates them.

The point estimates say otherwise, which is the interesting part.

| arm | best R² | 95% interval across folds | features |
|---|---|---|---|
| network edges, counted matrix | **0.481** | [0.222, 0.659] | 136 |
| network edges, inferred matrix | 0.465 | [0.240, 0.593] | 136 |
| volume + network edges | 0.464 | [0.219, 0.599] | 137 |
| volume + tracts | 0.448 | [0.257, 0.624] | 81 |
| **location — Model A** | 0.444 | [0.299, 0.625] | 463 |
| volume + location | 0.435 | [0.269, 0.649] | 464 |
| **tract severance — Model B** | 0.421 | [0.266, 0.606] | 80 |
| **lesion volume alone** | **0.398** | [0.200, 0.623] | 1 |

Read that table on its own and disconnection wins. `verdict()` originally said exactly that.
It was wrong.

## Why the ordering is not a result

Every arm is scored on the **identical 25 outer folds** — `RepeatedKFold` with a fixed seed —
so the comparison is *paired*, and the intervals above span roughly [0.2, 0.65] for every
single arm. Ranking medians of distributions that overlap that much will call a coin flip a
finding.

Wilcoxon signed-rank per arm, Bonferroni over the comparisons actually made:

| comparison | median diff | folds better | p | p (Bonferroni) |
|---|---|---|---|---|
| network edges (counted) vs volume | +0.055 | 16/25 | 0.032 | 0.22 |
| location vs volume | +0.026 | 16/25 | 0.134 | 0.94 |
| tracts vs volume | +0.012 | 14/25 | 0.252 | 1.00 |
| network edges (counted) vs location | +0.011 | 14/25 | 0.325 | 1.00 |
| network edges (inferred) vs volume | +0.006 | 14/25 | 0.491 | 1.00 |
| **network edges (inferred) vs location** | **−0.022** | 11/25 | 0.634 | 1.00 |
| tracts vs location | +0.014 | 13/25 | 0.653 | 1.00 |

**Nothing beats location. Nothing survives correction.** The single nominally significant
result is against *volume*, not against the baseline the thesis is about, and it dies at
p = 0.22 corrected.

`modeling.verdict()` now runs this test itself and prints it last, so the harness can no
longer report an ordering of overlapping medians as a finding.

## The number that should worry you most

**Lesion volume alone — one column — reaches R² = 0.398.** Every other arm is fighting over
the remaining ~0.05. This was predicted before the run: on the 108, log lesion volume is
recoverable from the disconnection features at R² 0.47 (tracts) to 0.66 (network edges), so
most of what those features carry *is* size.

## The correction family is a function of the command line

The per-comparison numbers above are stable. The **corrected** ones are not, and that is a
property of the harness rather than of the data.

`paired_table` Bonferroni-corrects over the rows it happens to produce, and it produces one row
per (arm, baseline) pair among the arms that were run. Re-running the identical measurement with
the five `volume+X` incremental arms in the same command grows the table from 7 rows to 15, and
every corrected p-value moves with it:

| | family of 7 | family of 15 |
|---|---|---|
| network edges (counted) vs volume | p = 0.032 → **0.22** | p = 0.032 → **0.48** |
| location vs volume | p = 0.134 → 0.94 | p = 0.134 → 1.00 |

**Every `p_wilcoxon` is identical between the two runs** — 0.0318, 0.1336, 0.2521, 0.3254,
0.4908, 0.6338, 0.6528 reproduce exactly. Only the denominator changed, because a different
number of questions was asked in one invocation.

Nothing survives at either family size, so the Phase 3 answer does not depend on this. But the
fragility is real and is recorded rather than fixed quietly, because the tempting fix — shrinking
the family — moves p-values in the direction that manufactures results. The table at the top of
this document is the 7-comparison family: the four feature sets against the two baselines, with
the incremental `volume+X` arms reported separately in their own block, which is how
`verdict()` prints them.

## Age, sex and time post-stroke: measured, and they are nothing

`participants.tsv` carries all three for **228 of 228** subjects, so this cost no new data. The
hypothesis was that time post-stroke explains variance currently attributed to anatomy. It does
not.

| arm | best R² | 95% interval across folds | features |
|---|---|---|---|
| lesion volume alone | +0.3984 | [0.200, 0.623] | 1 |
| **volume + covariates** | **+0.3868** | [0.164, 0.611] | 4 |
| **covariates alone** | **−0.0143** | [−0.058, +0.031] | 3 |

Adding age, sex and log days post-stroke to lesion volume makes the model **worse**. Alone they
are indistinguishable from noise.

Univariate correlations with WAB-AQ, and the same correlations once log lesion volume is
partialled out:

| covariate | r | partial r, given log volume |
|---|---|---|
| `wab_days` | −0.079 (p = 0.23) | **−0.002** (p = 0.98) |
| `age_at_stroke` | +0.018 (p = 0.79) | −0.095 (p = 0.15) |
| `sex` (F) | **+0.185 (p = 0.005)** | +0.113 (p = 0.088) |

Two things to read here. **`wab_days` is not weak, it is absent** — partial r = −0.002 is as
close to nothing as this cohort can measure. ARC is a *chronic* aphasia cohort with a median 721
days post-stroke, so the subjects were sampled after recovery had largely plateaued; the
variable would matter in a cohort scanned in the first weeks, and this is not one.

**The apparent sex effect is lesion volume.** It is nominally significant univariately
(point-biserial r = +0.185, p = 0.005; Mann-Whitney p = 0.008, mean WAB-AQ 69.9 against 59.2)
and dies under the volume control, because women in this cohort have smaller lesions: mean log
volume 10.40 against 10.85 for men. Reporting the univariate p would have been a lesion-size finding
dressed as a demographic one — the same confound this whole document is organised around.

`covariates` is registered with `kind="control"` and `is_control_arm` keeps every all-control
arm out of the thesis correction family, so asking this question does not make the thesis
comparisons worse (see the section above for why that matters).

## Volume regressed out properly, and it takes everything with it

The `volume+X` arms were the weak form of the incremental question, and on trees they are
close to useless for it: `volume+network_edges` (0.4641) scores *below* `network_edges` alone
(0.4653), because the tree already recovers volume from the features and the extra column only
costs it a split.

The sharp form is to remove volume from the **target**. `evaluate(confounds=...)` fits a
`hist_gb` nuisance model on log lesion volume **inside each training split**, subtracts its
prediction from both sides of that split, and scores the arm on what is left. Fitting the
nuisance model once over all 228 subjects — the obvious implementation — leaks the test
outcomes into the residual and inflates every arm at once, so it is refitted 25 times.

| arm | best R² on the volume-free residual | 95% interval | folds > 0 |
|---|---|---|---|
| location | **−0.0054** | [−0.312, +0.120] | 12/25 |
| tracts | **−0.0107** | [−1.223, +0.034] | 7/25 |
| *volume — the control* | *−0.0146* | *[−0.100, +0.010]* | *5/25* |
| network edges, inferred | **−0.0161** | [−3.156, +0.018] | 2/25 |
| covariates | −0.0244 | [−0.161, +0.025] | 7/25 |
| network edges, counted | **−0.0263** | [−374.8, +0.067] | 6/25 |

**Every arm is at or below zero. So is every one of the 18 (feature set × model) cells** — not a
single combination in the table predicts the residual better than its own mean.

Read the control row first. Volume predicting its own residual comes back at −0.0146, which is
what a correctly removed confound looks like; anything appreciably above zero there would mean
the nuisance model had leaked or left signal behind, and the whole table would be void.

**What this licenses, and what it does not.** It licenses a stronger statement than the raw
table supports: the ~0.05 that separates the arms up there is not information about topology
sitting on top of lesion size, because when size is taken out of the target there is nothing
left for any arm to find — *including the location baseline*. It does **not** license "there is
no signal". R² ≤ 0 on 228 subjects is a statement about detectability, and the intervals here
are the widest in the project: ridge and elastic-net on the counted network edges reach −147.8
and −374.8 on their worst folds, the same column-scaling pathology documented below. The
`hist_gb` rows are the ones with bounded intervals ([−0.31, +0.15] at worst), and they centre
below zero too.

The honest summary is that on this cohort, at this n, **lesion volume is the model** and
everything else is decoration.

## A result that had to be thrown out

`network_edges_streamline` scored **−0.119 with p = 0.84** in the permutation null while
scoring +0.481 in the main comparison. Not a contradiction — a different model.

The permutation test runs **ridge**, and ridge is unusable on these features: its fold
interval reaches **−63.5**, with only 17 of 25 folds positive. The mechanism is measured, not
guessed — column standard deviations across the 136 network edges span a factor of **7×10⁵**,
so a column that is near-constant within a training fold explodes when a test subject is
nonzero, and `StandardScaler` amplifies exactly that.

`hist_gb` was stable in every arm (25/25 folds positive, every lower bound ≥ 0.20), which is
why the paired test uses it.

~~**The consequence is a real gap:** the permutation null validates the *linear* arms and not
the boosted-tree arms that produce the headline numbers. It is not closed.~~ **Closed** — see
below.

## The null, on the models the headline numbers come from

200 shuffles per arm, `hist_gb`, the same family the paired test uses. 106.7 minutes of
compute, checkpointed per arm; a session ending under it killed the run twice and it resumed
from disk both times.

| arm | features | observed R² | shuffled median | shuffled p95 | p | minutes |
|---|---|---|---|---|---|---|
| network edges, counted | 136 | **+0.5104** | −0.2044 | −0.1087 | **0.005** | 10.7 |
| location — Model A | 463 | +0.4487 | −0.2068 | −0.0748 | **0.005** | 74.7 |
| tract severance — Model B | 80 | +0.4340 | −0.2061 | −0.0881 | **0.005** | 4.2 |
| network edges, inferred | 136 | +0.4325 | −0.2070 | −0.0817 | **0.005** | 12.0 |
| lesion volume | 1 | +0.4160 | −0.1036 | −0.0391 | **0.005** | 3.3 |
| *covariates — the control* | *3* | *−0.1347* | *−0.1406* | *−0.0504* | *0.4527* | *1.8* |

**p = 0.005 is the floor**, `(1 + 0)/(1 + 200)`: not one shuffle in 200 reached the observed
score for any of the five real arms. The family here is 6 tests, so Bonferroni gives
0.005 × 6 = 0.03 — all five survive correction, and the family size is stated because the
section above this document's paired table exists entirely about how easily it moves.

**Read the control row first.** `covariates` lands at p = 0.45, and it must: its observed score
is −0.1347, indistinguishable from its own shuffled median of −0.1406. A harness that returned
a small p for age, sex and days-post-stroke would be manufacturing signal, and every other row
here would be void. This was written down as a prediction before the arm ran.

**The thrown-out result is explained rather than merely retracted.** `network_edges_streamline`
scored −0.119 at p = 0.84 under ridge. Under `hist_gb` the same features, the same shuffles and
the same seed give **+0.5104 at p = 0.005**. Nothing about the data changed; the column-scaling
pathology described above was the whole of it.

**Observed here ≠ the headline R².** `permutation_p` scores with single-level CV tuned on R²,
while the table at the top of this document is nested CV tuned on MAE. The two agree to within
0.037 across all six arms (largest divergences: `covariates` −0.037, `network_edges` −0.033,
`network_edges_streamline` +0.030). They are the same quantity measured two ways, not a
discrepancy.

**What this closes, and what it does not.** It closes the gap: the boosted-tree arms that
produce every headline number are now shown to beat randomised outcomes, decisively. It says
**nothing about B vs A** — each arm is tested against chance, never against another arm. The
Phase 3 answer is still **B ≈ A**, and it is still the paired Wilcoxon over identical folds
that decides it. A permutation p of 0.005 on four different feature sets is exactly what you
would expect from four feature sets that are all largely measuring lesion size.

## Thresholding the maps: tested, and it changes nothing

The one thing the literature said to try (LITERATURE.md, Khalilian 2025). The design and four
predictions were committed before any fit; the full sweep, controls and scoring are in
[THRESHOLDING.md](THRESHOLDING.md) §9. The threshold promoted by the pre-fixed rule was a
**≥ 10-fibre support floor on the counted matrix**, and it was run alongside the published arms
in their own 9-row family:

| | `network_edges_streamline@sup10` |
|---|---|
| raw R², hist_gb | **+0.5010** [0.267, 0.657] (unthresholded: +0.4806) |
| vs location, paired | +0.0384, 18/25, p = 0.0136, **p_bonf = 0.1224** |
| volume-free residual | **−0.0268** |
| permutation null | +0.5382 observed, p = 0.005 |

**Nothing beats location after correction, and nothing survives removing volume.** The small raw
gain is not selection: a size-matched set of **randomly chosen** edges scores +0.5084, higher
than the support-selected one, and the edges the threshold throws away score +0.4622 on their
own. Thresholding here reduces dimensionality; it does not find the connections that matter.

The run also reproduced every published number in this document exactly (`location` +0.4436,
`volume` +0.3984, residual control −0.0146), so adding the new arms moved nothing that was
already here.

A by-product worth keeping: dropping network columns that are nonzero in fewer than 2% of
subjects (`@prev0.02`) takes ridge on the counted features from a worst fold of **−63.5** to
**+0.265**, 25/25 folds positive. That is the conditioning pathology described above, fixed at
its source. It is a fix for the linear model, not a new signal.

## What this means for Phase 5

- **Coarser, not finer.** The 415×415 parcel edges were already known not to replicate between
  the two matrices (median Spearman −0.318, [DISCONNECTION.md](DISCONNECTION.md)). Network
  edges are the resolution that survives, and they still do not beat location.
- **Volume must be regressed out, not merely reported.** ~~A feature set that is two-thirds
  lesion size cannot demonstrate topology.~~ **Done** — see *Volume regressed out properly*
  above. Every arm, and every model within every arm, scores at or below zero once volume is
  removed from the target fold-safe.
- **n = 228 is the binding constraint, and it cannot be relieved from ARC.** With 95% fold
  intervals ~0.4 wide, a real +0.03 effect is undetectable here.

  > **Correction.** An earlier version of this line read "ARC has 902 sessions and this uses
  > 228 — more subjects is a cheaper path to an answer than a cleverer model." The first half
  > is a misreading and the second half does not follow from it. **902 is scanning sessions,
  > not participants.** `participants.tsv` has **245 rows**; `derivatives/lesion_masks` holds
  > **228 masks, exactly one per subject**; all 228 carry a WAB-AQ *and* complete covariates.
  > The remaining **17 participants have a WAB-AQ and no lesion mask at all**, so they cannot
  > enter a lesion-symptom model on any terms. 228 is the entire labelled cohort, not a
  > sample of it. More subjects means a *different* cohort — and ATLAS, the one with 1,453
  > scans, is the one that withholds outcome scores (ROADMAP Phase 0). Counting the masks
  > costs one command and would have caught this before it was written down.
- ~~**A permutation null on the tree models** would close the gap above.~~ **Done.** All five
  real arms reach the 1/201 floor and the control does not; see the table above.

## What was NOT done

- No enantiomorphic normalisation; cost-function masking only.
- No comparison of `SyNRA` against other transforms — Dice 0.924 was good enough to proceed,
  which is not evidence nothing else is better.
- ~~No correction for age, sex or time post-stroke.~~ **Done, and they are nothing** — see
  *Age, sex and time post-stroke* above. The bullet used to add "all of which plausibly matter
  for WAB-AQ", which was the assumption, not a measurement.
- **No hold-out test set. Every number here is cross-validated on the same 228 subjects** — and
  this is now the *only* one of the four original gaps still open, because it is the one that
  n = 228 makes expensive rather than merely undone. Holding out 20% costs the comparison the
  power it already does not have.

  **Two different claims have been conflated under "no hold-out", and they deserve different
  answers.**

  1. *The A-vs-B comparison.* A hold-out is the wrong tool. Twenty percent is ~46 subjects, and
     the comparison is already underpowered using all 228 across 25 paired folds; a single 46-subject split would decide B vs A by the noise in one draw.
     Nothing is lost by not having one here.
  2. *The shipped M4 model.* This is the real gap. `predict.fit_final` does report out-of-fold
     numbers — `cross_val_predict` wraps the whole `GridSearchCV`, so the quoted `cv_r2`,
     `cv_mae` and 90% residual interval are not in-sample fit. What they cannot account for is
     that the **feature set, the model family and the representation were all chosen by looking
     at these same 228 subjects**, across every arm in this document. That selection makes the
     shipped numbers optimistic by an amount that is not measurable from inside this cohort.
     The report's "typical error ~X points" should be read as a best case.

  No hold-out was spent. Closing (2) honestly would need subjects that played no part in any
  choice above, and ARC has none left: 228 is the entire labelled cohort.

  > **Correction.** "The only one still open" was written while the permutation null was
  > outstanding, so it was not true when written — there were two. With the null closed it is
  > now accurate. Recorded rather than quietly corrected, because a document that silently
  > becomes right is indistinguishable from one that was never wrong.
