# Thresholding the disconnection matrices — the design, and the predictions

> **Design document. Nothing here has been run.** Predictions P1–P4 are committed below
> *before* any model is fitted, with explicit falsification criteria, because this project
> has set thresholds by first guess and been wrong both times it did so — the ">5 mm off
> midline" laterality rule and the centre-of-mass tolerance, both recorded at
> ROADMAP.md:121, and treated as a standing rule at REGISTRATION.md:62 (*"every threshold in
> this project so far was wrong on the first guess"*). A prediction written after the run is
> not a prediction.
>
> Every number in the *measured* sections was computed read-only from the cached artifacts
> under `data/derived/cohort/features/` and `data/atlases/streamline_index.npz`. No code was
> added or changed. The commands are reproduced at the bottom.

Phase 5's first bullet (ROADMAP.md) and the single actionable finding of LITERATURE.md are
the same instruction: **threshold the disconnection maps before using them as features.**
This document says exactly what "threshold" would have to mean here, what values to sweep,
and what the result must look like for the sweep to count as evidence rather than as a search
over 20 arms for one that clears p = 0.05.

---

## 1. What decision is actually being made

### 1.1 The object

`streamlines.edge_support_matrix()` (`streamlines.py:372`) returns a 415×415 integer matrix:
for each parcel pair, **how many of `S35_1mm.trk`'s 479,457 streamlines have one end in each
parcel**. It is the denominator behind every entry of the counted disconnection matrix —
`streamline_disconnection_matrix` divides severed fibres by `edge_totals` and emits a
fraction in [0, 1].

`MIN_EDGE_STREAMLINES = 10` (`streamlines.py:369`) is the fibre floor. Its docstring is
explicit about scope:

> The matrix ITSELF is never filtered; M4 gets everything and decides for itself. This
> threshold governs reporting only.

It is enforced in exactly one place — `top_disconnections(min_streamlines=...)` at
`streamlines.py:396`, which zeroes low-support edges *for the ranking* and prints the fibre
count beside every row. `build_features` (`modeling.py:198`) never sees it. So the arm that
produces this project's best number, `network_edges_streamline` (R² 0.4806, PHASE3.md), is
built from a matrix in which **a one-fibre edge and a 1,553-fibre edge are the same number
on the same scale.**

### 1.2 Why that matters here specifically, measured

The support distribution is severely skewed. Over the 6,575 connected parcel pairs
(7.65% density of the 85,905 upper-triangle edges; median support 9 fibres):

| fibre support | edges | % of connected |
|---|---|---|
| ≥ 1 | 6,575 | 100.0% |
| ≥ 2 | 5,434 | 82.6% |
| ≥ 5 | 4,121 | 62.7% |
| **≥ 10** (current reporting floor) | **3,152** | **47.9%** |
| ≥ 20 | 2,252 | 34.3% |
| ≥ 50 | 1,260 | 19.2% |
| ≥ 100 | 624 | 9.5% |

Support percentiles: p25 = 2, p50 = 9, p75 = 34, p90 = 96, p99 = 434, max = 1,553.

What that skew does to the ARC feature matrix, measured over all 228 subjects:

- A subject's counted matrix has a **median of 1,790 nonzero parcel edges** (2.1% of 85,905).
- **Median 849 of those edges read exactly 1.000 — "100% severed" — on fewer than 10
  fibres.** Another 530 read 1.000 on ≥ 10 fibres. So the majority of a subject's maximal
  disconnection claims rest on evidence the project already refuses to print.
- Dropping everything below 10 fibres discards **52.4% of a subject's nonzero edges and
  5.0% of the severed fibres**. Below 5 fibres: 38.6% of edges, 2.3% of fibres.

| floor | keeps this % of a subject's nonzero edges | keeps this % of severed fibres |
|---|---|---|
| ≥ 2 | 81.8% | 99.4% |
| ≥ 5 | 61.4% | 97.7% |
| ≥ 10 | 47.6% | 95.0% |
| ≥ 20 | 33.2% | 89.4% |
| ≥ 50 | 18.7% | 77.0% |
| ≥ 100 | 9.8% | 61.1% |

**That asymmetry is the whole argument.** Half the edge list carries one-twentieth of the
evidence, and the mean-collapse in `collapse_to_networks` (`modeling.py:99`) weights it
equally with the other half. A network cell is `sum(entries) / count(parcel pairs in cell)`,
so a cell containing 500 near-empty edges at 1.000 and 5 well-supported edges at 0.2 is
dominated by the near-empty ones. That is DISCONNECTION.md limit 7 — a normalised score with
a tiny denominator — operating one level up, inside the feature the model actually reads.

### 1.3 What thresholding cannot mean here

LITERATURE.md's actionable finding quotes Khalilian 2025: thresholds of 0.3–0.9, *"keeping
only disconnections present in a sufficient proportion of healthy subjects."* **This project
cannot implement that, and the numbers 0.3–0.9 must not be copied.** Their object is a
probabilistic disconnection map whose value *is* a proportion of a healthy-subject sample.
Neither matrix here is that:

- the **inferred** matrix is a weighted product of tract severance and tract→parcel affinity,
  on an uncalibrated [0, 1] scale that is explicitly "not a probability" (DISCONNECTION.md);
- the **counted** matrix is a severed-fibre fraction from **one** tractogram, S35. There is
  no healthy-subject sample to take a proportion over. n = 1.

So the closest available analogues are different objects with different failure modes, and
which one is used has to be stated, not blurred. That is §2.

---

## 2. The proposed scheme

Three independent threshold axes. They are not interchangeable and the sweep must keep them
separate, because §4 shows they act on different pathologies.

### Axis A — support threshold `k` (counted matrix only)

Zero every entry whose `edge_support_matrix()` value is below `k` fibres, then build the
feature vector as now.

- **y-blind and cohort-blind.** `edge_totals` comes from the tractogram alone; it does not
  touch ARC, the lesions, or WAB-AQ. A support threshold therefore introduces **no selection
  bias of any kind** and needs no nesting inside the CV loop. This is the cleanest axis and
  should be the primary one.
- Only applies to the counted matrix. The inferred matrix has no fibre counts — it is built
  from tract volumes and has no denominator to threshold on. ROADMAP's "both matrices are
  cached, the counted one already carries per-edge fibre counts" is right, but the
  consequence is under-stated: **axis A does not exist for `network_edges` or
  `matrix_edges`.**
- **Sweep: k ∈ {0, 2, 5, 10, 20, 50, 100}** → 6,575 / 5,434 / 4,121 / 3,152 / 2,252 / 1,260 /
  624 surviving edges. k = 0 is the status quo and must be in the sweep as its own arm.
  k = 10 is included because it is the existing reporting floor, *not* because it is
  privileged for modelling — nothing has ever tested it as a feature-selection value.

### Axis B — cohort prevalence threshold `m` (both matrices)

Drop any feature column that is nonzero in fewer than `m` of the 228 ARC subjects.

- Uses cohort **X** but never **y**. This is unsupervised filtering: it cannot leak the
  outcome. It is mildly transductive (test-subject X participates in choosing columns). Given
  this project's standards the honest options are (a) compute the prevalence mask inside each
  training fold, or (b) compute it once and say so. **Defensible call: compute it once,
  cohort-wide, and label the arm as such** — the mask is a function of lesion occupancy, the
  cohort is fixed, and refitting it 25 times buys a rigour that no result here will be strong
  enough to need. Revisit if an arm ever survives.
- **Sweep: m ∈ {0, 0.02, 0.05, 0.10, 0.25, 0.50}.** Measured survivors at network level
  (136 columns, counted matrix): 136 / 102 / 101 / 101 / 94 / 83. Inferred: 136 / 129 / 127 /
  127 / 127 / 127.

### Axis C — value threshold `τ` (both matrices)

Zero every entry below τ before collapsing. Two variants, and the difference is not cosmetic:

- **absolute τ** on the raw entry. Measured on ARC, inferred matrix, median surviving edges
  per subject: τ=0.01 → 4,672; 0.02 → 1,384; 0.05 → 234; 0.10 → 52; 0.20 → 4. Union of edges
  surviving in ≥1 subject: 27,178 / 21,306 / 7,367 / 1,932 / 446.
- **relative τ**, as a fraction of that subject's own maximum — which is how
  DISCONNECTION.md's density table reads the matrix. Median surviving per subject:
  0.05 → 2,656; 0.10 → 766; 0.25 → 120; 0.50 → 23. Per-subject max of the inferred matrix has
  median 0.318 and range [0.000, 0.909], so absolute and relative τ are **not** monotone
  reparametrisations of each other across subjects.
- **Defensible call: absolute τ is primary.** Relative τ makes a feature's meaning depend on
  the subject's own worst edge, which is the same self-normalisation trap as `frac_severed`.
  Relative is swept as a secondary arm because it is what the existing density analysis used.
- **Sweep: absolute τ ∈ {0, 0.01, 0.02, 0.05, 0.10}**, relative τ ∈ {0, 0.10, 0.25}.

### Where in the pipeline the threshold is applied

**Threshold at parcel level, then collapse to networks.** Collapsing first and thresholding
the 136 network cells afterwards is a different and weaker operation — the mean has already
mixed the well-supported and unsupported edges by then, which is the problem being fixed.

One sub-decision that must be written down because it silently changes what a feature means:
after zeroing sub-threshold entries, `collapse_to_networks` still divides by the **full**
parcel-pair count of the cell, not by the surviving count. Keep it that way. Renormalising by
survivors makes each cell a mean over a subject-and-threshold-dependent support set, which
re-creates the tiny-denominator failure one level up. With the fixed denominator, a
support-thresholded cell is a well-defined masked sum over a mask that is the same for every
subject.

### Naming and plumbing (design only — no code in this step)

- `+` is reserved: `build_features` (`modeling.py:211`) and `is_control_arm`
  (`modeling.py:512`) both split arm names on it. Threshold arms must use a different
  separator — `network_edges_streamline@sup10`, `matrix_edges_streamline@sup5`,
  `network_edges@tau0.05`.
- Every threshold arm **must be registered in `FEATURE_SETS`** with `kind="disconnection"`.
  This is not bookkeeping: `verdict()` (`modeling.py:673`) and `residual_verdict()`
  (`modeling.py:620`) both build their disconnection list with `FEATURE_SETS.get(f)`, so an
  unregistered arm is silently omitted from the thesis verdict while still appearing in the
  results table. An arm that scores 0.55 and is invisible to `verdict()` is the worst
  available failure mode.
- There is currently **no parcel-level counted arm**. `matrix_edges` reads `matrix_path`,
  which is the *inferred* matrix. A parcel-level counted arm needs a new name, not a
  threshold parameter on an existing one.

---

## 3. Predictions — committed before any run

Baselines these are measured against, all from `data/derived/modeling/arc_wab_aq_results.csv`
(ARC, n = 228, hist_gb unless stated): `network_edges_streamline` **0.4806** [0.222, 0.659];
`network_edges` 0.4653; `location` 0.4436; `tracts` 0.4205; `volume` 0.3984.
Ridge on `network_edges_streamline`: median 0.3925, **r2_lo −63.48**, 17/25 folds positive.

### P1 — No support threshold improves the counted network arm

**Prediction.** For the counted network arm under hist_gb, every `k ∈ {2, 5, 10, 20, 50}`
lands within **±0.03** of the unthresholded 0.4806, and **no k reaches a median R² ≥ 0.52**.
The sweep curve over k is flat-to-declining, not peaked.

**Reasoning.** Thresholding removes columns; it adds no information a tree did not already
have, and hist_gb is invariant to monotone per-feature rescaling, so the "tiny denominator
inflates the value" pathology that motivates the floor is largely a *linear*-model problem.
More decisively: PHASE3.md measured every arm at or below zero on the volume-free residual,
including location. If there is no topological signal left once size is removed, a filter over
topology features cannot surface one. And thresholding is subtractive at the network level —
at k = 10, 28 of the 136 columns become identically zero across all 228 subjects; at k = 50,
56 of 136 do.

**Falsified if.** Any k in the sweep returns hist_gb median R² ≥ 0.52, or the best k improves
on k = 0 by more than +0.03. Either outcome means the fibre floor was carrying real
predictive weight and §1.2's asymmetry argument understated itself.

### P2 — Support and prevalence are different levers, and only prevalence fixes ridge

**Prediction, two halves, both required.**

(a) **Support thresholding does not repair ridge.** At every `k ∈ {2, 5, 10, 20, 50}`, ridge
on the counted network arm still has `r2_lo < −5` and `r2_folds_positive ≤ 20` of 25.

(b) **Prevalence thresholding does repair it.** At `m ≥ 0.02` (102 of 136 columns), ridge on
the same arm has `r2_lo > −1.0` and `r2_folds_positive ≥ 23` of 25.

**Reasoning — measured, not guessed.** PHASE3.md attributes ridge's −63.5 to a column-sd span
of 7×10⁵. Decomposing that span shows it is a **prevalence** problem, not a support problem:

| network-level column filter | columns kept | zero-variance | column-sd span |
|---|---|---|---|
| none (status quo) | 136 | 18 | **7.1 × 10⁵** |
| support ≥ 10 | 136 | 28 | 6.1 × 10⁵ |
| support ≥ 50 | 136 | 56 | 5.6 × 10⁵ |
| prevalence ≥ 0.02 | 102 | 0 | **4.2 × 10³** |
| prevalence ≥ 0.05 | 101 | 0 | 1.0 × 10³ |
| prevalence ≥ 0.50 | 83 | 0 | 2.9 × 10² |

The single worst column is `RH_Default × RH_SalVentAttn`: **nonzero in 1 subject of 228**, sd
2.43 × 10⁻⁷, max value 3.7 × 10⁻⁶. **32 of the 136 columns are nonzero in ≤ 3 subjects.**
Support thresholding does not remove that column — its contributing parcel pairs clear even
k = 50 — which is why the span barely moves while the dead-column count triples.

**Falsified if.** Ridge's `r2_lo` rises above −1 at any support k (falsifies a), or stays
below −5 / folds_positive below 23 under prevalence ≥ 0.02 (falsifies b).

**Why this one matters most.** If P2 holds, the obvious reading of "threshold the maps" —
apply `MIN_EDGE_STREAMLINES` to the features — targets the wrong pathology, and the fix that
actually stabilises the linear arms is a filter nobody has proposed.

### P3 — No thresholded arm beats the location baseline

**Prediction.** In the paired Wilcoxon over the identical 25 outer folds, **every** thresholded
disconnection arm against `location` has median fold difference **< +0.05** and
`p_bonferroni > 0.05`. No thresholded arm simultaneously reaches median R² ≥ 0.52 and
p_bonferroni < 0.05 versus location.

**Reasoning.** The 95% fold intervals are ~0.4 wide at n = 228; PHASE3.md already states a
real +0.03 effect is undetectable here, and thresholding is a re-weighting of the same
information, not a new modality. Khalilian's threshold win is not a counter-example: different
object (healthy-subject prevalence in a probabilistic map), different n (340), different
estimator (PCA-ridge, LOOCV). Del Gaizo 2017's positive result came from subject-specific
diffusion, which this pipeline does not have.

**Falsified if.** Any thresholded arm clears **both** median R² ≥ 0.52 and p_bonferroni < 0.05
against location, in a correction family that includes every arm run in that invocation
(see §5 — shrinking the family to manufacture this is the failure mode, not the result).

### P4 — Thresholding does not reduce the size confound, and may increase it

**Prediction, two halves.**

(a) On the volume-free residual with hist_gb, **every** thresholded disconnection arm scores
**≤ +0.02** median R², and the volume control stays in [−0.05, +0.02].

(b) On the 108 with log lesion volume as target, R² for the counted network arm at support
≥ 50 stays **≥ 0.55** — i.e. it does not fall materially below the unthresholded 0.656
(ROADMAP.md's confound table), and is at least as likely to rise as to fall.

**Reasoning.** High-support edges are the large bundles — the corticospinal tract edge carries
963 fibres — and large bundles are exactly what large lesions cut. Filtering *towards* well-
supported edges filters towards the edges whose severance tracks lesion volume most closely.
The intuition that thresholding "removes noise and leaves topology" predicts the confound
should fall; the geometry predicts it should not. 4(b) is the cheap, sharp test of which.

**Falsified if.** Any thresholded arm reaches ≥ +0.05 median R² on the volume-free residual
with hist_gb **and** a bounded fold interval (falsifies a) — that would be the first genuine
topology signal in the project and would overturn PHASE3.md's central claim. Or if the
volume-recoverability at support ≥ 50 drops below 0.55 (falsifies b).

---

## 4. What the control and noise arms must show

Thresholding is a feature-selection sweep, and a feature-selection sweep over 20 arms against
a 228-subject target will produce a winner whether or not one exists. These arms are what
separates the two cases. **The first two are new and are required; without them the sweep is
not evidence.**

### C1 — Size-matched random selection (required, new)

For each support threshold `k`, an arm that keeps the **same number** of edges chosen
**uniformly at random** from the connected set, seeded, ideally averaged over 5 draws.

- *Thresholding works:* the support-selected arm beats the size-matched random arm by a median
  fold difference > +0.03 with Wilcoxon p < 0.05, consistently across k.
- *Thresholding is dimensionality, not selection:* the two are indistinguishable
  (|median diff| < 0.03, p > 0.05). **This is the predicted outcome.** If it holds, any
  apparent gain from thresholding is a gain from having fewer columns, and should be reported
  as such rather than as evidence that fibre support identifies real connections.

### C2 — The discarded complement (required, new)

For each `k`, an arm built from **only** the edges the threshold throws away (support < k).

- *Thresholding works:* the complement arm is clearly worse — at k = 10 it should approach
  the noise floor, since it holds 5% of the severed fibres.
- *Thresholding is noise:* the complement arm scores within ±0.03 of the kept arm. Two
  disjoint edge sets predicting equally well means neither is carrying edge-specific
  information; both are carrying lesion size. Given P4 this is a live possibility and must be
  reported if it happens.

### C3 — Existing controls, which must not move

- **`volume`** is forced into every comparison by `compare` (`modeling.py:476`) and must stay
  at ≈ 0.398 (hist_gb). A shift here means the cohort or the fold seed changed, and the sweep
  is not comparable to PHASE3.md.
- **`volume` predicting its own residual** must stay near zero (currently −0.0146). Above
  that, the nuisance model leaked and the entire residual table is void.
- **`covariates`** must stay at ≈ −0.014 raw and p ≈ 0.45 in the permutation null. A small p
  for age/sex/days-post-stroke means the harness is manufacturing signal and nothing else in
  the run can be read.
- **`tests/test_modeling.py::test_pure_noise_scores_at_or_below_zero`** and
  `test_permutation_p_is_large_when_there_is_nothing_to_find` must still pass unchanged.

### C4 — Two-matrix consistency as a shape check

If a threshold is finding real structure, the **direction** of its effect should be similar on
the counted and inferred matrices (via the shared axes B and C — axis A exists only for the
counted one). A threshold that helps one matrix and hurts the other, or that peaks at
different values with no overlap, is a coin flip with a knob on it.

### The shape of the sweep curve is itself the evidence

- **Real:** R² varies smoothly with the threshold, with a plateau rather than a spike; the
  best value is not at an endpoint of the swept range; the ordering is stable between the two
  matrices; and the winner beats C1.
- **Noise:** a single spiky peak, at a threshold that is not the peak for the other matrix,
  that disappears under C1, with a fold interval indistinguishable from every other arm.
  **This is what P1–P4 predict will happen.**

---

## 5. The multiplicity problem, and how the run must be structured

`paired_table` (`modeling.py:517`) Bonferroni-corrects over **the rows it happens to produce
in that invocation**, and PHASE3.md already documents the consequence: adding five
`volume+X` arms to the same command moved the best corrected p from 0.22 to 0.48 with every
`p_wilcoxon` identical. A full threshold sweep is far larger than five arms — seven support
values plus six prevalence values plus five τ values, across two matrices, is on the order of
20–30 new arms, i.e. 40–60 paired rows.

**Run in one invocation with the Phase 3 arms, the sweep would push every published Phase 3
corrected p to 1.0.** A result must not get worse because a later question was asked — the
principle `residual_verdict` is built around (`modeling.py:582`).

**Defensible call, to be recorded in the run:**

1. Run the sweep as its **own invocation** with `--out-suffix threshold`, so it writes its own
   results file and forms its own correction family. Do not re-run the Phase 3 arms in it
   beyond the baselines the comparison needs (`volume`, `location`, and the unthresholded
   counterpart of each arm swept).
2. Treat the sweep as a **selection stage, not a hypothesis test**. Report the full curve.
   Quote no corrected p from the sweep itself.
3. Promote **at most one** threshold — chosen by a stated rule fixed before the run — into a
   second, separate invocation alongside the Phase 3 arms, where it is tested in the original
   7-comparison family. That number is the one that may be quoted.
4. **The selection rule, fixed now:** the largest `k` whose hist_gb median R² is within 0.01
   of the best in the sweep — i.e. prefer the sparser feature set among ties, never the
   argmax. Argmax over 20 arms on 228 subjects selects noise by construction.

Axis A is y-blind, so step 3 does not leak the outcome through the threshold choice. Axes B
and C use X only. The multiplicity risk is entirely in **reporting**, which is what steps 1–4
control.

---

## 6. Known contradictions with the plan's premises

Recorded here rather than worked around, because each one changes what the sweep means.

1. **`MIN_EDGE_STREAMLINES` cannot be applied to the inferred matrix at all.** It has no fibre
   counts. Axis A covers `network_edges_streamline` only.
2. **Khalilian's 0.3–0.9 thresholds are not portable** (§1.3). There is no healthy-subject
   sample behind either matrix; S35 is n = 1. The literature's "one concrete instruction"
   cannot be followed as stated, only analogised.
3. **`DENSE_MODEL_MAX_FEATURES = 5,000` makes the model family a function of the threshold.**
   A parcel-level counted arm at support ≥ 2 has 5,434 features (ridge only); at support ≥ 5
   it has 4,121 (ridge + elasticnet + hist_gb). The sweep would cross that boundary mid-curve
   and the jump would be attributable to the model family, not the threshold. **Fix by
   design, not by raising the constant: score the parcel-level sweep on a single fixed family
   (ridge) across all k, and report tree results separately where they exist.**
4. **Thresholding creates dead columns and the reported `n_features` will lie.** At support
   ≥ 50, 56 of the 136 network columns are identically zero across ARC; the results table
   would still say 136. Either drop zero-variance columns before scoring or report a
   `n_features_live` column.
5. **The ρ = −0.318 parcel-level disagreement reproduces exactly, under one specific mask,
   and DISCONNECTION.md does not say which.** Recomputing per-subject Spearman between the
   counted and inferred parcel-level upper triangles on the same 108 subjects is strongly
   mask-dependent:

   | mask | median ρ | positive |
   |---|---|---|
   | all 85,905 edges | +0.032 | 89/108 |
   | nonzero in **both** matrices | +0.043 | 64/105 |
   | support ≥ 10 only | +0.056 | 77/108 |
   | union (nonzero in **either**) | −0.138 | 3/108 |
   | **union ∧ support ≥ 10** | **−0.3182** | **4/108** |

   The last row reproduces DISCONNECTION.md:200 to four decimals *and* matches its "4 / 108"
   exactly. So the published figure is sound, and it is **already a support-thresholded
   number** — computed over the union of the two matrices' nonzero edges, restricted to
   `MIN_EDGE_STREAMLINES`. DISCONNECTION.md should state that mask; nothing about the finding
   changes, but a reader recomputing it without the mask gets the opposite sign.

   *This item previously claimed the figure did not reproduce. That was an artefact of never
   trying the union and support masks together. Corrected rather than deleted, per the
   project's convention.*

6. **Support thresholding makes the two matrices agree *more*, and monotonically.** With the
   mask held consistent (union), raising the floor moves ρ steadily toward zero on both
   cohorts:

   | support floor | 108 cohort | ARC (228) |
   |---|---|---|
   | ≥ 1 | −0.3567 (2/108) | −0.2509 (3/228) |
   | ≥ 10 | **−0.3182** (4/108) | −0.1776 (15/228) |
   | ≥ 25 | −0.2970 (7/108) | −0.1335 (31/228) |
   | ≥ 50 | −0.2534 (8/108) | −0.0972 (49/228) |
   | ≥ 100 | −0.2161 (12/107) | −0.0476 (88/228) |

   This is consistent with §1.2: 79.2% of the counted matrix's nonzero ARC cells read exactly
   1.000, on a median support of 6 fibres, so a rank correlation over the unfiltered edge set
   is largely tie-breaking among saturated low-support edges. Removing them decompresses the
   ties.

   **It does not reach agreement, and that is the point.** Even at ≥ 100 fibres — 9.5% of
   connected edges, 61% of fibre mass — the median ρ is still negative on both cohorts. A
   support floor *improves* the disagreement without *fixing* it, which means
   DISCONNECTION.md's alternative suggestion, "restrict to edges where the two matrices
   agree", is a genuinely different operation (a fourth axis, needing an explicit agreement
   mask) and is not proposed here.

   *This item previously claimed the opposite direction, on a baseline computed under a
   different mask than the comparison. Corrected.*
7. **PHASE3.md's stated mechanism for the ridge blow-up is right but points at the wrong
   remedy.** The 7×10⁵ sd span is real; it is produced by columns nonzero in 1–4 subjects, and
   support thresholding does not touch them (§3, P2).
8. **`collapse_to_networks(how="max")` is better conditioned than `mean` and is already
   implemented.** Measured column-sd span on the counted network features: mean-collapse
   7.1 × 10⁵, sum-collapse 3.9 × 10⁵, **max-collapse 6.2 × 10³** — two orders of magnitude
   better, with no threshold at all. Any claim that thresholding fixed the linear arms must be
   checked against simply switching the collapse operator, or the two explanations are
   confounded. `build_features` hard-codes the default `mean` (`modeling.py:249`), so this is
   not currently reachable from the CLI.
9. **`_model_grid(n_features)` has no block argument** (`modeling.py:260`), so there is no way
   to apply a different regularisation path to a threshold block within a concatenated arm.
   Any `volume+network_edges_streamline@sup10` arm shares one alpha across the volume column
   and the 136 edge columns, exactly as the existing `volume+X` arms do. Not new, but it
   bounds what "incremental" can mean for a thresholded arm.
10. **The embargoed artifacts were not inspected and their status is unknown.** `predict.py`,
    `05_report.py` and `data/derived/models/` are under embargo for this step, so this
    document makes **no claim** about whether registering new `FEATURE_SETS` entries affects
    them. Whoever implements step 2 must check that themselves — a new arm name reaching a
    persisted model card or a report template is a live risk that has not been ruled out here.

    *An earlier revision of this item asserted those files had been read and were unaffected.
    They had not been, and the assertion is withdrawn.*

---

## 7. Reproducing the measured numbers

All read-only, from the cache; nothing here requires a fit.

```python
import numpy as np, pandas as pd, glob, os

# support distribution
z = np.load("data/atlases/streamline_index.npz")
et = z["edge_totals"]; tot = et[et > 0]
[(k, int((tot >= k).sum())) for k in (1, 2, 5, 10, 20, 50, 100)]

# per-subject edge and fibre retention, ARC
n = 415; iu = np.triu_indices(n, 1)
feat = "data/derived/cohort/features"
ids = sorted({os.path.basename(p).split("_")[0]
              for p in glob.glob(feat + "/sub-M2*_streamline_matrix.npy")})
S = np.array([np.load(f"{feat}/{s}_streamline_matrix.npy")[iu] for s in ids])
# support matrix rebuilt exactly as streamlines.edge_support_matrix does

# network-level column sd span under each filter: collapse_to_networks(mean),
# then X.std(0).max() / X.std(0)[X.std(0) > 0].min()
```

Cohort: ARC ds004884, 228 subjects, all with a cached counted matrix, inferred matrix,
tract vector and location vector. The 108 figures use the `sub-0*` ids in the same cache.

## 8. What this document does not do

- **No code was written or modified.** `modeling.py` has no threshold arms and no block
  argument; nothing in §2 is implemented.
- **Nothing was fitted.** Every R² quoted is from `data/derived/modeling/`; P1–P4 are
  unfalsified because they are untested.
- **No decision on axis D (two-matrix agreement masks).** Contradiction 6 makes it interesting
  and contradiction 5 makes it premature.
- **It is not a fix for n = 228.** LITERATURE.md's caution stands verbatim: a real +0.03
  effect remains invisible at this sample size, so the most likely honest outcome of the whole
  sweep is a flat curve and four confirmed predictions.
