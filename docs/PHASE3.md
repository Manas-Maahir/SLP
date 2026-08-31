# Phase 3 — does disconnection predict outcome better than location?

The go/no-go question for the whole project, run on **ARC**: 228 subjects with both a lesion
mask and a WAB Aphasia Quotient, registered to MNI by `src/lesionconn/registration.py`.

`src/lesionconn/modeling.py`, `scripts/19_train.py`. Registration: [REGISTRATION.md](REGISTRATION.md).

## The answer

**B ≈ A.** Disconnection is not distinguishable from the location baseline. That is the
roadmap's middle branch — *dig into representation before declaring anything* — and it is
neither the win nor the failure.

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

**The consequence is a real gap:** the permutation null validates the *linear* arms and not
the boosted-tree arms that produce the headline numbers. It is not closed.

## What this means for Phase 5

- **Coarser, not finer.** The 415×415 parcel edges were already known not to replicate between
  the two matrices (median Spearman −0.318, [DISCONNECTION.md](DISCONNECTION.md)). Network
  edges are the resolution that survives, and they still do not beat location.
- **Volume must be regressed out, not merely reported.** A feature set that is two-thirds
  lesion size cannot demonstrate topology.
- **n = 228 is the binding constraint.** With 95% fold intervals ~0.4 wide, a real +0.03 effect
  is undetectable here. ARC has 902 sessions and this uses 228 — more subjects is a cheaper
  path to an answer than a cleverer model.
- **A permutation null on the tree models** would close the gap above.

## What was NOT done

- No enantiomorphic normalisation; cost-function masking only.
- No comparison of `SyNRA` against other transforms — Dice 0.924 was good enough to proceed,
  which is not evidence nothing else is better.
- No correction for age, sex or time post-stroke, all of which are in `participants.tsv` and
  all of which plausibly matter for WAB-AQ.
- No hold-out test set. Every number here is cross-validated on the same 228 subjects.
