# Presentation script — Lesion → Disconnection → Outcome

A speaking script for explaining this project to someone who has never seen it.
Roughly 15 minutes plus questions. Stage directions are in `>` blockquotes.

**The one-line frame, and the thing to hold onto for the whole talk:**

> I built the pipeline, I tested the thesis honestly, and the thesis lost.
> Here is how I know — and here is the machinery that caught six of my own
> errors on the way there.

Do not try to sell disconnection as a win. The point estimates rank it first,
and this project proved that ranking is a coin flip. The negative result, with
this much instrumentation behind it, is the stronger talk and it is much harder
to attack in questions.

---

## 0 — Setup (30 seconds)

> No slide yet. Just say it.

"A stroke destroys a piece of brain tissue. The obvious question a clinician
asks is *where* — which region died. But a brain region is only useful because
it is connected to other regions, and a lesion also cuts the white-matter
cables running underneath it. So there is a competing hypothesis: what predicts
a patient's deficit is not *where* the damage is, but *which connections it
severs*.

That is a testable claim. This project tests it."

---

## 1 — The question (1 minute)

> Slide: the question, and the two feature sets side by side.

"The outcome I predict is the **Western Aphasia Battery Aphasia Quotient** — a
0 to 100 language score. The cohort is **ARC**, a public dataset: 245
participants, 228 of whom have both a lesion mask and a WAB score. Mean 63,
standard deviation 28, full range 5 to 100.

Two models compete on identical data.

**Model A, the baseline — location.** A 463-region atlas. How much of each
region does this lesion destroy?

**Model B, the thesis — disconnection.** Which of 80 named white-matter tracts
does it sever, and which network-level connections does it cut?

If the thesis is right, B beats A. That is the whole project's go/no-go, and
I deliberately put it first — before building any polished infrastructure —
because if it fails, no amount of good engineering saves the idea."

---

## 2 — It is end-to-end and it is real (2 minutes)

> Slide: the pipeline diagram. Mask → registration → disconnection → report →
> prediction. Or just show the four output figures.

"This is not a notebook. A lesion mask goes in one end and a readable report
comes out the other, with a 415×415 disconnection matrix, an 80-tract severance
profile, four figures, and a predicted WAB score.

Two datasets feed it. 108 real, de-identified lesions that arrive already in
MNI space. And the 228 ARC subjects — whose masks are drawn in **native scanner
space**, on five different per-subject grids, none of them MNI-aligned. I
verified that from the file headers, not the filenames.

So ARC needed registration, and that is the part worth pausing on."

### Registration — the detail that shows the work

"T2-weighted scan to the FSL MNI152 template, using ANTs. Two things mattered.

First, this is **cross-modality** — a T2 scan against a T1 template — so both
stages use mutual information, not correlation.

Second, and this is the one people get wrong: **cost-function masking on every
stage.** ANTsPy's default masks only the nonlinear stage, which leaves the
affine free to pull healthy tissue into the hole where the lesion is. The
lesion would distort its own warp. I forced it on for all stages.

Result: median whole-brain Dice **0.924**, worst subject 0.901.

And here is why I ran a pilot of 8 and *looked at the images* before the batch.
Every lesion shrank by about **30%** entering MNI space. That looks like a bug.
It is not. I pulled the affine determinant out of the transform: 1.514, which
implies a moving-to-fixed volume scale of 0.660. The measured lesion ratio was
0.704. The lesion is scaling by essentially the same factor as the whole brain.
The compression is uniform and correct — and I only know that because I
measured it against the transform that produced it instead of assuming."

---

## 3 — The answer (3 minutes) ★ the core of the talk

> Slide: the results table. Leave it up for a while.

"Here is the comparison. Nested cross-validation, 25 outer folds, every arm
scored on the **identical** folds with a fixed seed, so this is a paired
comparison — not seven separate experiments.

| arm | R² | 95% interval across folds |
|---|---|---|
| network edges (counted matrix) | **0.481** | [0.222, 0.659] |
| network edges (inferred matrix) | 0.465 | [0.240, 0.593] |
| **location — Model A** | 0.444 | [0.299, 0.625] |
| **tract severance — Model B** | 0.421 | [0.266, 0.606] |
| **lesion volume alone** | **0.398** | [0.200, 0.623] |

Read that table on its own and disconnection wins. My code originally said
exactly that. **It was wrong**, and I want to walk through why, because it is
the most important thing I learned."

> Beat. Point at the interval column.

"Look at the intervals. Every single arm spans roughly 0.2 to 0.65. Ranking
the medians of distributions that overlap that much will confidently call a
coin flip a finding.

So I ran the paired test the folds entitle me to — Wilcoxon signed-rank,
Bonferroni corrected."

> Slide: the paired table.

| comparison | median diff | folds better | p | corrected |
|---|---|---|---|---|
| network edges vs volume | +0.055 | 16/25 | 0.032 | 0.22 |
| location vs volume | +0.026 | 16/25 | 0.134 | 0.94 |
| tracts vs volume | +0.012 | 14/25 | 0.252 | 1.00 |
| network edges vs location | +0.011 | 14/25 | 0.325 | 1.00 |
| **network edges (inferred) vs location** | **−0.022** | 11/25 | 0.634 | 1.00 |

"**Nothing beats location. Nothing survives correction.** The single nominally
significant result is against *lesion volume*, not against the baseline the
thesis is actually about — and it dies at p = 0.22 corrected.

The answer is **B ≈ A**. Disconnection is not distinguishable from location.

I then changed the harness so it runs this test itself and prints it last, so
it can never again report an ordering of overlapping medians as a result."

> That last sentence is the most credible thing in the talk. Say it slowly.

---

## 4 — The number that should worry you most (2 minutes)

> Slide: just the one number, large. `0.398`

"**Lesion volume alone — one single column, how big is the hole — reaches
R² 0.398.** Every other arm in that table is fighting over the remaining 0.05.

I predicted this before the run. On the other cohort I measured how much of
each feature set is *recoverable lesion size*: 47% for tract severance, **66%
for the network edges**. Two-thirds of my best feature set is a proxy for how
big the lesion is.

So beating location is not enough. The honest question is incremental —
does `volume + disconnection` beat `volume` alone?

The weak form of that test was inconclusive. The sharp form is to remove volume
from the **target** and see if anything is left."

> Slide: the residual table.

"I fit a nuisance model on log lesion volume **inside each training split**,
subtract its prediction from both sides, and score on the residual. Note
*inside each split* — fitting it once over all 228 subjects, which is the
obvious implementation, leaks the test outcomes into the residual and inflates
every arm at once. So it is refit 25 times.

| arm | R² on the volume-free residual |
|---|---|
| location | −0.005 |
| tracts | −0.011 |
| *volume — the control* | *−0.015* |
| network edges | −0.026 |

**Every arm is at or below zero. So is every one of the 18 model-by-feature-set
cells.** Not one combination predicts the residual better than its own mean.

Read the control row first: volume predicting its own residual comes back at
−0.015, which is what a correctly removed confound looks like. If that row were
appreciably above zero, the nuisance model had leaked and the whole table would
be void.

What this licenses: the 0.05 separating those arms is **not** topology
information sitting on top of lesion size. When size comes out of the target,
there is nothing left for any arm to find — *including the location baseline*.

What it does **not** license: 'there is no signal.' R² ≤ 0 at n = 228 is a
statement about **detectability**, not about existence.

The honest summary is: on this cohort, at this n, **lesion volume is the model,
and everything else is decoration.**"

---

## 5 — A finding nobody asked for (2 minutes)

> Slide: the two matrices, and the two correlation numbers.

"One more result, and this one is independent of whether the thesis holds.

I built the disconnection matrix **twice**, from two completely independent
sources. Once by inferring which regions a tract connects from which regions it
touches. And once by counting for real — taking a tractogram of **479,457
streamlines**, assigning both ends of every fibre to a region, and counting how
many of each connection's fibres pass through the lesion.

Two methods, same input, same output shape. Here is how much they agree, across
all 108 lesions:

| level | median Spearman ρ | subjects positive |
|---|---|---|
| individual region pairs (415 nodes) | **−0.318** | 4 / 108 |
| network level (17 systems) | **+0.331** | 104 / 107 |

**The sign flips.** At network level they agree. At the level of individual
connections they *anti-correlate*.

I tested and rejected three explanations before accepting the fourth —
normalisation, fraction-versus-count, and a laterality bug. What survives is
granularity: both methods name the same anatomy and then pick different
*adjacent* parcels within it. On a test lesion, both say motor cortex to
brainstem; one picks parcel 35, the other picks parcel 37.

The consequence is a real constraint on anyone doing this work: **a single
region-pair edge is not a reliable feature.** Coarsen to networks, or restrict
to edges where two independent methods agree. That holds whether or not my
thesis does."

---

## 6 — What the guardrails caught (2 minutes)

> Slide: pick two of these. They land every time.

"The reason I trust the negative result is that the same instrumentation caught
a series of errors that would each have produced a confident, plausible, wrong
answer.

**One. Two atlases, same space, opposite storage order.** Harvard-Oxford ships
RAS-ordered; the Schaefer parcellation ships LAS. Identical world coordinates,
identical 2mm voxels, x-axis stored in reverse. Merging the raw arrays — the
obvious implementation — mirrors every subcortical structure. A **left**
putamen lesion comes back labelled **right** putamen. No shape check, no voxel
count, no volume check catches that. It is now asserted in code, and there is a
test that shows the assertion failing on a deliberately mirrored atlas.

**Two. 'MNI space' is not one space.** The template nilearn hands you by
default is 99×117×95. The one the atlases ship on is 91×109×91. Both are
legitimately called MNI152. Intersect a lesion with a misaligned parcellation
and you get a disconnection matrix that looks entirely plausible and is quietly
wrong. Every function that combines two volumes now asserts they are on the
same grid first.

**Three, and this is the uncomfortable one. The de-identification check.** The
108 lesions arrived with patient names and apparent medical record numbers in
the filenames. So there is a test that greps every artifact this project emits
— source, docs, CSVs, JSON, image headers, compressed PNG text chunks — for
every identifying token. It also scans this project's own source code. **It has
caught a real patient name three times. Twice it was in a code comment written
by the person who wrote the check.**

**Four — reading the output, not the tests.** The anatomy section of my report
once described a lesion that was 70% cerebellum using only the 18% of it that
clipped occipital cortex — because the node set has no cerebellar labels, so it
saw a minority of the lesion and presented it as a confident table. That would
have passed every test written from the spec. It only failed when a human read
one real report."

---

## 7 — Live demo (3 minutes)

> Two commands. Both verified working. Have them already typed.

```bash
cd scripts
../.venv/Scripts/python.exe 05_report.py ../data/derived/arc/lesions/sub-M2001_lesion.nii.gz
```

"Real ARC subject. Mask in. Out comes anatomy, severed tracts, the matrix, four
figures — and this."

```
features                      predicted     90% interval   CV R2    MAE
volume                             41.1          [6, 72]   0.424   17.2
location                           65.3         [32, 97]   0.454   16.3
network_edges_streamline           60.1         [31, 93]   0.519   15.0
```

> Now say the honest part before anyone else spots it.

"Three things about this, in order of how much they matter.

**All three arms are always printed.** Never just the disconnection number. If
I printed that alone I would be presenting a coin flip as a capability. The
report says so in its own text.

**The three arms disagree by 24 points** on this one lesion, on a 0–100 scale.
That is section four made visceral.

**And this subject's true score is 87.1** — and this subject is *in the
training set*, because the final model is fit on all 228. So all three arms
miss an in-sample subject by 22 to 46 points. That is not a failed demo. That
is exactly what 'n = 228 is the binding constraint' looks like when you stop
reading tables and look at one patient."

```bash
../.venv/Scripts/python.exe 05_report.py sub-0035
```

```
NO PREDICTION -- this lesion is outside the training distribution:
  - only 0% of this lesion is in the LEFT hemisphere. The model was trained
    on ARC, which is 228/228 left-dominant aphasia. WAB-AQ is a language
    score and this lesion is not in the territory it was learned from.

A number is deliberately not shown. The model would return one, and it
would be meaningless.
```

> This is the closing beat of the demo. Let it sit.

"This lesion is entirely in the right hemisphere. The training cohort is
228 out of 228 left-hemisphere aphasia patients, and WAB is a language score.
The model *would* happily return a number. It would be meaningless. So the
applicability check is **enforced, not warned about** — and 67 of my own 108
lesions get refused by it."

### Optional: the viewer hand-off

> Only if it is already running. Rehearse it, or cut it.

"The disconnection output also exports into an existing neuroimaging viewer —
writing the exact directory layout that viewer's own upload endpoint produces,
so pointing it at this bundle loads the lesions with **zero changes to its
code**. Two projects, one shared space contract, no coupling.

Caveat I will state rather than hide: only the file-drop path is proven. The
programmatic upload client has been tested against a stub of the documented
contract, never against a live backend. Until it is, it should be considered
unproven."

---

## 8 — Close (1 minute)

"So. The thesis did not hold. Disconnection is not distinguishable from
location on this cohort, and once lesion volume is removed properly, neither of
them is distinguishable from noise.

What exists anyway is a pipeline that takes a raw native-space scan to a
registered, quality-controlled, de-identified disconnection profile and a
report a human can read — plus a measured constraint on how finely anyone
should trust these matrices, plus a harness that tests itself against pure
noise and refuses to answer questions outside what it was trained on.

The most useful thing I can say is this: the version of this project that
reports 'disconnection beats location, R² 0.48' exists. It is one Wilcoxon test
away from the version I am showing you. **The difference between them is not
data. It is whether you run the test that can embarrass you.**"

---

## Anticipated questions

**"Why not just use more subjects?"**
228 is the *entire* labelled cohort, not a sample of it. I want to be careful
here because I got this wrong once myself and wrote the correction into the
docs: the dataset advertises 902, but that is scanning **sessions**, not
people. The participant table has 245 rows, 228 have a lesion mask, and the
remaining 17 have a WAB score and no mask at all — so they cannot enter a
lesion-symptom model on any terms. More subjects means a *different* cohort.
And the large dataset — 1,453 scans — is the one that withholds outcome scores
for privacy. That is the bind the whole field is in.

**"Did you control for age, sex, time since stroke?"**
Yes, and they are complete for 228 of 228, so it cost no new data. Alone they
score R² **−0.014** — indistinguishable from noise. Added to lesion volume they
make it *worse*, 0.387 against 0.398. Time post-stroke has a partial
correlation of **−0.002** given volume: not weak, absent. This is a *chronic*
cohort at a median 721 days, so recovery has plateaued; that variable would
matter in a cohort scanned in the first weeks and this is not one. And the
apparent sex effect — nominally significant at p = 0.008 — dies under the
volume control, because women in this cohort have smaller lesions. Reporting
that univariate p would have been a lesion-size finding dressed up as a
demographic one.

**"How do I know your cross-validation harness isn't manufacturing signal?"**
Two controls, both tests. **Positive:** the volume arm predicting log volume
returns R² = 1.0000 exactly, so the machinery is wired correctly end to end.
**Negative:** pure noise must score R² ≤ 0 and must not reach a small
permutation p. A harness that invents signal would have produced a thesis
result out of nothing, so it is tested against itself.

**"What about a permutation null?"**
Honest answer: it validated the *linear* arms and not the tree arms that
produce the headline numbers, and that was a real gap. The reason is that ridge
is unusable on these features — its fold interval reaches **−63.5**, because the
column standard deviations across the 136 network edges span a factor of
**7×10⁵**, so a near-constant column explodes when a test subject is nonzero.
I am closing it on the tree models now: volume over 200 permutations gives an
observed 0.416 against a null median of −0.10, **p = 0.005**. The remaining
arms are compute time, not a design gap.

**"Do you have a held-out test set?"**
No, and it is the one gap I have not closed. Every number here is
cross-validated on the same 228 subjects. It is undone because n = 228 makes it
expensive rather than merely tedious — holding out 20% costs the comparison
statistical power it already does not have. I would rather say that plainly
than hold out 45 subjects and report a single noisy number as validation.

**"Isn't a normative connectome a big assumption?"**
The biggest one in the project, and it is stated in the report header, its
footer, the JSON output, and the caption of every figure. Disconnection is
inferred by intersecting a lesion with a tractography atlas built from
**healthy** brains. No diffusion imaging of the patient is used. It describes
average wiring, not this individual's — not their pre-existing anomalies and
not their post-stroke reorganisation. Every number is a lower bound: a small
lesion at a bundle's narrow waist can disconnect it almost entirely while
scoring low.

**"So was this a failure?"**
The thesis lost. The project answered its own question, which is what it was
built to do, and it produced two things that outlive the answer: a measured
constraint on the resolution at which these matrices can be trusted, and a
demonstration that lesion volume is a far stronger baseline than this
literature usually admits. The failure mode I avoided was shipping the version
with the 0.48 in the headline.

---

## Pre-flight checklist

- [ ] `git status` is clean, or do not show the repo — there is uncommitted work.
- [ ] Both demo commands run once beforehand, so figures are cached and fast.
- [ ] Know that `sub-M2001` true WAB-AQ is **87.1** and that it is in-sample.
- [ ] If showing the viewer hand-off: the export bundle is written and the
      viewer is already running. Otherwise cut it and show the four figures.
- [ ] Have the four output PNGs open in a folder as a fallback if anything
      fails live.
