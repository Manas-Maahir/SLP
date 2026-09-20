# ROADMAP.md

> **Update, 2026-08-31.** Three of the four gaps `docs/PHASE3.md` listed are now closed, and
> one of them was closed by discovering the question was wrong. **The covariates are nothing**
> (R² −0.014 alone; they make `volume` worse). **Volume regressed out fold-safe takes every
> arm to ≤ 0**, the location baseline included — a stronger and less flattering statement than
> the raw table supported. And **"use more of ARC" is not available**: 902 is scanning
> *sessions*, not participants. There are 245 participants, 228 have a lesion mask, and the
> other 17 have no mask at all. **n = 228 is the entire labelled cohort**, not a sample of it.
>
> **Status, 2026-08-30 (late).** Phases 0, 1, 2 and 6 are done: a lesion mask goes in and
> a disconnection profile and a readable report come out, over 108 real de-identified
> lesions. **Phase 3 is no longer blocked on labels** — ARC's 228 labelled subjects were
> on disk all along (`data/raw/arc/`), and the earlier "blocked on outcome labels" note
> was stale for that cohort. The blocker was registration, and M1 now exists:
> `src/lesionconn/registration.py`, validated at median Dice 0.924. See
> [docs/REGISTRATION.md](docs/REGISTRATION.md).
>
> **Two disconnection matrices now exist**, built from independent references, and they
> agree at network level (ρ +0.33) while anti-correlating parcel by parcel (ρ −0.32).
> That is the most consequential result so far and it constrains Phase 3:
> **individual parcel-pair edges are not reliable features.** See
> [docs/DISCONNECTION.md](docs/DISCONNECTION.md).

The ordering principle: **test the thesis before building the infrastructure.** The whole
premise ("disconnection predicts deficit better than location") can be checked on
pre-normalized public data with zero registration code. If that check fails, no amount of
polished pipeline saves the project — so it goes first. Everything else is earned only after
the core claim holds.

---

## Phase 0 — Setup & de-risking  ·  (~a few days)

Goal: environment ready, data in hand, no modeling yet.

- [x] Native-Windows Python env: `nibabel`, `numpy`, `nilearn`, `pandas`, `pytorch`, `monai`.
      Project `.venv` on Python 3.13, pinned in `requirements.txt`. CPU torch is deliberate —
      Phases 0–3 are `numpy`/`sklearn` only; revisit CUDA at Phase 5.
- [x] One WSL2 Ubuntu on standby (installed, stopped, FreeSurfer not installed).
- [ ] Download **ATLAS** — BLOCKED ON A HUMAN STEP. Google form at
      <https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html> → encryption key shown on
      submission (save it there; the promised email often never arrives). **R3.0 supersedes
      R2.0** (1,453 scans vs 955). Decrypt in **WSL2**, not Windows, with
      `openssl aes-256-cbc -md sha256 -d -a`. Full operational notes, the known-bad-subject QC
      list, and the template-space analysis: [docs/ATLAS-notes.md](docs/ATLAS-notes.md).
      **Note:** ATLAS's own normalization is 12-DOF affine with *no* cost-function masking, to
      `MNI152NLin2009aSym` — not our canonical FSL MNI152 grid. Decision: bridge that gap by
      affine regrid (Dice 0.968, ~1 mm offset). TemplateFlow's nonlinear alternative was
      tested and **rejected** — it made correspondence worse in every configuration tried.
      `scripts/04_template_gap_report.py` reproduces the comparison.
- [x] Pick a parcellation: Schaefer-400 (2mm, 7-network) + Harvard-Oxford subcortical,
      merged to **N = 415** on the FSL MNI152 2mm grid. `scripts/01_build_parcellation.py`.
- [x] Sanity check: `scripts/02_sanity_overlay.py` — a synthetic sphere at MNI (-26, 0, 8)
      lands in left putamen (31% of it), left pallidum and left fronto-opercular cortex.
      Renders to `data/derived/sanity_overlay.png`.
- [x] **Decision point — RESOLVED: NO.** ATLAS R2.0 withholds demographics and clinical
      outcome measures for privacy. It cannot be the Phase 3 target. The labelled cohort is
      **ARC** (OpenNeuro `ds004884`): 245 participants, *all* with a WAB Aphasia Quotient
      (range 5–100, mean 64.2), plus age at stroke, sex and days-post-stroke.

Exit: **met** for the synthetic path — lesion, parcellation and template all load on one grid
that is asserted in code (`lesionconn.spaces`), not assumed. Still open: the same check
against a real ATLAS mask, once the encryption code arrives.

### What Phase 0 actually caught

Two bugs that no shape, count or volume check would have revealed:

1. **Harvard-Oxford ships RAS-ordered; Schaefer ships LAS.** Identical world box, identical
   2mm voxels, x-axis stored in reverse array order. Merging the raw arrays — the obvious
   implementation — mirrors every subcortical structure, so a left-putamen lesion comes back
   labelled *right* putamen. Fixed by reorienting in `atlases.fetch_harvard_oxford_subcortical`;
   guarded by `test_left_and_right_lesions_are_mirror_images`.
2. **`nilearn.load_mni152_template()` is ICBM 2009c (99×117×95), not FSL MNI152 (91×109×91).**
   Two different grids both legitimately called "MNI". This is why `assert_canonical` exists.

### Consequence for the ordering: Phase 4 partly precedes Phase 3

ARC lesion masks are in **native T2w space** — 228 masks across 5 distinct per-subject grids
(1mm; 176×256×256 and 160×256×256 dominate; two are PSR-oriented). Zero are MNI-aligned.
Verified from the headers, not the filenames, by `scripts/03_fetch_arc.py`.

So the roadmap's assumption that the labelled cohort would be pre-normalized is false. The two
cohorts are complementary, and neither alone is a shortcut:

| | lesion masks | outcome labels |
|---|---|---|
| ATLAS R2.0 | pre-normalized to MNI | **none** |
| ARC (ds004884) | native T2w space | **WAB-AQ, n=245** |

**M3 still gets built registration-free on ATLAS.** But Phase 3 — the go/no-go thesis test —
now needs M1 registration (ANTsPyx; `cp313` Windows wheels confirmed available) to bring ARC
masks into MNI first. Budget that before Phase 3, and note that lesioned-brain registration is
exactly where cost-function masking / enantiomorphic normalization becomes mandatory.

---

## Phase 1 — Disconnectome for one brain  ·  **DONE**

Goal: one lesion → one disconnection matrix. No ML yet.

- [x] Normative connectome: **HCP-842**, 80 named tracts, sourced from the NeuroVue project
      and read in place. `data/atlases/tract_atlas.nii.gz` is the canonical-grid build.
- [x] **Path B (build)** taken. LQT was not needed — `src/lesionconn/disconnectome.py` does
      the intersection in `nibabel` + `numpy`. `scripts/11_build_tract_atlas.py`.
- [x] `disconnection_matrix.npy` (415×415) and `severed_tracts.json` per subject, plus the
      80-dim tract-severance vector. Heatmap and bar chart in `scripts/05_report.py`.

Exit: **met.** `disconnectome.severed_tracts(lesion)` and `disconnectome.disconnection_matrix()`
are reproducible functions over any canonical-grid mask.

### What Phase 1 caught

The atlas is 1mm **LPS**; the canonical grid is 2mm **LAS**. Same failure class as Phase 0's
Harvard-Oxford/Schaefer bug, so laterality is asserted twice and independently — once on
anatomy (sign, bulk, L/R pair order) and once with no anatomy at all (every tract's centre of
mass must land where it started). Both are shown *failing* on a deliberately mirrored atlas in
`tests/test_disconnectome.py`.

Two thresholds were set by measuring, and the first guess was wrong both times:

1. A ">5mm off midline" rule for lateralised tracts rejected seven **brainstem** tracts whose
   centres of mass genuinely sit 2–5mm out — in the native atlas too, so the resampling was
   faithful and the check was the error.
2. The centre-of-mass tolerance is 5mm, from a measured distribution of median 0.19 / p95 0.91
   / max 3.94mm, whose three largest shifts are all tracts under 100 voxels — sampling noise
   going as 1/√N, not misregistration.

Resampling cost is recorded per tract, as `intake_lesion_mask` records it for lesions. Real
tracts lose under 2% of volume at 2mm; the cranial nerves lose ~10%. 18 of 80 are flagged
`thin`, never dropped.

---

## Phase 2 — Batch the cohort  ·  **DONE** (108 lesions, no labels)

Goal: run Phase 1 over the whole dataset, build the feature store.

- [x] M3 over **108 real lesions** (from the NeuroVue project, `(182,218,182)` 1mm LAS —
      already MNI-resliced, so no registration needed). `src/lesionconn/cohort.py`.
- [x] `manifest.csv`: de-identified id, cached artifact paths, native volume, QC status and
      reason, location-baseline summary. **No outcome label column — there are none.**
- [x] Location baseline features from the N=463 atlas, cached per subject.
- [x] Everything cached under `data/derived/cohort/`; a re-run reuses all 108 in ~5s.

Exit: **met.** Matrices, tract vectors, location vectors, a manifest, and a per-subject
`severed_tracts.json`.

### Milestone 0 — de-identification came first

The 108 filenames carried patient names and apparent MRNs. Ids are `sub-0001…sub-0108`,
assigned by sorted filename so a re-run never renumbers; the id↔filename map is the only
artifact holding them and lives in `data/private/`, gitignored twice.

`tests/test_deid.py` greps **every** emitted artifact — source, docs, CSV, JSON, NIfTI, PNG —
for every identifying token, and it scans this repo's own source too. That has caught a real
name three times: twice quoted in a code comment by the author, once in a report path.

Two narrowings were needed, both measured before being made. Raw float32 `.npy` payloads and
compressed PNG `IDAT` streams produce **coincidental** matches (3 matrices and 4 figures), so
those regions are excluded — while `.npy` headers, text-dtype `.npy` payloads (decoded, since
numpy stores unicode as UTF-32), NIfTI headers, and PNG `tEXt`/`zTXt`/`iTXt` chunks
(decompressed) are all still searched, each proven by a test that plants a name there.

### QC: 0 of 108 excluded, and that is a measured result

The gate ran on every subject and nothing came near a threshold: smallest lesion 496 mm³
against a 100 mm³ floor, worst 2.2% outside brain against a 10% limit, largest 14.7% of brain
against a 50% limit. No subject sits within 2× of any threshold. This cohort arrives clean.

13 subjects change volume by more than 10% on the 1mm→2mm resample — exactly B2, and exactly
why the native figure is the one recorded and the one the report quotes.

---

## Phase 3 — Test the thesis  ·  (~1–2 weeks)  ★ the pivotal phase

Goal: answer *does disconnection beat location?* This is the go/no-go for the whole idea.

> **UNBLOCKED as of 2026-08-30.** The note that stood here — "blocked on outcome labels" —
> was true of the 108 and stale for ARC. `data/raw/arc/participants.tsv` has a WAB-AQ for
> all 245 participants, and **228 of them have a lesion mask**: mean 63.3, sd 28.1, full
> range 5–100. The 108 still have no labels and none have been invented.
>
> The real blocker was registration, and it is done (Phase 4a). What remains is to run the
> comparison and read it honestly. Feature objects ready: 80-dim tract severance (primary),
> 136 network edges from each of the two matrices, 463-dim location baseline (Model A), and
> the `volume+X` incremental arms that the confound measurement made necessary.

- [x] Model A (baseline): location features → outcome. Ridge / elastic-net / GBT.
- [x] Model B (thesis): tract severance and network edges → outcome, same families.
- [x] Nested CV, 25 outer folds, distributions not point scores, permutation null.
- [x] **Answered: B ≈ A.** Full write-up and the paired test: [docs/PHASE3.md](docs/PHASE3.md).
- [x] M4 shipped anyway: `src/lesionconn/predict.py` fits on all 228, persists, and
      `05_report.py` prints a predicted WAB-AQ — refusing outright for lesions outside the
      left-hemisphere aphasia distribution it was trained on.
- [x] **Covariates added and measured: they are nothing.** age, sex and days post-stroke are
      complete for 228/228. Alone they score R² −0.014; added to lesion volume they make it
      *worse* (0.387 against 0.398). `wab_days` has partial r = −0.002 given volume — ARC is a
      chronic cohort at a median 721 days, so recovery has plateaued. The nominally significant
      sex effect is a volume confound.
- [x] **Volume regressed out fold-safe, not merely reported.** Every arm, and all 18
      (feature set × model) cells, score at or below zero on the volume-free residual —
      the location baseline included.
- [x] **Permutation null on the tree models — all 6 arms done, and the gap is closed.**
      `permutation_p` takes a model name and is parallelised over shuffles (verified
      elementwise identical to the serial version). 200 shuffles per arm on `hist_gb`, the
      family the headline numbers come from. **All five real arms reach p = 0.005**, the 1/201
      floor — counted network edges +0.5104, location +0.4487, tracts +0.4340, inferred network
      edges +0.4325, volume +0.4160 — each against a shuffled median near −0.21. **The
      `covariates` control does not**: −0.1347 at p = 0.4527, indistinguishable from its own
      null, which is what makes the other five readable. Bonferroni over the 6-test family is
      0.03, so all five survive. Full table: [docs/PHASE3.md](docs/PHASE3.md).

      This does **not** touch the Phase 3 answer. Every arm beats chance; none beats another.
      B ≈ A is decided by the paired Wilcoxon, not by this.

      106.7 minutes of compute, unevenly spread: `location` alone is 74.7 of it at 463
      features, against 4.2 for the 80-dim tract vector. Cost grows faster than feature count,
      so estimates scaled linearly from single-fit times come out ~2× optimistic — the
      checkpoint-per-arm design is what made a run that was killed twice by session teardown
      recoverable both times.
- [x] Status deck for a non-specialist audience: `docs/Lesion_Disconnection_Status.pptx`,
      15 slides, built from the result CSVs by a script so the numbers cannot drift from the
      data. Emitting it exposed **docs/BUGS.md B9** — a ZIP container was invisible to the
      de-identification scan.

Superseded checklist (kept for the record):
- [ ] Model A (baseline): location features → outcome. Ridge / elastic-net / GBT.
- [ ] Model B (thesis): flattened disconnection matrix → outcome. Same model families.
- [ ] **Nested cross-validation**, heavy regularization — features hugely outnumber subjects,
      so overfitting is the default. Report confidence intervals, not point scores.
- [ ] Compare A vs B honestly.

Exit / decision:
- **B > A** → thesis holds, the project has a spine. Proceed.
- **B ≈ A** → dig into representation (Phase 5) before declaring anything.
- **B < A** → stop and rethink; something's off in M3 or the labels. Better to learn this now
  than after building a UI.

---

## Phase 4a — M1 registration, ARC only  ·  **DONE**

Pulled forward, because Phase 3 cannot run without it. `src/lesionconn/registration.py`,
`scripts/16_fetch_arc_anat.py`, `scripts/17_register_arc.py`, `scripts/18_build_arc_cohort.py`.
Full write-up: [docs/REGISTRATION.md](docs/REGISTRATION.md).

- [x] The 228 T2w anatomicals the masks are drawn on, fetched by sidecar reference (~0.91 GB,
      targeted — not the 902-session blind sync `03_fetch_arc.py` warns about).
- [x] T2w → FSL MNI152 1mm (the grid ANTsPy ships, asserted against `Grid.canonical()`),
      `SyNRA`, Mattes MI on both stages because T2-against-T1 is cross-modality.
- [x] **Cost-function masking with `mask_all_stages=True`** — ANTsPy's default masks the SyN
      stage only, leaving the affine free to pull healthy tissue into the lesion.
- [x] Pilot of 8 before the batch, looked at as images and not only as numbers.

### What the pilot caught

**A −30% lesion volume change that is not a defect.** Every subject's lesion shrinks by about
a third entering MNI. Measured against the transform that produced it: affine linear
determinant 1.5144, so the implied moving→fixed volume scale is 0.660, against a measured
lesion ratio of 0.704. The lesion scales by very nearly the global factor, and whole-brain
Dice against the template is **0.924 median / 0.901 min**. The compression is uniform and
correct. Consequence: for ARC the MNI volume is the honest one, which is the *opposite* of the
rule for the 108 (docs/BUGS.md B2), where no registration happened.

**Laterality is the QC that matters.** ARC is an aphasia cohort, so a right-dominant warped
lesion has been mirrored — the failure that hit Phase 0 and Phase 1. All 8 pilot subjects came
back left-dominant (median `frac_left` 1.000). Two ARC masks are PSR-oriented against 226 RAS,
so both paths have to survive it.

---

## Phase 3 harness  ·  **BUILT AND VALIDATED** (on the 108, against lesion volume)

`src/lesionconn/modeling.py`, `scripts/19_train.py`. Nested CV, distributions rather than point
scores, a permutation null, and a lesion-volume arm forced into every comparison.

Validated on the 108 with **log lesion volume** as the target — a real measured quantity, not
an invented outcome. Two things came out of it:

- **The positive control passes.** The volume arm predicting log volume returns R² = 1.0000
  exactly, so the machinery is wired correctly end to end.
- **The confound is large, and now measured.** How much of each feature set is simply lesion
  size:

  | feature set | R² predicting log lesion volume |
  |---|---|
  | network edges, counted matrix | **0.656** |
  | network edges, inferred matrix | 0.612 |
  | 80-dim tract severance | 0.467 |
  | 463-dim location baseline | 0.377 |

**Consequence for Phase 3, and it changes the design:** beating the location baseline is not
enough, and neither is beating volume head-to-head. Two-thirds of the network-edge features is
recoverable lesion size, so the question has to be asked incrementally — `volume+tracts`
against `volume` alone. Those arms are now in the default run.

The harness is also tested against itself: pure noise must score R² ≤ 0 and must not reach a
small permutation p. Both are in `tests/test_modeling.py`, because a CV harness that
manufactures signal would produce a thesis result out of nothing.

---

## Phase 4 — Raw-MRI intake (make it end-to-end)  ·  (~2–3 weeks)

Goal: accept a real brain, not a pre-cooked mask. Only worth it once Phase 3 holds.

- [ ] M1 registration: start with **ANTsPyx** (native Windows) to warp a raw T1 → MNI.
      Try SynthMorph/EasyReg via WSL2 if you want the speed.
- [ ] Add **cost-function masking** (or enantiomorphic normalization) so the lesion doesn't
      distort its own warp. This matters the moment you leave pre-normalized data.
- [ ] M2 segmentation: begin with manual/provided masks; train an nnU-Net/MONAI segmenter
      only if "auto-find the lesion" is a goal.
- [ ] Wire M1→M2→M3 so a raw T1 + mask flows through to a disconnection matrix.

Exit: raw T1 in → disconnection matrix out, no manual normalization step.

---

## Phase 5 — Sharpen the model (the contribution)  ·  (open-ended)

Goal: go past the baseline into something novel and publishable.

- [ ] GNN over the disconnection graph; justify it by beating the flattened-vector baseline.
- [ ] Try connectome-normalization variants — which one predicts best is an open question.
- [ ] Low-rank / harmonic embeddings of the matrix as features.
- [ ] Ablations: which connections carry the predictive signal?

Exit: a result that says something the off-the-shelf tools don't.

---

## Phase 6 — Report & viz layer  ·  **DONE**

Goal: the output a human actually wants.

- [x] Atlas-referenced text report: volume, anatomy with % overlap, severed tracts with %
      severed *and* absolute volume, top disconnected parcel pairs.
- [x] Glass brain, lesion on template, severed-tract bar chart, matrix heatmap.
- [x] One entry point: `scripts/05_report.py <lesion.nii.gz>` (or a `sub-NNNN` id).
- [x] "Research tool, not a diagnostic" in the report header, its footer, the JSON, and the
      caption of every figure.

Exit: **met.** Verified by running it on three real lesions and reading the output.

### What reading the output caught

Neither of these would have failed a test written from the spec; both were only visible by
looking at a real report.

1. **The anatomy section used the wrong atlas.** Built from the N=415 node set it led with
   *left* occipital parcels for a lesion that is 70% *right* cerebellum — the node set has no
   cerebellar labels, so it saw only the 17.8% of that lesion outside the cerebellum and
   presented that minority as a confident table. Anatomy now comes from the N=463 location
   atlas, and the node-set section states its own coverage and says so loudly when most of the
   lesion is invisible to it.
2. **The "native" volume was measured on resampled data.** A cohort id resolved to the cached
   2mm mask, so sub-0035 reported 424 mm³ against a true 496 mm³ — a 14.5% error on precisely
   the number B2 exists to protect.

A third, milder finding: ranking tracts by `% severed` structurally favours small tracts.
Across the 108 subjects **none of the six largest tracts was ever the top hit**, though the
corpus callosum is involved in nearly every large lesion. The report and the bar chart show
absolute severed volume beside the percentage for this reason.

---

## Phase 1b — the counted matrix  ·  **DONE**

A second, independent M3a, from real streamline endpoints rather than an inferred
tract→parcel mapping. `src/lesionconn/streamlines.py`, `scripts/15_build_streamline_index.py`.

- [x] `S35_1mm.trk` — 479,457 streamlines, FSL MNI152 1mm LAS, the same world box as the
      canonical grid — read in place from the NeuroVue project.
- [x] Each fibre's two ends assigned to a parcel (4 mm radial search); entry (i,j) is the
      counted fraction of edge i–j's fibres passing through the lesion.
- [x] Built for all 108 subjects alongside the inferred matrix. Neither replaces the other.
- [x] Edge fibre-counts carried alongside, with a ≥10-fibre floor on anything a human reads.

### What Phase 1b caught

**The premise of the whole approximation was false.** `docs/DISCONNECTION.md` justified the
inferred matrix by stating we had no tractogram. We did — it was in the same project as the
other assets, in a directory that was never opened. The document now carries the correction
rather than a quiet edit.

**The two matrices disagree, and where they disagree is the finding.** Over all 108 lesions:

| | median Spearman ρ | subjects positive |
|---|---|---|
| parcel level (415 nodes) | **−0.318** | 4 / 108 |
| network level (17 groups) | **+0.331** | 104 / 107 |

The sign flips. Three explanations were tested and rejected before the surviving one —
normalisation (+0.045 / −0.029, far too small), fraction-vs-count (still −0.292), and a
laterality or space bug (both matrices are correctly lateralised and mirror properly). What
is left is granularity: both methods name the same anatomy and pick different *adjacent*
Schaefer parcels within it. On the corona radiata probe both say motor cortex ↔ brainstem;
one picks `SomMot_35`, the other `SomMot_37` (963 fibres, 100% cut).

**Consequence for Phase 3:** coarsen to networks, or restrict to edges where the two agree,
before treating 415×415 edges as features. This supersedes the old limit 8.

**A third normalisation trap.** Median edge support is 9 fibres, so an unfiltered ranking is
led by edges resting on a single streamline reading "100% severed" — the same failure as
ranking tracts by `frac_severed`. Reporting floors at 10 fibres and always prints the count;
the matrix itself is never filtered.

---

## Stretch — NeuroVue hand-off  ·  **partly done**

`src/lesionconn/neurovue.py` + `scripts/14_export_neurovue.py`. **NeuroVue is not modified.**

- [x] File drop: writes the exact `LESION_DIR/<slug>/lesion_<timestamp>.nii.gz` layout
      NeuroVue's own `POST /api/lesions` produces, plus a per-subject `disconnection.json`
      and an `index.json`. `LESION_DIR=<bundle> uvicorn server:app` picks it up with no code
      change and no database write. Verified: bundle written for 3 subjects, layout and
      de-identification asserted.
- [x] Upload client for NeuroVue's existing route. Verified against a **stub** implementing
      that route's declared multipart contract.
- [ ] **Not run against a live NeuroVue backend** — that needs its FastAPI app and MongoDB
      running. The upload path should be considered unproven until it is.
- [ ] No NeuroVue UI work: the viewer shows the mask, but nothing there reads
      `disconnection.json` yet. Rendering the severed-tract list in the sidebar is the
      obvious next step, and is the first thing that would touch NeuroVue's own code.

---

## Throughline reminders

- **Labels are the bottleneck.** Chase the outcome-labelled cohort early (Phase 0), not late.
- **Baseline or it didn't happen.** Every disconnection result is meaningless without the
  location-feature comparison.
- **Normative-connectome humility.** State the average-wiring assumption wherever you report.
- **Registration is inherited; disconnection is a lookup; only M4 is learned.** Don't blur them.
