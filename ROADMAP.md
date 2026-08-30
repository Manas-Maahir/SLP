# ROADMAP.md

> **Status, 2026-08-30.** Phases 0, 1, 2 and 6 are done: a lesion mask goes in and a
> disconnection profile and a readable report come out, over 108 real de-identified
> lesions. Phase 3 — the thesis test — is blocked on outcome labels, not on code.
> Phases 4 (registration) and 5 (model) are untouched.

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

> **BLOCKED, and not by code.** The pipeline is built and the feature store exists for 108
> subjects, but **those 108 have no outcome labels**. Model A and Model B can both be
> trained the moment a target variable arrives; there is nothing to train on today, and
> inventing one would answer the question with an artefact. ARC (`ds004884`, WAB-AQ,
> n=245) remains the labelled cohort and needs M1 registration first — see Phase 4.
> Feature objects ready: 80-dim tract severance (primary), 415×415 disconnection matrix
> (secondary), 463-dim location baseline (Model A).

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

## Throughline reminders

- **Labels are the bottleneck.** Chase the outcome-labelled cohort early (Phase 0), not late.
- **Baseline or it didn't happen.** Every disconnection result is meaningless without the
  location-feature comparison.
- **Normative-connectome humility.** State the average-wiring assumption wherever you report.
- **Registration is inherited; disconnection is a lookup; only M4 is learned.** Don't blur them.
