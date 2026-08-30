# ROADMAP.md

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

## Phase 1 — Disconnectome for one brain  ·  (~a week)

Goal: one lesion → one disconnection matrix. No ML yet.

- [ ] Get a normative connectome (LQT-bundled or HCP-842).
- [ ] **Path A (wrap):** run Lesion Quantification Toolkit on one ATLAS lesion → matrix.
      Check whether the Python build does what you need before committing to MATLAB.
- [ ] **Path B (build):** intersect the lesion with HCP-842 in `nibabel`+`numpy` → N×N matrix
      yourself. Fall back here if MATLAB blocks you.
- [ ] Produce `disconnection_matrix.npy` for a single subject and visualize it (heatmap +
      a chord diagram). Does it look sane — sparse, symmetric, hitting plausible regions?

Exit: a reproducible function `lesion_mask → N×N disconnection matrix`.

---

## Phase 2 — Batch the cohort  ·  (~a few days)

Goal: run Phase 1 over the whole dataset, build the feature store.

- [ ] Loop the M3 function over all ATLAS lesions → one matrix per subject.
- [ ] Build `manifest.csv`: subject_id, matrix path, outcome label.
- [ ] Also compute the **location baseline features** (which parcels each lesion overlaps) —
      you need these for the benchmark in Phase 3.
- [ ] Cache everything to disk so you never recompute.

Exit: a folder of matrices + a manifest + a parallel location-feature table.

---

## Phase 3 — Test the thesis  ·  (~1–2 weeks)  ★ the pivotal phase

Goal: answer *does disconnection beat location?* This is the go/no-go for the whole idea.

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

## Phase 6 — Report & viz layer  ·  (~1 week)

Goal: the output a human actually wants.

- [ ] Atlas-referenced text report (region overlaps + severed tracts, `atlasquery`-style).
- [ ] Glass-brain + chord-diagram renders (`nilearn`).
- [ ] Wrap the whole pipeline behind one entry point (a script or a thin web UI).
- [ ] Keep the **"research tool, not a diagnostic"** framing visible in the output itself.

Exit: drop in a brain, get back a readable disconnection report.

---

## Throughline reminders

- **Labels are the bottleneck.** Chase the outcome-labelled cohort early (Phase 0), not late.
- **Baseline or it didn't happen.** Every disconnection result is meaningless without the
  location-feature comparison.
- **Normative-connectome humility.** State the average-wiring assumption wherever you report.
- **Registration is inherited; disconnection is a lookup; only M4 is learned.** Don't blur them.
