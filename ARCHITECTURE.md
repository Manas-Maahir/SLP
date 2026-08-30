# ARCHITECTURE.md

## Design principle

The single most important structural decision: **keep the learned parts and the plumbing
parts separate.** The failure mode is trying to build one model that "learns the brain and
does everything." Don't. Registration is inherited, disconnection is a lookup, and the entire
learning budget goes into one module (outcome prediction).

Modules communicate through files on disk (NIfTI masks, `.npy` matrices, a manifest CSV), not
through a shared in-memory object. That keeps each stage independently runnable, cacheable,
and swappable.

## The five modules

```
                 ┌─────────────────────────────────────────────────────┐
   Raw T1  ─────►│ M1: Registration      (INHERITED, pretrained)        │
                 │     SynthMorph / EasyReg / ANTsPyx                    │
                 │     out: native↔MNI transform + brain in MNI         │
                 └─────────────────────────────────────────────────────┘
                                        │
   Lesion  ─────►┌─────────────────────────────────────────────────────┐
   (raw)         │ M2: Segmentation      (TRAIN LATER, or skip in v0)   │
                 │     nnU-Net / MONAI  → binary lesion mask            │
                 └─────────────────────────────────────────────────────┘
                                        │
                                        ▼
                 ┌─────────────────────────────────────────────────────┐
                 │ M3: Disconnectome     (LOOKUP, not learning)         │
                 │     lesion-in-MNI ∩ normative connectome             │
                 │     out: disconnection matrix + severed-tract list   │
                 └─────────────────────────────────────────────────────┘
                                        │
                                        ▼
                 ┌─────────────────────────────────────────────────────┐
                 │ M4: Outcome model     ◄── THIS IS THE RESEARCH       │
                 │     disconnection features → predicted deficit       │
                 │     benchmarked vs a lesion-location baseline        │
                 └─────────────────────────────────────────────────────┘
                                        │
                                        ▼
                 ┌─────────────────────────────────────────────────────┐
                 │ M5: Report / viz      (presentation layer)          │
                 │     atlas-referenced text + glass-brain / chord      │
                 └─────────────────────────────────────────────────────┘
```

---

## M1 — Registration (MRI → MNI)

**Status: inherited. Do not train.**

Job: compute a transform from native patient space to MNI152, and resample the brain (and
anything drawn on it) into MNI space.

| Option | Runtime | Windows story | Notes |
|---|---|---|---|
| SynthMorph / EasyReg | seconds (GPU) | via WSL2 or a PyTorch port | contrast-agnostic, deep |
| ANTsPyx | minutes | `pip install antspyx`, native | classic SyN, very validated |
| FSL FNIRT / SPM | minutes–hours | WSL2 / MATLAB | the old guard, most references |

**The lesion-distorts-registration gotcha lives here.** When we accept raw lesioned brains,
apply cost-function masking or enantiomorphic normalization *before* computing the warp,
or the labels near the lesion will be subtly wrong. Not relevant in v0 (data is pre-normalized).

**v0 action: skip this module.** ATLAS R2.0 lesions are already in MNI.

---

## M2 — Lesion segmentation

**Status: skip in v0, train later.**

Job: turn a raw MRI into a binary lesion mask.

- **v0:** use masks that already exist (ATLAS ships them).
- **v1+:** nnU-Net or a MONAI segmentation net trained on lesion masks for the pathology of
  interest. This is the "drop a raw MRI and it finds the lesion" magic — worth it only once
  the rest works.

Output contract: a binary NIfTI in MNI space, same grid as the connectome/parcellation.

---

## M3 — Disconnectome (the core mechanic)

**Status: lookup, not learning.** No per-patient diffusion imaging. This is the module that
makes the whole thing tractable.

Two representations — build the matrix one first, keep the tract one as the human-readable view:

### 3a. Parcel-based disconnection matrix  ← the ML substrate
1. Take a parcellation (Schaefer-400 + subcortical) → N regions.
2. Take a normative structural connectome defined over those parcels (region×region edges,
   each edge backed by a bundle of streamlines).
3. For each edge, compute how much of its streamline bundle passes through the lesion mask.
4. Output an **N×N disconnection matrix**: entry (i,j) = fraction of the i–j connection
   severed. Symmetric, mostly sparse.

This matrix *is* the feature object for M4.

### 3b. Tract-based severance  ← the human-readable view
1. Take a tractography atlas (HCP-842).
2. Intersect the lesion with it → which named tracts pass through the lesion.
3. Output a severed-tract list + a whole-brain probabilistic disconnection map.

### Build vs. wrap
- **Wrap:** Lesion Quantification Toolkit (Griffis) does both 3a and 3b out of the box.
  Check whether the Python version covers your needs — the core historically leaned MATLAB.
- **Build:** if MATLAB is a blocker, reimplement 3a yourself in `nibabel` + `numpy` by
  intersecting the lesion with the HCP-842 atlas. More code, full control, pure Python.

Output contract: `disconnection_matrix.npy` (N×N float) + `severed_tracts.json`.

---

## M4 — Outcome model (THE RESEARCH)

**Status: this is the part that's yours.** Everything above is plumbing whose job is to
produce a good feature object cheaply.

Input: the N×N disconnection matrix (per subject).
Output: a predicted clinical outcome (regression score, syndrome class, or recovery label).

**Representations to try, simplest first:**
1. **Flatten** the upper triangle of the matrix → a feature vector → ridge / elastic-net /
   gradient-boosted trees. Boring, strong, and the honest baseline. Start here.
2. **Graph neural network** — treat the disconnection matrix as a weighted graph, GNN over it.
   The "proper" representation; justify it by beating (1), don't assume it wins.
3. **Low-rank / connectome-harmonic** embeddings of the matrix as features.

**The benchmark that must exist:** a parallel model trained on **lesion-location features**
(which parcels the lesion overlaps, ignoring connectivity). The entire thesis is
*disconnection > location*. Without this baseline, a good score means nothing.

Watch: tiny-N, high-dimensional features → overfitting is the default outcome. Heavy
regularization, nested cross-validation, and report confidence intervals, not point scores.

---

## M5 — Report / visualization (presentation)

**Status: last, and deliberately so.** Clinicians want the report, not the MNI overlay — the
overlay is a researcher's artifact.

- **Text report:** atlas-referenced, e.g. "lesion involves left corticospinal tract, 3.2 cm³,
  40% overlap with primary motor cortex; disconnects left M1 ↔ brainstem." Generated by
  running the lesion through `atlasquery`-style probabilistic lookups + the M3 output.
- **Visuals:** glass-brain rendering (nilearn), chord diagram of the disconnection matrix,
  lesion overlaid on the MNI template.

`nilearn` covers most of this in pure Python on native Windows.

---

## Data contracts between modules

| From → To | Artifact | Format |
|---|---|---|
| M1 → M2/M3 | brain + transform in MNI | NIfTI + transform file |
| M2 → M3 | lesion mask | binary NIfTI, MNI grid |
| M3 → M4 | disconnection matrix | `.npy`, N×N float |
| M3 → M5 | severed tracts | `.json` |
| M4 → M5 | prediction + weights | `.json` / model file |
| any → any | cohort index | `manifest.csv` (subject_id, paths, outcome label) |

## The Windows-native stack

Everything genuinely *yours* is pure Python on native Windows:
`nibabel` (I/O), `numpy` (matrix math), `nilearn` (viz + atlases), `PyTorch` + `MONAI`
(M2 and M4), `antspyx` (M1 fallback). Keep one **WSL2 Ubuntu** around only for the day you
want real FreeSurfer SynthMorph/EasyReg. You won't need it for v0.
