# PLAN.md

## What this is

A tool + research pipeline that takes a brain lesion and works out **not just where it
sits, but what it disconnects** — which white-matter connections it severs — and then uses
that disconnection pattern to predict clinical outcome.

The core bet: *a lesion's location is a weak predictor of deficit; the connections it cuts
are a much stronger one.* This is well-supported in the literature. The contribution here is
a clean, reproducible pipeline that gets to the disconnection features fast, plus a model
that maps those features to outcome.

## Why it's worth building

The mechanics already exist across FSL / ANTs / SPM / BCBtoolkit / Lesion Quantification
Toolkit — but they're fragmented, CLI-heavy, MATLAB-flavored, and assume the user already
knows the failure modes (e.g. that a lesion distorts its own registration). Nobody has made
the disconnectome path feel like a product, and the *disconnection-pattern → outcome* model
is genuinely unsolved territory.

So the value is in two places:
1. **Packaging** — a reproducible, Windows-friendly, mostly-Python path from lesion to
   disconnection matrix.
2. **The model** — disconnection features → deficit prediction, benchmarked against a
   lesion-location baseline. This is the part that's a real contribution rather than a wrapper.

## The central idea in one line

Lesion mask (in MNI space) → intersect with a **normative connectome** → disconnection
matrix → ML model → predicted deficit.

The unlock that makes this tractable: **the patient never needs diffusion imaging.** You use
a precomputed healthy-population connectome and just intersect the lesion with it. This
collapses a hard per-patient tractography problem into a masking-and-lookup problem. Patient
input is a single structural T1 (or, for v0, a lesion mask that's already drawn).

## Scope boundaries (what this is NOT)

- **Not a diagnostic device.** Framed as a *research annotation / analysis tool*. Anything
  that says "here is what is damaged in this patient, clinically" starts flirting with being
  a regulated medical device. Stay on the research side of that line.
- **Not a registration research project.** Registration (MRI→MNI) is an inherited, pretrained
  component. We do not train it.
- **Not per-patient tractography.** We use normative connectomes, with eyes open about the
  approximation that entails.

## Honest caveats to keep pinned

1. **Normative connectomes assume average wiring.** They're most wrong in exactly the
   reorganized brains we most care about. This is the field's standard sin — use it, but
   never claim more than it gives.
2. **A lesion distorts its own registration.** The intensity abnormality pulls the warp off.
   Standard fixes: cost-function masking (Brett) or enantiomorphic normalization (Nachev).
   Only relevant once we accept raw MRIs — v0 sidesteps it by using pre-normalized data.
3. **Outcome labels are the bottleneck, not the imaging.** The imaging pipeline is the easy
   part. Whether we can predict anything depends entirely on having lesions paired with
   clinical scores. **Confirmed in Phase 0:** ATLAS R2.0 withholds all clinical outcome
   measures for privacy, so it can never be the prediction target — it is an imaging
   development set only. The labelled cohort is ARC (OpenNeuro `ds004884`, n=245, WAB-AQ).
4. **The two usable cohorts are complementary, not interchangeable.** ATLAS is pre-normalized
   but unlabelled; ARC is labelled but its masks are in native T2w space. Registration is
   therefore *not* optional for the thesis test, as the roadmap originally assumed — see the
   Phase 0 notes in ROADMAP.md.
5. **"MNI" is not one space.** `nilearn`'s MNI152 template is ICBM 2009c (99×117×95); Schaefer
   and Harvard-Oxford ship on FSL MNI152 (91×109×91); Harvard-Oxford is RAS-ordered where
   Schaefer is LAS. Volumes that are all nominally "in MNI" routinely disagree on grid and on
   array orientation, and an L/R flip is invisible to every shape, count and volume check.
   Everything in this project asserts its grid via `lesionconn.spaces` rather than assuming it.

## Success criteria

**v0 succeeds if:** on a public cohort, a model using disconnection features beats a model
using lesion-location features at predicting an outcome label. That reproduces the field's
central claim on our own pipeline and validates the whole premise cheaply.

**v1 succeeds if:** the pipeline accepts a raw T1 + lesion mask, normalizes it, and produces
a disconnection matrix + a readable atlas-referenced report end-to-end.

**Stretch success:** the outcome model does something novel — a better architecture, a better
connectome-normalization choice, or a disconnection representation that outperforms the
standard parcel-based matrix.

## Key external assets (all public)

- **ATLAS R2.0** — ~1000 stroke lesions, pre-segmented AND pre-normalized to MNI. This is the
  de-risking dataset: it lets v0 skip registration and segmentation entirely.
- **Normative connectome** — HCP-842 tractography atlas, or the connectome bundled with
  Lesion Quantification Toolkit (LQT).
- **Parcellations** — Schaefer-400 (+ subcortical), AAL3, Harvard-Oxford for gray matter;
  JHU for white matter.
- **Registration (later)** — SynthMorph / EasyReg (deep, seconds, contrast-agnostic) or
  ANTsPyx (classic SyN, pure-Python, native Windows).
