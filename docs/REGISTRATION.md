# M1 — bringing ARC into MNI

The 108-lesion cohort arrived pre-normalised, so M3 was built registration-free. The labelled
cohort does not have that luxury, and Phase 3 cannot run without this step.

`src/lesionconn/registration.py`, `scripts/17_register_arc.py`.

## What arrives

228 ARC subjects have both a lesion mask and a WAB Aphasia Quotient. Every mask is in **native
T2w space** — read from the headers, not the filenames:

| masks | grid | orientation |
|---|---|---|
| 117 | (176, 256, 256) 1 mm | RAS |
| 108 | (160, 256, 256) 1 mm | RAS |
| 1 | (160, 254, 256) 1 mm | RAS |
| 2 | (212,193,157) and (230,164,176) 1 mm | **PSR** |

Zero are MNI-aligned. Five distinct grids and two orientations means anything that indexes
arrays rather than world coordinates will mirror a subset of the cohort and nothing else.

**The masks alone are not enough.** A binary mask carries no anatomy, so nothing in it can
drive a warp. Each sidecar names the image it was drawn on:

```json
"SpatialReference": "bids:raw:sub-M2001/ses-1253/anat/sub-M2001_ses-1253_acq-spc3_run-3_T2w.nii.gz"
```

`scripts/16_fetch_arc_anat.py` pulls exactly those 228 files (~0.91 GB) rather than syncing all
902 sessions. One sidecar (`sub-M2272`) says `"SpatialReference": "Text"`; BIDS naming is
deterministic, so the path is derived from the mask's own filename and the download would 404
loudly if that ever stopped being true. It did not.

## The three decisions

**The fixed image is the FSL MNI152 1mm grid.** ANTsPy ships it as `get_ants_data('mni')` —
(182,218,182) LAS, origin (90,−126,−72). That is the same grid the 108 lesions and
`S35_1mm.trk` already live on, so ARC lands where the rest of the project already asserts, and
the 2 mm step afterwards goes through the same `to_canonical` every other lesion takes.
`mni_template()` asserts shape, orientation and origin rather than trusting that sentence —
Phase 0 caught `nilearn.load_mni152_template()` returning ICBM 2009c, a different image with
the same name.

**Cost-function masking, at every stage.** A lesion is a hole in the anatomy the registration
is trying to match, so an unmasked warp pulls healthy tissue into it and distorts the exact
region being measured. The moving mask is the brain minus the lesion, dilated 3 mm because a
hand-drawn boundary sits inside abnormal tissue rather than on its edge. `mask_all_stages=True`
is not decoration: ANTsPy's default applies the mask to the SyN stage only, leaving the rigid
and affine stages free to make the same mistake at a coarser scale.

**Mutual information, because this is cross-modality.** A T2w moving image against a T1
template has inverted contrast; a correlation metric is meaningless across it. `aff_metric` and
`syn_metric` are both set to `mattes` explicitly, so a future ANTsPy default cannot silently
change it.

Transform: `SyNRA` (rigid → affine → SyN).

## The pilot, and why there was one

A 228-subject batch is long enough that launching it on unvalidated settings costs a session,
and every threshold in this project so far was wrong on the first guess (docs/BUGS.md). Eight
subjects were registered first, with the warped T2w saved for visual comparison.

```
  seconds              median 60.9    p5 57.0     p95 73.6
  volume_delta_pct     median -30.1   p5 -36.3    p95 -25.7
  frac_outside_brain   median 0.001   p5 0.000    p95 0.008
  frac_left            median 1.000   p5 0.983    p95 1.000
  left-dominant (frac_left > 0.5): 8/8
```

Whole-brain Dice against the template, measured afterwards:

| | median | min |
|---|---|---|
| Dice(warped T2w brain, template brain) | **0.924** | 0.901 |

That is ordinary published quality for affine + SyN normalisation. The overlays
(`data/derived/arc/pilot_qc_overlay.png`) were looked at, not just scored: brains fill the
template outline, ventricles land on the template's ventricles, and every lesion sits on the
left.

### The −30% volume change is the affine, not a defect

Every subject's lesion shrinks by roughly a third going into MNI, which looks alarming until
it is measured against the transform that produced it:

| | value |
|---|---|
| affine linear determinant (fixed→moving) | 1.5144 |
| implied moving→fixed volume scale | 0.660 |
| measured lesion volume ratio | 0.704 |

The lesion scales by very nearly the global factor, so the warp is compressing everything
uniformly rather than mangling the lesion. Dice 0.92 says that compression is correct.

**Consequence:** for ARC, native and MNI lesion volumes differ by a subject-specific factor,
so the volume used downstream is the MNI one — every subject measured in the same space. This
is the opposite of the rule for the 108 (docs/BUGS.md B2), where the native figure is the
honest one because no registration happened and the only volume change was resampling.

## How a warp is rejected

On anatomy, not on a similarity score, and in `scripts/18_build_arc_cohort.py` rather than in
the registration itself — so a warp can be re-judged without re-registering 228 subjects.

| check | threshold | why |
|---|---|---|
| `frac_left` | ≥ 0.20 | ARC is an aphasia cohort. A right-dominant lesion has been mirrored — the bug class that hit Phase 0 (Harvard-Oxford vs Schaefer) and Phase 1 (HCP-842 LPS vs LAS). Two masks are PSR and both paths must survive it. |
| `frac_outside_brain` | ≤ 0.20 | The lesion has been pushed out of the head. |

Both are deliberately loose. They exist to catch failure, not to prune the cohort toward a
nicer answer, and every exclusion is printed with its subject id.

`mutual_information` is recorded per subject but **is not a gate**. It is the same family of
metric the registration itself optimised, so it is a consistency check rather than an
independent one — it will catch gross failure and will not catch subtle misalignment. Treat it
as a sortable column for finding the worst subjects to look at by eye.

## What this does not do

- **No enantiomorphic normalisation.** Cost-function masking is the lighter of the two standard
  options; filling the lesion with the mirrored contralateral hemisphere is the heavier one and
  is worth testing if large lesions turn out to warp badly.
- **No lesion segmentation.** ARC's masks are hand-drawn and used as given. M2 remains untouched.
- **One transform, unvalidated against alternatives.** `SyNRA` was not compared against `SyN`,
  `SyNCC` or a GPU method (SynthMorph via WSL2 would be seconds per subject rather than a
  minute). Dice 0.92 was good enough to proceed; it is not evidence that nothing else is better.
