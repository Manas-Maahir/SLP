# ATLAS — operational and technical notes

Sourced from <https://github.com/npnl/ATLAS/>, its issue tracker, the
[StrokePreprocessingPipeline](https://github.com/npnl/StrokePreprocessingPipeline) submodule,
and the INDI download page. Written 2026-08-30, before we hold the data.

## Release

**R3.0 is current** (1,453 T1w scans), superseding the R2.0 (955) the roadmap named.

| Archive | Size |
|---|---|
| `atlas3_training_raw.tar.gz` | 8.2 GB |
| `atlas3_training_preprocessed.tar.gz` | 14.8 GB |

Preprocessed lives at INDI; **raw native-space** data is archived separately on
[ICPSR](https://www.icpsr.umich.edu/web/ADDEP/studies/36684). The raw copy matters — it is
the escape hatch if we decide to redo registration ourselves (see *Registration* below).

## Getting the data

1. Google form → encryption key. **The key is shown on screen at submission.** Save it then:
   issues #26, #24, #22, #11 and #7 are all people who never received the promised email.
2. Decrypt with the flags the current page gives:

   ```
   openssl aes-256-cbc -md sha256 -d -a -in atlas3_training_preprocessed.tar.gz -out ATLAS_R3.0_preprocessed.tar.gz
   ```

   The `-md sha256` and `-a` (base64) flags matter. The long tail of decryption failures in
   the tracker (#25 "bad magic number", #12/#15/#6/#3/#2 "bad decrypt") mostly used `-md md5`
   without `-a`, from older instructions.
3. **Decrypt inside WSL2, not Windows.** Issue #25 reports Windows decryption failing where
   the same key worked on a Unix system. This — not FreeSurfer — is the first real use for the
   WSL2 box the roadmap put on standby.

## Known-bad subjects (build this into cohort QC)

| Subject | Problem | Issue |
|---|---|---|
| `r039s002` | mask not binary — max value 0.0099999 | #13 |
| `sub-r005s081` | empty file | #19 |
| `sub-r009s003` | mask does not match the visible lesion; possible mislabel | #21 |
| `sub-r040-s043` | microbleeds included in the lesion mask | #23 |

A commenter on #23 reports "4 out of 655 subjects failed our quick QC", so expect roughly this
rate rather than these four exactly.

`r039s002` is a live argument for how `lesions.load_lesion_mask` already binarizes: it
thresholds at `> 0`, so a mask whose maximum is 0.0099 survives intact. **A threshold at 0.5 —
the obvious choice — would silently produce an empty mask for that subject.** Do not change it.

## Registration — what ATLAS actually did

From `stroke_preproc.def`:

```
model="/share/mni-models/mni_icbm152_t1_tal_nlin_sym_09a.mnc"
model_label="MNI152NLin2009aSym"
bestlinreg.pl -clobber -lsq12 -nmi "${t1_mnc}" "${model}" "${transformation}"
mincresample -transformation ... -nearest_neighbour   # lesion mask
```

Three things follow, and two are caveats we have to carry:

1. **Target space is `MNI152NLin2009aSym`** (197×233×189 @ 1mm) — *not* the FSL MNI152 grid
   Schaefer and Harvard-Oxford use. See below.
2. **Registration is 12-DOF affine, not nonlinear.** "NLin" in the template name describes how
   the *template* was built, not the subject→template fit. Alignment is linear only.
3. **No cost-function masking and no enantiomorphic normalization.** The lesioned T1 is
   registered directly with normalized mutual information. This is exactly the failure mode
   pinned as caveat 2 in `plan.md` — *a lesion distorts its own registration* — and ATLAS's own
   pipeline does not mitigate it. "Pre-normalized" does not mean "well-normalized near the
   lesion", which is precisely where we need it to be right.

Useful: the pipeline saves the transform per subject as
`sub-X_ses-Y_desc-T1MNI152NLin2009aSym.xfm` (MINC format), and writes a per-subject QC PNG.

### File naming

```
sub-X_ses-Y_space-MNI152NLin2009aSym_desc-T1FinalResampledNorm.nii.gz
sub-X_ses-Y_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz
sub-X_ses-Y_space-orig_label-L_desc-T1lesion_mask.nii.gz        # native input
```

## The template gap, measured

Three different templates are in play, all called "MNI":

| Space | Used by | Grid @1mm |
|---|---|---|
| `MNI152NLin6Asym` (FSL MNI152) | Schaefer-400, Harvard-Oxford, **our canonical** | 182×218×182 |
| `MNI152NLin2009aSym` | **ATLAS preprocessed output** | 197×233×189 |
| `MNI152NLin2009cAsym` | `nilearn.load_mni152_template()` | 193×229×193 |

Measured against `MNI152NLin6Asym`, brain masks resampled by affine alone:

| | Dice | centre-of-mass shift |
|---|---|---|
| `MNI152NLin2009aSym` (ATLAS) | 0.968 | 0.88 mm |
| `MNI152NLin2009cAsym` (nilearn) | 0.968 | 1.28 mm |

**Read this carefully.** Gross alignment is good — under ~1 mm of global offset — so this is
*not* a catastrophic mismatch. The residual is local shape difference: ~3% of brain volume
disagrees, concentrated near boundaries. Against Schaefer parcels averaging 2,641 mm³ (a
~13.8 mm cube), a few mm of local error is a real but bounded source of parcel-assignment
noise, not a wholesale relabelling.

### The nonlinear warp was tested and rejected

TemplateFlow ships nonlinear transforms between these templates
(`tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5`, 205 MB). The obvious move
is to apply one instead of regridding by affine. **It does not work.** Applied via `antspyx`
0.6.3 it makes correspondence consistently *worse*:

| Configuration | T1 NCC vs FSL template | brain-mask Dice |
|---|---|---|
| affine regrid only | **0.782** | **0.968** |
| `tpl-NLin6Asym_from-2009cAsym` @ canonical 2mm | 0.661 | 0.943 |
| `tpl-2009cAsym_from-NLin6Asym` @ canonical 2mm | 0.662 | — |
| `tpl-NLin6Asym_from-2009cAsym` @ native 1mm | 0.661 | — |

Degrading equally in both directions and at both grids is the signature of a convention
mismatch rather than a wrong direction — but whatever the cause, the empirical answer stands.

Ruling out two false explanations along the way:

- Not a source-template mismatch: applying the warp to `2009cAsym` — the space it is built
  for — degrades just as much as applying it to ATLAS's `2009aSym`.
- Not a brain-mask artefact: the first version of this test scored brain masks, which encode
  each template's *mask definition* rather than anatomy. Rerunning on T1 intensity (NCC inside
  the FSL brain) gives the same verdict.

**Decision: regrid by affine and carry the residual.** ~1 mm global offset, Dice 0.968, NCC
0.782 — against Schaefer parcels averaging a ~13.8 mm cube, this is bounded parcel-assignment
noise, not wholesale relabelling. `scripts/04_template_gap_report.py` re-runs the whole
comparison so the decision stays falsifiable.

This affects ATLAS only. ARC masks are native-space and we own that registration, so the
Phase 3 thesis test is not exposed to this residual.

> Method note: an earlier estimate of this gap used the Harvard-Oxford subcortical union as a
> stand-in for an FSL brain mask. That excludes cerebellum while the ICBM mask includes it,
> manufacturing a spurious 7.6 mm inferior-posterior offset. The numbers above come from the
> real TemplateFlow masks and supersede it.
