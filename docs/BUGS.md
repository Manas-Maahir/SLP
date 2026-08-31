# Bug audit and fix plan

Audit run 2026-08-30 against Phase 0 code (31 tests passing at the time). Static analysis
(`ruff --select F,E7,E9,B,SIM,RUF`) found only cosmetic issues; everything below B7 was found
by probing behaviour, which is where this class of defect lives.

Ordered by blast radius, not by effort.

## Status: all resolved (B1-B9; see each entry for its verification)

| | bug | outcome |
|---|---|---|
| B1 | location baseline blind to white matter | fixed — location atlas, N=463 |
| B2 | small lesions gain 26% volume on intake | fixed — native volume captured at intake |
| B3 | canonical affine mutable | fixed — read-only |
| B4 | `to_grid` aliases its input | fixed — always returns a new image |
| B5 | no QC gate | fixed — `lesionconn/qc.py` |
| B6 | overlay renders label indices as brightness | fixed |
| B7 | 12 lint findings | fixed — clean, ruff added to verification |
| B8 | float32 NIfTI trips the de-id scan | fixed — QC image written as int16 |
| B9 | a ZIP container is invisible to the de-id scan | fixed — archive members decompressed |

Two things this audit got wrong before measuring, recorded because the reasoning matters more
than the fix:

- **The proposed B2 fix was wrong.** This document originally claimed linear interpolation
  "halves this error". Measured, linear and nearest give *identical* error to three decimal
  places — the error is voxel discretisation, not interpolation, so no resampling method
  touches it. Only resolution does.
- **B1 had a second instance.** Fixing white matter revealed that nothing labelled the
  cerebellum either, so cerebellar lesions were equally invisible.

---

## B1 — The location baseline is degenerate for white-matter lesions ★ critical · FIXED

**Evidence.** Only 61.1% of the MNI brain carries a parcel; **38.9% is unlabelled**. Schaefer
is a cortical ribbon and Harvard-Oxford covers deep nuclei only, so deep white matter and the
whole cerebellum have no label. Probing three synthetic 8 mm lesions:

| location | parcels hit | lesion voxels in any parcel | feature vector |
|---|---|---|---|
| left corona radiata | 0 | 0 / 257 (0.0%) | **all zeros** |
| left centrum semiovale | 0 | 0 / 257 (0.0%) | **all zeros** |
| left putamen (reference) | 3 | 183 / 257 (71.2%) | 3 non-zero |

**Why this is the worst bug here.** Corona radiata and centrum semiovale are among the most
common stroke locations. `location_feature_vector` returns all zeros for them, so Model A — the
location baseline — has *literally no information* about those subjects. The Phase 3 comparison
would then show "disconnection beats location" for a reason that has nothing to do with the
thesis: we'd be racing a real feature set against a broken one. And the bias runs in our own
favour, because white-matter lesions are exactly where disconnection is supposed to win.
ROADMAP.md already says it: *baseline or it didn't happen*.

**Fix.** The N=415 parcellation is not wrong as the M3 node set — cortical + subcortical nodes
are the standard connectome substrate, and it stays. The bug is reusing it, unmodified, as the
location baseline. So:

1. Add the **JHU white-matter atlas** as a second feature block for the location baseline only.
   `plan.md` already specifies it ("JHU for white matter"); Phase 0 simply didn't build it.
   `nilearn.datasets.fetch_atlas_juelich` is already available, and the FSL JHU atlases sit on
   the same grid as Harvard-Oxford.
2. Keep the disconnection node set at N=415, unchanged.
3. Regression test: assert that no plausible stroke location yields an all-zero location
   vector. Seed it with the two coordinates above, which currently fail.
4. Report per-subject coverage in the manifest, so a degenerate baseline is visible in Phase 2
   rather than discovered in Phase 3.

**Fix applied.** `atlases.build_location_atlas` emits a separate `location_atlas.nii.gz`,
**N = 463**, built strictly first-come so earlier blocks are never overwritten:

| block | labels | source |
|---|---|---|
| grey matter + deep nuclei | 415 | the unchanged disconnection node set |
| fibre tracts, split L/R | 20 | Juelich (its tracts are bilateral in one label, so a left vs right corticospinal lesion would otherwise tie) |
| cerebellum + vermis | 26 | AAL3, already lateralised |
| unassigned white matter L/R | 2 | catch-all, so no lesion can ever be invisible |

Brain coverage 61.1% → **100%**. JHU was the atlas `plan.md` named, but nilearn has no JHU
fetcher; Juelich carries real fibre-tract labels, sits on the same grid, and covers 97%/91% of
the two failing lesions while overlapping the grey-matter parcellation by only 2%.
`tests/test_location_baseline.py` pins both the fix and the original defect.

---

## B2 — Small lesions gain up to 26% volume on intake ★ high · FIXED

**Evidence.** ARC masks are native 1 mm; canonical is 2 mm. Round-tripping spheres through
`load_lesion_mask`:

| radius | native volume | canonical volume | error |
|---|---|---|---|
| 3 mm | 123 mm³ | 152 mm³ | **+23.6%** |
| 5 mm | 515 mm³ | 648 mm³ | **+25.8%** |
| 8 mm | 2,109 mm³ | 2,056 mm³ | −2.5% |
| 12 mm | 7,153 mm³ | 7,400 mm³ | +3.5% |
| 20 mm | 33,401 mm³ | 33,352 mm³ | −0.1% |

Nearest-neighbour downsampling claims or drops whole 8 mm³ voxels, which is a large relative
error when the lesion is only a few voxels across. Lesion volume is both a covariate and a
confound in outcome prediction, so a systematic +25% on small lesions is not cosmetic.

**What was measured, against the 1 mm source as ground truth:**

| radius | 2 mm nearest | 2 mm linear, threshold 0.5 | 1 mm nearest |
|---|---|---|---|
| 3 mm | +23.6% | **+23.6%** | +0.0% |
| 5 mm | +25.8% | **+25.8%** | +0.0% |
| 8 mm | −2.5% | −2.5% | +0.0% |
| 20 mm | −0.1% | −0.1% | +0.0% |

Linear interpolation changes nothing. The error is voxel discretisation, not interpolation.
Only resolution removes it, and 1 mm removes it exactly.

**Fix applied — stay at 2 mm, measure volume before resampling.** `lesions.intake_lesion_mask`
returns a `LesionIntake` carrying `native_volume_mm3` (measured on the source grid),
`canonical_volume_mm3`, and `volume_delta_pct`, warning above 10%. `qc.QCResult` carries both
through to the manifest. Anything volume-dependent uses the native figure; the 2 mm grid is
only ever the intersection substrate.

Rationale for not moving to 1 mm: the disconnection matrix is N×N regardless of resolution, so
1 mm buys fidelity only in the intersection step, at 7.9× compute and memory. ARC — the cohort
the Phase 3 thesis test actually runs on — is chronic aphasia, where lesions are large cortical
MCA territory and the 2 mm error is under 3%. `config.CANONICAL_*` is parameterised, so 1 mm
stays a reversible decision if a small-lesion cohort ever arrives.

---

## B3 — The canonical affine is mutable global state ★ high · FIXED

**Evidence.** `Grid.canonical()` hands out `config.CANONICAL_AFFINE` itself, not a copy.
Mutating `grid.affine[0, 3]` rewrites the module constant, corrupting every subsequent
`Grid.canonical()` in the process. `Grid` is `frozen=True`, which protects the *binding* and
does nothing for the array it points at.

The guardrail's own foundation being unprotected is exactly the sort of thing this codebase
claims to prevent elsewhere.

**Fix.** Set `CANONICAL_AFFINE.flags.writeable = False` in `config`, have `Grid.of` store a
copy, and add a test asserting the mutation attempt raises.

---

## B4 — `to_grid` aliases its input ★ medium · FIXED

**Evidence.** When the image is already on the target grid, `to_grid` returns *the same object*
(`test_to_grid_is_a_noop_when_already_aligned` currently asserts this identity). A caller that
mutates the "output" silently mutates the "input".

**Fix.** Decide the contract and enforce it. Recommended: always return a new image, and change
that test to assert equality of content rather than object identity.

---

## B5 — No QC gate for the bad masks ATLAS is known to ship ★ medium · FIXED

`docs/ATLAS-notes.md` lists subjects that are empty (`sub-r005s081`), non-binary
(`r039s002`), mislabelled (`sub-r009s003`), and one with microbleeds folded into the mask
(`sub-r040-s043`). Right now `load_lesion_mask` prints a warning for an empty mask and carries
on, so a bad subject would flow silently into the feature store.

**Fix.** A `lesionconn/qc.py` with an explicit exclusion list plus automated checks — empty
mask, implausible volume (outlier by MAD), excessive fraction outside the brain mask, mask
covering most of the brain. Every subject gets a pass/fail plus reason in `manifest.csv`.
Exclusions must be recorded, never silent.

---

## B6 — Overlay renders label indices as brightness ★ low · FIXED

`viz.plot_lesion_on_parcellation` passes the parcellation as `bg_img`, so the backdrop is
label *numbers* shown as grey levels — an arbitrary patchwork rather than anatomy
(visible in `data/derived/sanity_lesion_on_parcellation.png`). It confirms containment but is
useless for judging *which* parcel a lesion sits in.

**Fix.** Use the MNI template as background and draw the parcellation as a separate coloured
overlay.

---

## B7 — 12 static-analysis findings ★ cosmetic · FIXED

Unused imports across `scripts/`, `src/lesionconn/atlases.py`, `viz.py` and two test modules;
one `zip()` without `strict=`; three stale `# noqa` directives. No behavioural impact.

**Fix.** `ruff check --fix`, then add ruff to the verification sequence so this stays at zero.

---

## B8 — A float32 QC image trips the de-identification scan ★ medium · FIXED

`scripts/17_register_arc.py --pilot` saved each warped T2w as **float32**, and the repo-wide
de-identification scan then reported patient-token hits in all 8 of them.

**Measured before concluding anything.** Every match sat in the voxel payload, none in the
header, and all were 3–4 character fragments of cohort first names sitting inside float
mantissa bytes. **The fragments are deliberately not reproduced here** — this document is
itself scanned, and quoting a matched token turns a false positive into a real leak. That is
not hypothetical: the first draft of this entry listed all five and the scan rejected it,
which is the third time in this project that a name reached a file by way of an author
explaining a name. Re-encoding one volume without touching a single voxel:

| payload dtype | token matches |
|---|---|
| float32 | **26** |
| int16 | **0** |
| uint16 | 0 |

The raw ARC images are int16 and matched nothing across 210 MB when scanned earlier. The
coincidence was created entirely by the dtype this project chose to write, not by the data.
This is the same phenomenon already documented for `.npy` float32 buffers, where 3 of the 108
cohort matrices matched a three-letter name inside their mantissas.

**Fix — the artifact, not the check.** The warped T2w is a visual QC image whose source is
int16, so it is written as int16 (clipped, rounded). Matches go to zero and the files halve to
3.1 MB. `tests/test_deid.py` is **unchanged**: NIfTI payloads are still scanned in full, and a
name in a float32 NIfTI would still be caught.

**The class is not closed.** Any future float32 NIfTI — a probability map, a warp field, a
distance transform — will reproduce this. The options at that point are to narrow the NIfTI
scan to headers (the `.npy` precedent) or to keep eating false positives. That is a guardrail
decision, so it is recorded here rather than taken quietly.

---

## B9 — A ZIP container is invisible to the de-identification scan ★ high · FIXED

Introduced by this repo, not by the data: `docs/Lesion_Disconnection_Status.pptx` is the first
ZIP-format artifact the project emits. **A .pptx is a ZIP**, so `tests/test_deid.py`'s whole-repo
byte grep was reading deflate output — noise — and would have passed on a deck with a patient
name on every slide.

This is the same blind spot the `.npy` and PNG narrowings were measured against, with the sign
reversed. Those narrowings were deliberate decisions to scan *less* after measuring that the
excluded region produced only coincidences. This one was an accident that scanned *nothing*.

**Measured, before deciding anything.** Planting a name in a slide and grepping the archive:

| | name found by a raw byte grep |
|---|---|
| name in `ppt/slides/slide1.xml`, stored deflated | **no** |
| same file, after decompressing members | **yes** |

**Fix — the detector, because the detector was genuinely blind.** B8's rule was "fix the
artifact, not the detector", and it does not apply here: there is nothing wrong with the
artifact, and no way to write a PowerPoint file that is not a ZIP. `_zip_searchable` decompresses
every member and runs each one back through the per-format narrowings, so a PNG embedded in a
deck is treated exactly like a PNG on disk — `tEXt`/`zTXt`/`iTXt` searched, `IDAT` dropped as
B8 measured it should be. `.npz` is also a ZIP and is claimed by `_npz_searchable` first, so its
stricter handling is unchanged.

Two tests pin it: one plants a name in a deck slide and asserts the raw grep misses it while the
scan catches it, and one asserts `.npz` is still claimed by the narrower rule.

**Verified on the real artifact.** The deck's 75 archive members and 9 embedded images were
scanned. One raw hit appeared, in an `IDAT` stream of a matplotlib figure — B8's coincidence
class again, in a chunk the measured rule already excludes. Zero hits in any text-bearing region.

**This class now covers .docx, .xlsx and .zip too**, none of which the project emits yet.

---

## Suggested order

1. **B3, B4, B7** — small, local, no design decisions. Clears the deck.
2. **B1** — needs the JHU atlas wired in; blocks any honest Phase 3 comparison.
3. **B2** — needs the 1 mm-vs-2 mm decision first, because it determines what Phase 2 caches.
4. **B5** — naturally lands with the Phase 2 manifest work.

B1 and B2 both need settling **before** Phase 2 builds the feature store, or the cohort gets
cached in a form that has to be thrown away.
