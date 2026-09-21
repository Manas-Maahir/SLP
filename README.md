# Lesion → Disconnection → Outcome

Research pipeline: a brain lesion mask becomes a **disconnection** profile — which
white-matter connections it severs — which then predicts clinical outcome.

Design docs: [plan.md](plan.md) (what and why) · [ARCHITECTURE.md](ARCHITECTURE.md) (the five
modules) · [ROADMAP.md](ROADMAP.md) (phases, and what each one found) ·
[docs/DISCONNECTION.md](docs/DISCONNECTION.md) (**read before using the matrix**) ·
[docs/LITERATURE.md](docs/LITERATURE.md) (does the published field agree? — largely yes) ·
[docs/DEMO.md](docs/DEMO.md) (**runbook for showing this live**).

> Research annotation / analysis tool. **Not a diagnostic device.**

## Status

**Complete** (2026-09-21). **End to end and working.** A lesion mask goes in; a disconnection profile and a readable
report come out. Run over 108 real, de-identified lesions.

| | |
|---|---|
| M3b tract severance | 80 HCP-842 tracts, laterality asserted twice and shown failing on a mirrored atlas |
| M3a disconnection matrix | 415×415, **two of them**, from independent references — [docs/DISCONNECTION.md](docs/DISCONNECTION.md) |
| cohort | 108 lesions, 0 excluded by QC, all cached |
| M5 report | `scripts/05_report.py` → text + 4 figures |
| M1 registration | ARC T2w → MNI, cost-function masked, **Dice 0.924** — [docs/REGISTRATION.md](docs/REGISTRATION.md) |
| M4 outcome model | trained on ARC (n=228), persisted, wired into the report — **but B ≈ A**, and volume-residualised it is B ≈ A ≈ 0. See [docs/PHASE3.md](docs/PHASE3.md) |
| thresholding (Phase 5) | the one experiment the literature named, run against pre-committed predictions: **changes nothing**. Random edges score as well as selected ones. [docs/THRESHOLDING.md](docs/THRESHOLDING.md) |

The thesis test runs on **ARC** (228 subjects with a lesion mask *and* a WAB Aphasia
Quotient), not on the 108. Bug audit: [docs/BUGS.md](docs/BUGS.md).

> **Before building a model on the matrix, read this.** Four things are measured and each
> constrains what a model here can honestly claim.
>
> 1. The two matrices agree about which brain *systems* a lesion disconnects (median
>    Spearman +0.33) and anti-correlate about which individual *parcel* (−0.32). A single
>    parcel-pair edge is not a reliable feature.
> 2. **Lesion volume is recoverable from the disconnection features** at R² 0.47 (tracts)
>    to 0.66 (network edges), so beating the location baseline is not enough.
> 3. **Regress volume out of the target and nothing survives.** With a `hist_gb` nuisance
>    model fitted inside each training fold, every arm — tracts, both matrices, *and the
>    location baseline* — scores at or below zero on the volume-free residual, in all 18
>    (feature set × model) cells. `volume+X` against `volume` was the weak form of this
>    question; on trees it can score *below* `X` alone. Use `19_train.py --residualise volume`.
> 4. **Age, sex and days post-stroke add nothing** (R² −0.014 alone; they make `volume`
>    worse). ARC is chronic — median 721 days — so time post-stroke has partial r = −0.002.

## Three label volumes, deliberately kept apart

- **`parcellation.nii.gz`, N=415** — the disconnection node set. Grey matter and deep nuclei,
  the standard connectome substrate. This defines the shape of every disconnection matrix.
- **`location_atlas.nii.gz`, N=463** — Model A's feature space, and *only* that. It adds
  lateralised fibre tracts, cerebellum, and a catch-all so brain coverage is 100%.
- **`tract_atlas.nii.gz`, 80 tracts** — HCP-842, the white-matter bundles M3b intersects.
- **`streamline_index.npz`** — `S35_1mm.trk` reduced to endpoints and a voxel lookup. The
  counted matrix. Read in place from the NeuroVue project; nothing is copied here.

The first two are separate because reusing the node set as the location baseline made
white-matter lesions score an all-zero feature vector — which would have handed the Phase 3
thesis test a win it hadn't earned. See [docs/BUGS.md](docs/BUGS.md) B1.

That same gap bit the report: the N=415 node set has no cerebellum, so a report built on it
described a 70%-cerebellar lesion using the 18% of it that happened to clip occipital cortex.
The report now describes **anatomy** from the N=463 atlas and states how much of the lesion
the node set can see at all.

## Setup

```bash
py -3.13 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Run

Scripts `00`–`04` are Phase 0 setup; `10`–`13` build the cohort; `05` is the entry point a
human uses. **`10_deidentify.py` must run before anything else that touches the cohort.**

```bash
cd scripts
../.venv/Scripts/python.exe 00_env_check.py           # versions + every atlas's grid
../.venv/Scripts/python.exe 01_build_parcellation.py  # -> data/atlases/  (N = 415, N = 463)
../.venv/Scripts/python.exe 02_sanity_overlay.py      # -> data/derived/*.png
../.venv/Scripts/python.exe 03_fetch_arc.py           # ARC metadata + lesion masks
../.venv/Scripts/python.exe 04_template_gap_report.py # ATLAS template-gap residual

../.venv/Scripts/python.exe 10_deidentify.py          # FIRST. -> data/private/subject_map.csv
../.venv/Scripts/python.exe 11_build_tract_atlas.py   # HCP-842 80 tracts -> canonical grid
../.venv/Scripts/python.exe 12_tract_parcel_map.py    # which parcels each tract connects
../.venv/Scripts/python.exe 15_build_streamline_index.py  # 479k streamlines -> endpoints
../.venv/Scripts/python.exe 13_build_cohort.py        # all 108 -> manifest + features

../.venv/Scripts/python.exe 16_fetch_arc_anat.py      # 228 ARC T2w (~0.91 GB, targeted)
../.venv/Scripts/python.exe 17_register_arc.py --pilot 8   # LOOK at the output first
../.venv/Scripts/python.exe 17_register_arc.py --workers 5 # then the batch
../.venv/Scripts/python.exe 18_build_arc_cohort.py    # ARC -> manifest + features
../.venv/Scripts/python.exe 19_train.py --target volume            # harness check
../.venv/Scripts/python.exe 19_train.py --target wab_aq --cohort arc  # the thesis test
../.venv/Scripts/python.exe 19_train.py --target wab_aq --cohort arc --residualise volume
# the permutation null, on the family the headline numbers come from (107 min, resumable)
../.venv/Scripts/python.exe 19_train.py --target wab_aq --cohort arc --models hist_gb \
    --feature-sets location tracts network_edges network_edges_streamline covariates \
    --out-suffix nullrun --permutations 200 --permutation-model hist_gb
../.venv/Scripts/python.exe 20_fit_model.py           # fit + persist the final models

../.venv/Scripts/python.exe 05_report.py sub-0007     # report + figures for one subject
../.venv/Scripts/python.exe 14_export_neurovue.py     # optional: bundle for the NeuroVue viewer
../.venv/Scripts/python.exe 21_build_deck.py          # -> docs/Lesion_Disconnection_Status.pptx
cd ..
./.venv/Scripts/python.exe -m ruff check --select F,E7,E9,B,SIM,RUF --ignore E731 src/ scripts/ tests/
./.venv/Scripts/python.exe -m pytest tests/ -q
```

## The one thing to know

Everything lives on **one** grid: FSL MNI152 2mm, `(91, 109, 91)`, LAS.

"MNI space" is not a single space. nilearn's MNI152 template is ICBM 2009c `(99, 117, 95)`;
Schaefer and Harvard-Oxford ship on FSL MNI152 `(91, 109, 91)`; and Harvard-Oxford is
RAS-ordered where Schaefer is LAS. A lesion intersected with a misaligned parcellation yields
a disconnection matrix that looks entirely plausible and is quietly wrong.

So `lesionconn.spaces` makes that an exception rather than a judgement call. Any function
combining two volumes calls `assert_same_grid` or `assert_canonical` first. Do not remove
those calls to make something run.

## Layout

| Path | What |
|---|---|
| `src/lesionconn/spaces.py` | grid guardrail — `assert_same_grid`, `to_canonical` |
| `src/lesionconn/deid.py` | patient names never leave `data/private/` |
| `src/lesionconn/disconnectome.py` | M3 — tract severance and the *inferred* parcel-pair matrix |
| `src/lesionconn/streamlines.py` | M3a again — the *counted* matrix, from real fibre endpoints |
| `src/lesionconn/registration.py` | M1 — native T2w → MNI, cost-function masked |
| `src/lesionconn/modeling.py` | Phase 3 — nested CV, the volume control, the paired test |
| `src/lesionconn/predict.py` | M4 — the persisted model, and what it refuses to answer |
| `src/lesionconn/cohort.py` | the 108-subject run and its manifest |
| `src/lesionconn/report.py` | M5 — the text report and its four figures |
| `src/lesionconn/neurovue.py` | hand-off to the NeuroVue viewer (NeuroVue unmodified) |
| `src/lesionconn/atlases.py` | the N=415 disconnection node set, and the N=463 location atlas |
| `src/lesionconn/lesions.py` | intake (with native-volume capture), synthetic lesions, parcel overlap |
| `src/lesionconn/qc.py` | cohort QC gate — exclusions are always recorded with a cited reason |
| `src/lesionconn/viz.py` | overlay renders |
| `data/` | gitignored — ATLAS is access-controlled, never commit it |
| `data/private/` | **the only place a patient name may exist**, gitignored twice |

## The second thing to know

Patient identifiers live in `data/private/` and nowhere else. The cohort arrives with names
and MRNs in its filenames; everything downstream addresses a subject as `sub-NNNN`.

`tests/test_deid.py` enforces that by grepping **every** file this repo emits — including its
own source. It has caught a real patient name three times, twice in a comment written by the
author of the check. Do not narrow it without measuring first: the two narrowings that exist
(raw `.npy` buffers, PNG `IDAT`) were each made only after confirming the matches were
arithmetic coincidences, and each is bounded by a test that plants a name in the region still
scanned.
