# Lesion → Disconnection → Outcome

Research pipeline: a brain lesion mask becomes a **disconnection** profile — which
white-matter connections it severs — which then predicts clinical outcome.

Design docs: [plan.md](plan.md) (what and why) · [ARCHITECTURE.md](ARCHITECTURE.md) (the five
modules) · [ROADMAP.md](ROADMAP.md) (phases, and what each one found).

> Research annotation / analysis tool. **Not a diagnostic device.**

## Status

Phase 0 complete apart from the ATLAS download, which is gated on a human step.
Bug audit complete — see [docs/BUGS.md](docs/BUGS.md).

## Two atlases, deliberately

- **`parcellation.nii.gz`, N=415** — the disconnection node set. Grey matter and deep nuclei,
  the standard connectome substrate. This defines the shape of every disconnection matrix.
- **`location_atlas.nii.gz`, N=463** — Model A's feature space, and *only* that. It adds
  lateralised fibre tracts, cerebellum, and a catch-all so brain coverage is 100%.

They are separate because reusing the node set as the location baseline made white-matter
lesions score an all-zero feature vector — which would have handed the Phase 3 thesis test a
win it hadn't earned. See [docs/BUGS.md](docs/BUGS.md) B1.

## Setup

```bash
py -3.13 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Run

```bash
cd scripts
../.venv/Scripts/python.exe 00_env_check.py           # versions + every atlas's grid
../.venv/Scripts/python.exe 01_build_parcellation.py  # -> data/atlases/  (N = 415)
../.venv/Scripts/python.exe 02_sanity_overlay.py      # -> data/derived/*.png
../.venv/Scripts/python.exe 03_fetch_arc.py           # ARC metadata + lesion masks
../.venv/Scripts/python.exe 04_template_gap_report.py # ATLAS template-gap residual
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
| `src/lesionconn/atlases.py` | the N=415 disconnection node set, and the N=463 location atlas |
| `src/lesionconn/lesions.py` | intake (with native-volume capture), synthetic lesions, parcel overlap |
| `src/lesionconn/qc.py` | cohort QC gate — exclusions are always recorded with a cited reason |
| `src/lesionconn/viz.py` | overlay renders |
| `data/` | gitignored — ATLAS is access-controlled, never commit it |
