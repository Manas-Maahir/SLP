# Demo runbook

Live terminal, technical audience, ~15 minutes. Every command below was run end to end on
2026-09-20 and the timings are measured, not estimated.

The deck (`Lesion_Disconnection_Status.pptx`) and its speaking script
(`PRESENTATION.md`) carry the argument. This document covers only the live portion.

---

## Pre-flight (do this before they arrive)

```bash
cd scripts
../.venv/Scripts/python.exe 05_report.py sub-0017     # ~13s
../.venv/Scripts/python.exe 05_report.py sub-0058     # ~11s
```

Both subjects are now cached and their figures are on disk, so the live run is warm.
**Run these anyway during the demo** — they are fast enough and a live run is the point.

Open a file browser at `data/derived/reports/` so you can show figures without hunting.
Set the terminal to at least **80 columns**; the report is formatted to 78 and wraps ugly
below that.

---

## The run

### 1. The working case — `sub-0017`

```bash
../.venv/Scripts/python.exe 05_report.py sub-0017
```

A large left-hemisphere lesion whose top severed tract is the **arcuate fasciculus** — the
textbook language pathway. 21 tracts involved. Takes 13 seconds.

> "One binary mask goes in. What comes out is anatomy, which white-matter bundles it severs
> and by how much, two independent disconnection matrices, four figures, and a predicted
> language score."

**Point at the outcome block.** This is the part worth dwelling on:

```
  features                      predicted     90% interval   CV R2    MAE
  volume                             43.8          [9, 75]   0.424   17.2
  location                           50.9         [18, 83]   0.454   16.3
  network_edges_streamline           46.6         [17, 79]   0.519   15.0
```

> "Three arms, shown side by side on purpose. Lesion volume — one number, the size of the
> hole — gets you most of the way. The disconnection row is not a better answer, it is a
> third opinion. The report says so in its own text, because a single number here would
> imply a confidence the paired test doesn't support."

The three arms differ by 7.1 WAB-AQ points on this lesion, which is a good thing to say out
loud: it is smaller than the ~15-point typical error.

### 2. The refusal — `sub-0058`

```bash
../.venv/Scripts/python.exe 05_report.py sub-0058
```

**This is the moment to build the demo around.** Nearly a mirror image of the first: a large
lesion whose top severed tract is the arcuate fasciculus *of the right hemisphere*.

```
  NO PREDICTION -- this lesion is outside the training distribution:
    - only 35% of this lesion is in the LEFT hemisphere. The model was trained on
      ARC, which is 228/228 left-dominant aphasia...

  A number is deliberately not shown. The model would return one, and it
  would be meaningless.
```

> "Same anatomy, other side. The model was trained on 228 subjects, every one of them
> left-hemisphere dominant, and WAB-AQ is a language score. So it declines. It would happily
> return a number — that's the problem. 67 of our 108 lesions get refused."

If you show one thing, show this. Everything else on screen is a pipeline producing output;
this is the pipeline knowing what it doesn't know.

### 3. The figures

`data/derived/reports/sub-0017/` — four PNGs. Worth showing two:

- `sub-0017_severed_tracts.png` — bar chart. Note it plots **absolute severed volume beside
  the percentage**, because ranking by percentage alone structurally favours small tracts:
  across all 108 lesions, none of the six largest tracts was ever the top hit.
- `sub-0017_glass_brain.png` — the lesion in standard space.

Skip the disconnection matrix heatmap unless asked. It is dense and hard to read live.

---

## If something goes wrong

| symptom | do this |
|---|---|
| Report is slow (>60s) | It is rendering figures. Let it finish; it is not hung. |
| `manifest not found` | You are in the wrong directory. `cd scripts` first. |
| Output wraps badly | Terminal under 80 columns. Widen, re-run — it is 13 seconds. |
| Anything else | Fall back to `data/derived/reports/sub-0017/report.txt`, which is already on disk from pre-flight. `cat` it. |

Nothing in the demo path needs network access, a GPU, or the ARC raw data.

---

## Questions you will get, and the honest answers

**"So does disconnection work or not?"**
Not measurably better than knowing where the lesion is. Paired Wilcoxon over 25 identical
folds, Bonferroni corrected — nothing survives. And when lesion volume is regressed out of
the target fold-safe, every arm including the location baseline drops to ≤ 0.

**"Isn't that just a weak model?"**
No, and that's checked. The permutation test (slide 13) shuffles the outcome scores 200
times per model: every arm scores ~+0.43 to +0.51 real against ~−0.21 shuffled, p = 0.005,
the lowest the test can produce. The models work; they just don't beat each other.
The control — age, sex, time since stroke — correctly fails to reach significance.

**"Has anyone else found this?"**
Yes, and it's written up in `LITERATURE.md`. Salvalaggio 2020 (*Brain*, n=132) found
normative-connectome disconnection predicts "to a level comparable to lesion information".
Khalilian 2025 (n=340) found lesion masks outright *superior* to unthresholded disconnection
maps. Our absolute R² values also sit inside the published range.

**"Why not use the patient's own diffusion MRI?"**
That is the strongest objection and the answer is: we should, and we can't. ARC has no
per-subject tractography. Del Gaizo 2017 did have it and *did* beat lesion-symptom mapping —
which is evidence the idea works when you have the data, and that the normative
approximation is where the signal is lost.

**"Why only 228 subjects?"**
That is every labelled subject that exists in ARC. 245 participants, 228 with a lesion mask,
17 with no mask at all. The "902" figure in the dataset is scanning *sessions*, not people.
More subjects means a different cohort.

**"What would you do next?"**
Threshold the disconnection maps before using them as features. It is the one concrete thing
the literature points at, our own two matrices disagree at parcel level (ρ = −0.318) in a way
that says the same thing, and it is cheap. It is a better feature object, not a fix for n=228.

**"Can I trust the numbers on the slides?"**
They are read from the result CSVs by the build script, not typed. The build fails if any
shape runs off the canvas. And a de-identification scan greps every file this repo emits —
including the .pptx, which is a ZIP and was once invisible to it.

---

## What not to claim

- Not a diagnostic device. The report says so in its header, its footer, its JSON and every
  figure caption. Say it once out loud too.
- Disconnection did **not** win. The point estimates rank it first and the deck has a slide
  explaining why that ranking is a coin flip. Don't let a questioner talk you into the
  stronger claim.
- No hold-out test set exists. Every number is cross-validated on the same 228 subjects. If
  asked, say so plainly — it is in `PHASE3.md` as a known limitation, and 228 is too few to
  spend a fifth of on a holdout.
