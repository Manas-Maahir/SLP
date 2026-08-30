# The disconnection matrix: what it is, and what it is not

> **This is not a streamline-derived connectome.** It is an approximation built from
> tract *volumes*, and the approximation is doing real work. Read this before using
> `disconnection_matrix.npy` for anything, and cite the limits below wherever you report it.

## What we have, and what we would need

ARCHITECTURE.md M3a specifies the intended object:

> Take a normative structural connectome defined over those parcels (region×region edges,
> each edge backed by a bundle of streamlines). For each edge, compute how much of its
> streamline bundle passes through the lesion mask.

That needs **streamlines with endpoints** — a tractogram where each fibre is a polyline you
can test against the lesion and whose two ends land in identifiable parcels. HCP-842 ships as
80 **binary volumes**, one per named tract: a tract is a region of space, with no fibres
inside it and no endpoints.

> **Correction.** An earlier version of this document said "We do not have one." That was
> wrong, and it was the premise the whole approximation rested on. `S35_1mm.trk` — 479,457
> streamlines on the FSL MNI152 1mm grid, the same world box as our canonical space — sits in
> the NeuroVue project alongside the other assets. It is now used: see
> **[The streamline-derived matrix](#the-streamline-derived-matrix)** below. The
> approximation described here is still built and still emitted, because the two rest on
> different references and neither is the truth.

So the object below is derived, not measured. Both stages are approximations:

| Stage | Ideal | What we do |
|---|---|---|
| which parcels an edge joins | streamline endpoints | which parcels the tract **abuts** |
| how much of an edge is cut | fraction of streamlines intersected | fraction of tract **voxels** overlapped |

## The construction

**1. Tract severance (exact, given the atlas).** For each of the 80 tracts,
`f_t` = (tract voxels overlapped by the lesion) / (tract voxels). Sound as far as it goes —
see *Limit 1*.

**2. Tract → parcel affinity (the approximation).** Dilate each tract by one canonical voxel
(2 mm) and count its contact with each of the 415 parcels. Normalise by the tract's own
strongest contact:

```
a(t, p) = contact(t, p) / max_q contact(t, q)          a ∈ [0, 1]
```

Keep `a ≥ 0.15`. Normalising **per tract** rather than per parcel keeps the scale from being
a function of parcel size; the floor separates termination from passage.

**3. Parcel-pair disconnection.**

```
D[i, j] = max over tracts t of   f_t · a(t, i) · a(t, j)          i ≠ j
```

`max`, not a sum and not `1 - Π(1 - f_t)`: two tracts joining the same parcel pair are not
independent damage channels, and we have no basis for combining them. The strongest single
disruption is the defensible summary. Symmetric, zero diagonal, row *i* is parcel index *i+1*
in `parcellation_labels.csv`.

**Read an entry as: weighted evidence that the i–j connection is disrupted.** Not a severed
streamline fraction. It is not a probability and it is not calibrated.

## What the threshold buys, measured

`Corpus_Callosum` abuts **402 of the 415 parcels**. Treated as a binary "connects", a single
callosal lesion would mark ~80,600 parcel pairs — over 90% of the matrix — as disconnected
from that one tract. That is why the affinity is graded and floored.

For an 8 mm left corona radiata lesion (MNI −26, −16, 28):

| | value |
|---|---|
| nonzero pairs | 27,038 of 85,905 (31.5%) |
| median nonzero entry | 1.2% of the maximum |
| entries above 10% of the maximum | 244 (0.9% of nonzero) |
| entries above 50% of the maximum | 7 |
| density from `Corpus_Callosum` alone | 27.8 of the 31.5 points |
| largest entry `Corpus_Callosum` contributes | 10.4% of the top entry |

So the matrix is **dense but steeply graded**. The callosum supplies most of the nonzero
entries and almost none of the signal. Anything that reads the matrix should rank by value,
never count nonzeros — a "number of disconnected connections" statistic computed off this
matrix would be dominated entirely by one tract and would be meaningless.

After the threshold, 965 tract–parcel pairs survive: median 4 parcels per tract,
`Corpus_Callosum` still the widest at 226.

## Limits

**1. Voxel overlap is not streamline severance, and understates focal cuts.** `f_t` measures
how much of a bundle's *territory* is destroyed. A small lesion at a bundle's narrow waist can
disconnect it almost completely while overlapping few voxels, and `f_t` will call that mild.
Every number here is a lower bound on functional disconnection. This is why a 10 mm sphere in
the corona radiata reports the corticospinal tract as only ~7% severed: the CST runs from
cortex to brainstem, and a 20 mm-diameter sphere cannot overlap much of it — while clinically
such a lesion may be devastating.

**2. Abutment is not termination.** A tract running *past* a parcel is scored the same way as
one terminating in it. The affinity floor removes the weakest of these, but the arcuate
fasciculus still registers parcels it merely passes beneath. Terminations cannot be recovered
from a volume; they would need the tractogram.

**3. Normative wiring, not this patient's.** HCP-842 is a population average of healthy
brains. Nothing here reflects the individual's own connectivity, any pre-existing anomaly, or
post-stroke reorganisation.

**4. `max` discards convergent damage.** A parcel pair joined by two tracts, both half severed,
scores the same as one joined by a single half-severed tract.

**5. The 2 mm grid under-samples thin tracts.** 18 of 80 tracts fall below 100 canonical voxels;
the cranial nerves lose ~10% of their volume on the way down from 1 mm. Flagged as `thin` in
`tract_labels.csv` and in `severed_tracts.json`, never silently dropped.

**6. `Vermis` maps to no parcel at all.** The N=415 node set is cortex plus deep nuclei and
contains no cerebellar labels — the same coverage gap docs/BUGS.md B1 found for the location
baseline, which is why the N=463 location atlas exists separately. A purely vermian lesion
therefore produces an all-zero matrix while still producing a correct tract-severance vector.
Another argument for the vector being the primary feature object.

**7. `frac_severed` is normalised, so ranking by it favours small tracts.** Measured over the
108-lesion cohort, the tract with the highest `frac_severed` was:

| | voxels | times ranked top of 108 |
|---|---|---|
| `Vertical_Occipital_Fasciculus_R` | 633 | 15 |
| `Vertical_Occipital_Fasciculus_L` | 735 | 12 |
| `Middle_Longitudinal_Fasciculus_R` | 780 | 8 |
| **`Corpus_Callosum`** | **61,553** | **0** |
| **`Corticothalamic_Pathway_L`** | **20,711** | **0** |
| **`Cortico_Striatal_Pathway_L`** | **16,557** | **0** |

**None of the six largest tracts is ever the top hit for any subject**, though the corpus
callosum is involved in nearly every large lesion. This is not a defect in `frac_severed` —
"what fraction of this bundle is destroyed" is the right question — but it means the
highest-`frac_severed` tract is *not* the same thing as the most consequential
disconnection. Any report ranking by `frac_severed` must show absolute severed volume beside
it, and `scripts/05_report.py` does.

**8. Untested against ground truth.** No comparison has been made against a real
streamline-derived matrix (LQT, or an HCP tractogram). Until one is, treat the matrix as a
plausible construction, not a validated measurement.

## The streamline-derived matrix

`src/lesionconn/streamlines.py` builds the object this document's first section says we could
not: **counted**, not inferred. Each of `S35_1mm.trk`'s 479,457 fibres has its two ends
assigned to a parcel, and entry (i,j) is the fraction of edge i–j's fibres that pass through
the lesion.

```bash
cd scripts
../.venv/Scripts/python.exe 15_build_streamline_index.py   # one-time, ~5 min
```

It removes limits 1, 2 and 4 — voxel overlap standing in for streamline severance, abutment
standing in for termination, and `max` discarding convergent damage. It introduces two of
its own:

**S35 is one subject.** Nothing in NeuroVue calls it normative or population-averaged, and the
name reads as "subject 35". HCP-842 is explicitly averaged over 842 people. So this matrix
buys exact endpoints at the cost of a single individual's anatomy and tractography artifacts
standing in for average wiring. **That is a real downgrade in normativeness**, which is why
both matrices are emitted rather than one replacing the other.

**Endpoint assignment needs a radius.** Tractography terminates at the grey/white boundary,
not inside the cortical ribbon, so only 50.8% of endpoints land directly in a parcel. Assigning
each to the nearest parcel within a radius:

| radius | endpoints assigned | streamlines with both ends assigned |
|---|---|---|
| 0 mm | 50.8% | 22.3% |
| 2 mm | 61.5% | 35.2% |
| **4 mm** | **73.4%** | **53.4%** |
| 6 mm | 85.2% | 74.3% |
| 10 mm | 94.9% | 92.5% |

The curve is smooth — there is no elbow to read a threshold from, so 4 mm is a judgement call
recorded as one. It is two canonical voxels, comparable to cortical thickness, and matches
MRtrix `tck2connectome`'s radial-search default. Past ~6 mm an endpoint can cross a sulcus into
a neighbouring gyrus, which is a different parcel and a wrong edge.

**Edges need a fibre floor before they are quoted.** 6,575 parcel pairs carry at least one
streamline (7.7% density — a real connectome is sparse, against 31.5% for the inferred matrix),
but median support is only **9 fibres**. An edge carried by one fibre reads "100% severed" the
moment a lesion clips it, so an unranked list is led by its own weakest evidence — the same
failure as limit 7, a normalised score with a tiny denominator. Reporting therefore filters to
edges with ≥10 fibres and always prints the count beside the fraction. **The matrix itself is
never filtered**; M4 receives everything.

### The two matrices disagree, and where they disagree is the finding

Measured across all 108 lesions, comparing the inferred (HCP-842) matrix against the counted
(S35) one:

| | median Spearman ρ | subjects positive |
|---|---|---|
| **parcel level** (415 nodes) | **−0.318** | 4 / 108 |
| **network level** (17 hemisphere × network groups) | **+0.331** | 104 / 107 |

The sign flips. Two methods sharing no data, no reference and no algorithm **agree about which
brain systems a lesion disconnects and contradict each other about which individual parcel
inside them**.

Three hypotheses were tested and rejected before this one:

- *normalisation* — correlation between edge fibre-count and each matrix's value is +0.045 and
  −0.029, far too small to drive it;
- *fraction vs count* — comparing the inferred value against raw severed-fibre counts instead
  of fractions still gives −0.292;
- *laterality or a space bug* — both matrices are correctly lateralised, and mirrored lesions
  produce mirrored matrices in each.

What survives is granularity. Schaefer-400 parcels average ~2,600 mm³, and a 4 mm endpoint
search and a voxel-adjacency inference distribute damage across *neighbouring* parcels within
the same functional region differently. On the synthetic left corona radiata probe both methods
name motor cortex ↔ brainstem — the inferred one picks `SomMot_35` and `SomMot_30`, the counted
one picks `SomMot_37` (963 fibres, 100% cut) and `SomMot_34`. Same anatomy, different parcel,
and at that resolution the choice between adjacent parcels is close to arbitrary. Competing for
rank among interchangeable neighbours produces exactly this: agreement above, anti-correlation
below.

**Consequence for M4, and it is not a small one.** A single parcel-pair edge in either matrix
is not a reliable feature. The reproducible signal is at network resolution. Before training on
415×415 edges, either coarsen to networks, or restrict to edges where the two matrices agree,
and expect the parcel-level feature space to be substantially noise. This does not say either
matrix is *right* — two methods can agree and both be wrong — but their disagreement is strong
evidence that parcel-level edges are not resolvable from this data.

This supersedes limit 8 ("untested against ground truth"). It is now tested against an
independent construction, with a specific and unflattering answer.

## What to use for what

| Purpose | Use | Why |
|---|---|---|
| M4 features (primary) | the 80-dim tract-severance vector | one approximation, not two; nothing derived |
| M4 features (secondary) | either matrix **coarsened to networks** | parcel-level edges anti-correlate between methods |
| M4 features (speculative) | flattened upper triangle, full resolution | the ARCHITECTURE contract, but expect noise |
| M5 report | severed tracts, then both matrices' top pairs | tracts are named anatomy a clinician can check |
| a single parcel-pair claim | neither, alone | the two methods disagree there |
| counting anything | neither | see the density table above |

The **80-dim tract vector is the primary feature object**. It rests on step 1 alone, is simpler,
and may well predict better. `disconnection_matrix.npy` is emitted because ARCHITECTURE.md's
data contract specifies it and because M4 should get the chance to falsify the vector's
advantage — not because it is the more trustworthy of the two.

## Reproducing / changing this

```bash
cd scripts
../.venv/Scripts/python.exe 11_build_tract_atlas.py    # 80 tracts -> canonical grid
../.venv/Scripts/python.exe 12_tract_parcel_map.py     # --min-affinity to change the floor
```

`disconnectome.MIN_AFFINITY` and `CONTACT_DILATION_VOXELS` are the knobs that change what the
inferred matrix means; `streamlines.ENDPOINT_SEARCH_MM` and `MIN_EDGE_STREAMLINES` are the
equivalents for the counted one. Changing either invalidates every cached matrix under `data/derived/`, and
the numbers in this document with it.
