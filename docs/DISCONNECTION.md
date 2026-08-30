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
can test against the lesion and whose two ends land in identifiable parcels. We do not have
one. HCP-842 ships as 80 **binary volumes**, one per named tract: a tract is a region of
space, with no fibres inside it and no endpoints.

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

**7. Untested against ground truth.** No comparison has been made against a real
streamline-derived matrix (LQT, or an HCP tractogram). Until one is, treat the matrix as a
plausible construction, not a validated measurement.

## What to use for what

| Purpose | Use | Why |
|---|---|---|
| M4 features (primary) | the 80-dim tract-severance vector | one approximation, not two; nothing derived |
| M4 features (secondary) | flattened upper triangle of `D` | the ARCHITECTURE contract; rank by value |
| M5 report | severed tracts, then top `D` pairs | tracts are named anatomy a clinician can check |
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

`disconnectome.MIN_AFFINITY` and `CONTACT_DILATION_VOXELS` are the two knobs that change what
the matrix means. Changing either invalidates every cached matrix under `data/derived/`, and
the numbers in this document with it.
