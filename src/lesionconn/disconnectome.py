"""M3b -- tract severance. Which named white-matter bundles does a lesion cut?

The atlas is HCP-842 (Yeh et al.), 80 named tracts as a 4D stack of binary masks.
It ships (157, 189, 136, 80) at 1mm in **LPS**; our canonical grid is
(91, 109, 91) at 2mm in **LAS**. Both are "MNI". Merging them by array index --
the obvious implementation -- mirrors every tract across the midline, so a
left-hemisphere lesion reports right-hemisphere tracts and nothing downstream
notices. That exact bug (Harvard-Oxford RAS vs Schaefer LAS) already bit this
project once; see docs/BUGS.md and the Phase 0 notes in ROADMAP.md.

So laterality is not assumed here, it is asserted, twice and independently.
`assert_tract_laterality` encodes anatomy: a ``*_L`` tract's centre of mass is at
negative world x, most of its voxels are in its own hemisphere, and it lies left
of its ``*_R`` counterpart. `assert_matches_native` needs no anatomy at all --
each tract's centre of mass is measured in the source atlas and again after
resampling, and the two must agree to within a millimetre and a half. A build
that mirrors the atlas cannot get past either.

**What the numbers mean.** `frac_severed` is the fraction of an atlas tract's
*voxels* that the lesion overlaps. It is not the fraction of streamlines
disconnected: a small lesion at a bundle's narrow waist can disconnect a tract
almost entirely while overlapping few of its voxels, and this measure will call
that mild. Read it as "how much of this bundle's territory is destroyed", which
is a lower bound on functional disconnection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.affines import apply_affine

from . import config, deid
from .spaces import Grid, assert_canonical, to_canonical

# The atlas is not in this repo (it is large and belongs to another project), so
# its location is a parameter with a default rather than a vendored asset.
DEFAULT_TRACT_SOURCE = Path(
    r"C:/Users/manas/Desktop/Projects/Cophe-NeuroVue-1-bug-small-fixes"
    r"/Cophe-NeuroVue-1-bug-small-fixes/HCP824"
)
TRACT_VOLUME_NAME = "HCP842_tractography.nii.gz"
TRACT_NAMES_NAME = "HCP842_tractography.txt"

ATLAS_NAME = "HCP842_tractography"

# Below this many canonical voxels a tract is too thin for 2mm sampling to
# represent honestly -- the cranial nerves lose ~10% of their volume on the way
# down (measured, see docs/DISCONNECTION.md). Flagged, never silently dropped.
THIN_TRACT_VOXELS = 100

# Laterality thresholds, set from the measured atlas rather than from intuition.
# A first version of this check demanded that every lateralised tract sit >5mm
# off the midline; it failed on seven BRAINSTEM tracts (medial longitudinal
# fasciculus, central tegmental, rubrospinal, dorsal longitudinal) whose centres
# of mass genuinely sit 2-5mm out -- in the NATIVE atlas too, so the resampling
# had reproduced them faithfully and the threshold was the error. Measured over
# all 80 tracts:
#
#   lateralised tracts, voxels in their own hemisphere : min 0.676, median 1.000
#     (the minimum is Rubrospinal_Tract_L, which decussates -- real anatomy)
#   commissural tracts, |centre of mass x|             : max 4.29mm
#   L/R pairs, com_x(R) - com_x(L)                     : min 4.60mm, always > 0
#
# So the checks below are about SIGN and ORDER, not magnitude, which is what
# actually distinguishes a correct atlas from a mirrored one.
MIN_OWN_HEMISPHERE_FRACTION = 0.5
COMMISSURAL_MAX_MM = 5.0
# Set from the measured build, not guessed. Resampling 1mm -> 2mm shifts a tract's
# centre of mass by median 0.19mm, p95 0.91mm, max 3.94mm across the 80 tracts;
# the three largest shifts are all tracts of 28-83 canonical voxels, i.e. sampling
# noise going as 1/sqrt(N). A flipped y or z axis, by contrast, displaces a tract
# by tens of millimetres. 5mm sits above the former with headroom and an order of
# magnitude below the latter.
#
# It does NOT separate them for an x-flip of a near-midline brainstem tract, whose
# mirror image is only ~5mm away. That case is covered instead by
# assert_tract_laterality's pair-ordering rule, which a mirror always breaks.
NATIVE_COM_TOL_MM = 5.0


def _hemisphere(name: str) -> str:
    if name.endswith("_L"):
        return "L"
    if name.endswith("_R"):
        return "R"
    return "M"


def read_tract_names(source: str | Path = DEFAULT_TRACT_SOURCE) -> list[str]:
    """The 80 tract names, in atlas volume order.

    The file is ``<index>\\t<name>`` per line and at least one name carries
    trailing whitespace, which would otherwise become part of a label.
    """
    path = Path(source) / TRACT_NAMES_NAME
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        idx = int(parts[0].strip())
        name = parts[1].strip() if len(parts) > 1 else parts[0].strip()
        if idx != len(names):
            raise ValueError(f"{path}: names are not index-ordered at line {len(names) + 1}")
        names.append(name)
    if len(names) != config.EXPECTED_N_TRACTS:
        raise ValueError(f"expected {config.EXPECTED_N_TRACTS} tract names, got {len(names)}")
    return names


def _com_ijk(mask: np.ndarray) -> np.ndarray:
    """Centre of mass in voxel indices, via marginal sums.

    Marginals rather than ``argwhere``: the corpus callosum alone is half a
    million voxels, and this runs over 80 tracts on two grids.
    """
    total = mask.sum()
    if total == 0:
        return np.full(3, np.nan)
    return np.array([
        float((mask.sum(axis=tuple(a for a in range(3) if a != ax))
               * np.arange(mask.shape[ax])).sum() / total)
        for ax in range(3)
    ])


def centre_of_mass_world(mask: np.ndarray, grid: Grid) -> tuple[float, float, float]:
    """World-space centre of mass of a boolean volume. Empty -> (nan, nan, nan).

    An affine maps the centre of mass of the voxel set to the centre of mass of
    the world points, so this transforms one point rather than millions.
    """
    ijk = _com_ijk(mask)
    if np.isnan(ijk).any():
        return (float("nan"),) * 3
    xyz = apply_affine(np.asarray(grid.affine, dtype=float), ijk)
    return tuple(float(v) for v in xyz)


def _world_x_field(grid: Grid) -> np.ndarray:
    """World x of every voxel on ``grid``, as a volume."""
    ijk = np.stack(np.indices(grid.shape), axis=-1).reshape(-1, 3).astype(float)
    return apply_affine(np.asarray(grid.affine, dtype=float), ijk)[:, 0].reshape(grid.shape)


def assert_tract_laterality(data4d: np.ndarray, labels: pd.DataFrame, grid: Grid) -> None:
    """Refuse a tract atlas whose hemispheres are mirrored.

    This is the guardrail the module exists for. `assert_canonical` proves two
    volumes share a grid; it cannot prove an axis was not flipped during
    resampling, because a mirrored volume is still perfectly canonical.

    Three independent properties, any one of which a mirroring bug breaks:
      1. sign -- a ``*_L`` tract's centre of mass is at negative world x;
      2. bulk -- most of a lateralised tract's voxels are in its own hemisphere;
      3. order -- within every L/R pair, the left tract is left OF the right one.
    """
    xw = _world_x_field(grid)
    problems: list[str] = []
    com_x: dict[str, float] = {}

    for row in labels.itertuples():
        mask = data4d[..., row.index] > 0
        if not mask.any():
            problems.append(f"{row.name}: empty after resampling")
            continue
        x = centre_of_mass_world(mask, grid)[0]
        com_x[row.name] = x

        if row.hemisphere == "M":
            if abs(x) > COMMISSURAL_MAX_MM:
                problems.append(f"{row.name}: commissural tract off midline at x={x:+.1f}mm")
            continue

        want_negative = row.hemisphere == "L"
        if (x < 0) != want_negative:
            problems.append(
                f"{row.name}: {row.hemisphere} tract with centre of mass at x={x:+.1f}mm"
            )
        own = (xw[mask] < 0).mean() if want_negative else (xw[mask] > 0).mean()
        if own < MIN_OWN_HEMISPHERE_FRACTION:
            problems.append(
                f"{row.name}: only {own:.1%} of its voxels are in the "
                f"{row.hemisphere} hemisphere"
            )

    for name, x_left in com_x.items():
        if not name.endswith("_L"):
            continue
        x_right = com_x.get(name[:-2] + "_R")
        if x_right is not None and not x_left < x_right:
            problems.append(
                f"{name[:-2]}: left tract ({x_left:+.1f}mm) is not left of its right "
                f"counterpart ({x_right:+.1f}mm)"
            )

    if problems:
        raise AssertionError(
            "tract atlas laterality is wrong -- the resampling mirrored the brain.\n  "
            + "\n  ".join(problems[:12])
            + f"\n  ({len(problems)} problem(s) total.) Do not relax this check; "
            "the atlas is LPS and the canonical grid is LAS, and getting that "
            "wrong makes every disconnection profile confidently backwards."
        )


def assert_matches_native(
    canonical_com: dict[str, tuple[float, float, float]],
    native_com: dict[str, tuple[float, float, float]],
    tol_mm: float = NATIVE_COM_TOL_MM,
) -> None:
    """Every tract must land where it started. The direct anti-mirroring proof.

    The laterality checks encode anatomy; this one needs none. Each tract's
    centre of mass is measured in the source atlas and again after resampling,
    and the two must agree in world coordinates. A flipped axis negates x --
    tens of millimetres for a typical tract, against a tolerance under a voxel.
    """
    problems = []
    for name, canon in canonical_com.items():
        native = native_com.get(name)
        if native is None or np.isnan(canon).any() or np.isnan(native).any():
            continue
        delta = np.abs(np.asarray(canon) - np.asarray(native))
        if delta.max() > tol_mm:
            problems.append(
                f"{name}: centre of mass moved {delta.max():.1f}mm during resampling "
                f"(native {native[0]:+.1f},{native[1]:+.1f},{native[2]:+.1f} -> "
                f"canonical {canon[0]:+.1f},{canon[1]:+.1f},{canon[2]:+.1f})"
            )
    if problems:
        raise AssertionError(
            f"resampling moved {len(problems)} tract(s) more than {tol_mm}mm:\n  "
            + "\n  ".join(problems[:12])
        )


def build_tract_atlas(
    source: str | Path = DEFAULT_TRACT_SOURCE,
    verbose: bool = True,
    native_com_tol_mm: float = NATIVE_COM_TOL_MM,
):
    """Resample HCP-842's 80 tracts onto the canonical grid. LPS -> LAS.

    Returns a 4D uint8 image and a labels frame carrying, per tract, the volume
    before and after resampling -- the same discipline `intake_lesion_mask`
    applies to lesions (docs/BUGS.md B2). Nothing is corrected for; the cost is
    recorded so it can be read in the report.
    """
    source = Path(source)
    vol_path = source / TRACT_VOLUME_NAME
    if not vol_path.exists():
        raise FileNotFoundError(
            f"{vol_path} not found. Point --source at the directory holding "
            f"{TRACT_VOLUME_NAME} and {TRACT_NAMES_NAME}."
        )

    names = read_tract_names(source)
    img4d = nib.load(str(vol_path))
    if img4d.shape[3] != len(names):
        raise ValueError(f"atlas has {img4d.shape[3]} volumes but {len(names)} names")

    native_grid = Grid.of(img4d)
    canon = Grid.canonical()
    native_voxel_mm3 = native_grid.voxel_volume_mm3

    if verbose:
        print(f"  source        : {native_grid}")
        print(f"  canonical     : {canon}")
        print(f"  resampling {len(names)} tracts (nearest; they are binary masks)")

    out = np.zeros((*canon.shape, len(names)), dtype=np.uint8)
    rows = []
    native_com: dict[str, tuple[float, float, float]] = {}
    canonical_com: dict[str, tuple[float, float, float]] = {}
    for k, name in enumerate(names):
        native = np.asarray(img4d.dataobj[..., k])
        native_voxels = int(np.count_nonzero(native))

        vol = nib.Nifti1Image(native.astype(np.int16), img4d.affine)
        resampled = to_canonical(vol, interpolation="nearest")
        assert_canonical(resampled, f"tract {name}")

        arr = (np.asarray(resampled.dataobj) > 0).astype(np.uint8)
        out[..., k] = arr
        n_vox = int(arr.sum())

        # Where the tract sat before and after. Compared below; a mirrored axis
        # negates x, which is tens of millimetres for a typical tract.
        native_com[name] = centre_of_mass_world(native > 0, native_grid)
        canonical_com[name] = centre_of_mass_world(arr > 0, canon)
        shift = float(
            np.abs(np.asarray(canonical_com[name]) - np.asarray(native_com[name])).max()
        )

        native_mm3 = native_voxels * native_voxel_mm3
        canon_mm3 = n_vox * canon.voxel_volume_mm3
        rows.append({
            "index": k,
            "name": name,
            "hemisphere": _hemisphere(name),
            "n_voxels": n_vox,
            "volume_mm3": canon_mm3,
            "native_n_voxels": native_voxels,
            "native_volume_mm3": native_mm3,
            "volume_delta_pct": (
                100.0 * (canon_mm3 - native_mm3) / native_mm3 if native_mm3 else 0.0
            ),
            "thin": n_vox < THIN_TRACT_VOXELS,
            "com_x_native_mm": round(native_com[name][0], 2),
            "com_x_canonical_mm": round(canonical_com[name][0], 2),
            "com_shift_mm": round(shift, 2),
            "source_atlas": ATLAS_NAME,
        })
        if verbose and (k + 1) % 20 == 0:
            print(f"    {k + 1}/{len(names)}")

    labels = pd.DataFrame(rows)
    assert_tract_laterality(out, labels, canon)
    assert_matches_native(canonical_com, native_com, tol_mm=native_com_tol_mm)

    atlas_img = nib.Nifti1Image(out, np.asarray(canon.affine, dtype=float))
    atlas_img.set_data_dtype(np.uint8)

    if verbose:
        thin = labels[labels["thin"]]
        worst = labels.reindex(labels["volume_delta_pct"].abs().sort_values().index[::-1])
        moved = max(
            float(np.abs(np.asarray(canonical_com[n]) - np.asarray(native_com[n])).max())
            for n in canonical_com
        )
        print(f"  laterality asserted for all {len(labels)} tracts: OK")
        shifts = labels["com_shift_mm"].to_numpy()
        print(f"  centre-of-mass shift vs the native atlas: median "
              f"{np.median(shifts):.2f}mm  p95 {np.percentile(shifts, 95):.2f}mm  "
              f"max {moved:.2f}mm  (tolerance {native_com_tol_mm}mm)")
        for row in labels.sort_values("com_shift_mm", ascending=False).head(3).itertuples():
            print(f"    {row.name:34s} {row.com_shift_mm:5.2f}mm  "
                  f"({row.n_voxels} canonical voxels)")
        print(f"  thin at 2mm (<{THIN_TRACT_VOXELS} voxels): {len(thin)} "
              f"-> {list(thin['name'])[:6]}{' ...' if len(thin) > 6 else ''}")
        print("  largest volume change on resampling:")
        for row in worst.head(3).itertuples():
            print(f"    {row.name:34s} {row.volume_delta_pct:+6.1f}%")

    return atlas_img, labels


def save_tract_atlas(img, labels: pd.DataFrame) -> None:
    config.ensure_dirs()
    img.to_filename(config.TRACT_ATLAS_PATH)
    labels.to_csv(config.TRACT_LABELS_PATH, index=False)


def load_tract_atlas() -> tuple[nib.Nifti1Image, pd.DataFrame]:
    """The canonical-grid tract atlas (M3b's substrate)."""
    if not config.TRACT_ATLAS_PATH.exists():
        raise FileNotFoundError(
            f"{config.TRACT_ATLAS_PATH} not found -- run "
            "scripts/11_build_tract_atlas.py first"
        )
    img = nib.load(config.TRACT_ATLAS_PATH)
    labels = pd.read_csv(config.TRACT_LABELS_PATH)
    assert_canonical(img, "tract atlas")
    if img.shape[3] != len(labels):
        raise ValueError(f"tract atlas has {img.shape[3]} volumes but {len(labels)} labels")
    return img, labels


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------


@dataclass
class TractSeverance:
    """One lesion's severed-tract profile."""

    subject_id: str
    table: pd.DataFrame          # every tract, sorted by frac_severed desc
    lesion_voxels: int
    lesion_volume_mm3: float
    native_volume_mm3: float | None = None

    @property
    def hit(self) -> pd.DataFrame:
        return self.table[self.table["n_voxels_hit"] > 0]

    def vector(self) -> np.ndarray:
        """The 80-dim tract-severance feature vector, in atlas index order.

        The honest first baseline for M4: simpler than the parcel-pair matrix,
        and quite possibly the stronger feature object.
        """
        return (
            self.table.sort_values("index")["frac_severed"].to_numpy(dtype=np.float32)
        )


def severed_tracts(
    lesion_img,
    tract_img=None,
    tract_labels: pd.DataFrame | None = None,
    subject_id: str = "unknown",
    native_volume_mm3: float | None = None,
) -> TractSeverance:
    """Per-tract voxels hit and fraction severed, sorted worst-first.

    Both volumes are asserted canonical before they are combined -- that
    assertion is the only thing standing between this function and a plausible,
    silently mirrored answer.
    """
    if tract_img is None:
        tract_img, tract_labels = load_tract_atlas()
    if tract_labels is None:
        raise ValueError("tract_labels must accompany a caller-supplied tract_img")

    assert_canonical(lesion_img, "lesion")
    assert_canonical(tract_img, "tract atlas")

    lesion = np.asarray(lesion_img.dataobj) > 0
    tracts = np.asarray(tract_img.dataobj)

    # (n_lesion_voxels, n_tracts) -> per-tract hit counts in one pass.
    hits = tracts[lesion].sum(axis=0).astype(np.int64) if lesion.any() else np.zeros(
        tracts.shape[3], dtype=np.int64
    )

    voxel_mm3 = Grid.of(lesion_img).voxel_volume_mm3
    table = tract_labels.copy()
    table["n_voxels_hit"] = hits[table["index"].to_numpy()]
    table["volume_hit_mm3"] = table["n_voxels_hit"] * voxel_mm3
    total = table["n_voxels"].to_numpy()
    table["frac_severed"] = np.where(total > 0, table["n_voxels_hit"] / np.maximum(total, 1), 0.0)
    n_lesion = int(lesion.sum())
    table["frac_of_lesion"] = table["n_voxels_hit"] / n_lesion if n_lesion else 0.0

    table = table.sort_values(
        ["frac_severed", "n_voxels_hit"], ascending=False
    ).reset_index(drop=True)

    return TractSeverance(
        subject_id=subject_id,
        table=table,
        lesion_voxels=n_lesion,
        lesion_volume_mm3=float(n_lesion * voxel_mm3),
        native_volume_mm3=native_volume_mm3,
    )


def severed_tracts_payload(sev: TractSeverance, top_n: int | None = None) -> dict:
    """The `severed_tracts.json` contract from ARCHITECTURE.md (M3 -> M5)."""
    grid = Grid.canonical()
    rows = sev.hit if top_n is None else sev.hit.head(top_n)
    return {
        "schema": "lesionconn/severed_tracts/1",
        "subject_id": sev.subject_id,
        "atlas": {
            "name": ATLAS_NAME,
            "n_tracts": len(sev.table),
            "citation": "Yeh et al. (2018), HCP-842 population-averaged tractography atlas",
        },
        "space": {
            "shape": list(grid.shape),
            "voxel_mm": list(grid.voxel_sizes),
            "orientation": grid.orientation,
        },
        "lesion": {
            "n_voxels": sev.lesion_voxels,
            "volume_mm3": round(sev.lesion_volume_mm3, 1),
            "native_volume_mm3": (
                None if sev.native_volume_mm3 is None else round(sev.native_volume_mm3, 1)
            ),
        },
        "n_tracts_involved": len(sev.hit),
        "tracts": [
            {
                "index": int(r.index),
                "name": str(r.name),
                "hemisphere": str(r.hemisphere),
                "n_voxels_hit": int(r.n_voxels_hit),
                "volume_hit_mm3": round(float(r.volume_hit_mm3), 1),
                "tract_n_voxels": int(r.n_voxels),
                "frac_severed": round(float(r.frac_severed), 6),
                "frac_of_lesion": round(float(r.frac_of_lesion), 6),
                "thin_at_2mm": bool(r.thin),
            }
            for r in rows.itertuples()
        ],
        "measure": (
            "frac_severed is the fraction of the atlas tract's VOXELS overlapped by "
            "the lesion, not the fraction of streamlines disconnected. A small lesion "
            "at a bundle's waist can disconnect it almost completely while scoring low "
            "here. Treat it as a lower bound."
        ),
        "disclaimer": config.DISCLAIMER,
    }


def write_severed_tracts(sev: TractSeverance, path: str | Path) -> Path:
    """Serialise the payload, refusing to write anything carrying a patient name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(severed_tracts_payload(sev), indent=2)
    deid.guard_output(text, f"severed_tracts.json for {sev.subject_id}")
    path.write_text(text, encoding="utf-8")
    return path
