"""Lesion masks: loading, synthesis, and parcel overlap.

`lesion_parcel_overlap` doubles as the location-baseline feature extractor that
Phase 2 needs for the benchmark. It is built here in Phase 0 because it is also
the only way to machine-check that a lesion and the parcellation are truly
aligned -- a lesion placed at a known coordinate must land in known parcels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.affines import apply_affine
from nilearn.image import new_img_like

from .spaces import Grid, assert_canonical, to_canonical


def make_synthetic_lesion(
    center_mni: tuple[float, float, float],
    radius_mm: float = 10.0,
    grid: Grid | None = None,
) -> nib.Nifti1Image:
    """A spherical binary mask centred on an MNI world coordinate.

    Stands in for a real lesion while ATLAS access is pending, and gives the
    alignment tests a ground truth they can assert against.
    """
    grid = grid or Grid.canonical()
    ijk = np.stack(np.indices(grid.shape), axis=-1).reshape(-1, 3).astype(float)
    xyz = apply_affine(grid.affine, ijk)
    inside = np.linalg.norm(xyz - np.asarray(center_mni, dtype=float), axis=1) <= radius_mm
    data = inside.reshape(grid.shape).astype(np.int16)

    img = nib.Nifti1Image(data, grid.affine)
    img.set_data_dtype(np.int16)
    return img


VOLUME_DELTA_WARN_PCT = 10.0


@dataclass
class LesionIntake:
    """What happened to a mask on its way onto the canonical grid.

    `native_volume_mm3` is measured BEFORE resampling and is the number to trust
    wherever lesion volume is used as a feature or covariate. Resampling a 1mm
    mask onto the 2mm canonical grid inflates small lesions by up to 26%, and no
    interpolation method fixes that -- linear and nearest give identical error,
    because it is voxel discretisation, not interpolation (docs/BUGS.md B2).
    """

    img: object
    subject_id: str
    native_volume_mm3: float
    canonical_volume_mm3: float
    native_grid: str
    n_distinct_values: int

    @property
    def volume_delta_pct(self) -> float:
        if self.native_volume_mm3 == 0:
            return 0.0
        return 100.0 * (self.canonical_volume_mm3 - self.native_volume_mm3) / self.native_volume_mm3


def intake_lesion_mask(
    path: str | Path,
    subject_id: str | None = None,
    warn_pct: float = VOLUME_DELTA_WARN_PCT,
    quiet: bool = False,
) -> LesionIntake:
    """Load a mask onto the canonical grid, recording what resampling cost it."""
    path = Path(path)
    subject_id = subject_id or path.name.split("_")[0]

    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    n_unique = len(np.unique(data))
    native_grid = Grid.of(img)

    if n_unique > 2 and not quiet:
        print(f"  note: {path.name} has {n_unique} distinct values; binarizing at > 0")

    binary = new_img_like(img, (data > 0).astype(np.int16), copy_header=True)
    binary.set_data_dtype(np.int16)
    native_volume = float(np.count_nonzero(data > 0) * native_grid.voxel_volume_mm3)

    canonical = to_canonical(binary, interpolation="nearest")
    assert_canonical(canonical, path.name)
    canonical_volume = float(
        np.count_nonzero(np.asarray(canonical.dataobj)) * Grid.canonical().voxel_volume_mm3
    )

    rec = LesionIntake(
        img=canonical, subject_id=str(subject_id),
        native_volume_mm3=native_volume, canonical_volume_mm3=canonical_volume,
        native_grid=str(native_grid), n_distinct_values=n_unique,
    )

    if not quiet:
        if canonical_volume == 0:
            print(f"  WARNING: {path.name} is empty after binarizing")
        elif abs(rec.volume_delta_pct) > warn_pct:
            print(
                f"  WARNING: {path.name} volume changed {rec.volume_delta_pct:+.1f}% on "
                f"resampling ({native_volume:.0f} -> {canonical_volume:.0f} mm3). "
                "Use native_volume_mm3 for anything volume-dependent."
            )
    return rec


def load_lesion_mask(path: str | Path, resample: bool = True) -> nib.Nifti1Image:
    """Load a lesion mask, binarize it, and put it on the canonical grid.

    Thin wrapper over `intake_lesion_mask`; use that when you need the native
    volume, which is most places lesion size actually matters.
    """
    if not resample:
        img = nib.load(str(path))
        data = np.asarray(img.dataobj)
        binary = new_img_like(img, (data > 0).astype(np.int16), copy_header=True)
        binary.set_data_dtype(np.int16)
        return binary
    return intake_lesion_mask(path).img


def lesion_volume_mm3(lesion_img) -> float:
    grid = Grid.of(lesion_img)
    return float(np.count_nonzero(np.asarray(lesion_img.dataobj)) * grid.voxel_volume_mm3)


def lesion_parcel_overlap(
    lesion_img,
    parc_img,
    labels: pd.DataFrame,
    include_zero: bool = False,
) -> pd.DataFrame:
    """Per-parcel overlap between a lesion and the parcellation.

    Returns one row per parcel with the raw voxel count, volume, the fraction of
    that parcel destroyed, and the fraction of the lesion it accounts for.
    """
    assert_canonical(lesion_img, "lesion")
    assert_canonical(parc_img, "parcellation")

    lesion = np.asarray(lesion_img.dataobj) > 0
    parc = np.asarray(parc_img.dataobj).astype(np.int64)
    n_labels = int(labels["index"].max())

    parcel_sizes = np.bincount(parc.reshape(-1), minlength=n_labels + 1)
    hit_counts = np.bincount(parc[lesion], minlength=n_labels + 1)

    voxel_mm3 = Grid.of(lesion_img).voxel_volume_mm3
    lesion_voxels = int(lesion.sum())

    out = labels.copy()
    idx = out["index"].to_numpy()
    out["n_voxels"] = hit_counts[idx]
    out["volume_mm3"] = out["n_voxels"] * voxel_mm3
    with np.errstate(divide="ignore", invalid="ignore"):
        out["frac_of_parcel"] = np.where(
            parcel_sizes[idx] > 0, hit_counts[idx] / np.maximum(parcel_sizes[idx], 1), 0.0
        )
    out["frac_of_lesion"] = out["n_voxels"] / lesion_voxels if lesion_voxels else 0.0

    if not include_zero:
        out = out[out["n_voxels"] > 0]
    return out.sort_values("n_voxels", ascending=False).reset_index(drop=True)


def location_feature_vector(lesion_img, atlas_img, labels: pd.DataFrame) -> np.ndarray:
    """Fixed-length location baseline: fraction of each region lesioned.

    Model A's feature vector in the Phase 3 benchmark -- the thing the
    disconnection features have to beat.

    Pass the **location atlas** (`atlases.load_location_atlas`), not the
    disconnection parcellation. The parcellation is grey matter only, so a white
    matter lesion scores all zeros through it and the baseline silently has no
    information about the subject (docs/BUGS.md B1).
    """
    overlap = lesion_parcel_overlap(lesion_img, atlas_img, labels, include_zero=True)
    return overlap.sort_values("index")["frac_of_parcel"].to_numpy(dtype=np.float32)


def location_features(lesion_img) -> np.ndarray:
    """`location_feature_vector` against the location atlas loaded from disk."""
    from .atlases import load_location_atlas

    atlas_img, labels = load_location_atlas()
    return location_feature_vector(lesion_img, atlas_img, labels)


def unlesioned_voxels_outside_brain(lesion_img, brain_mask_img) -> int:
    """Count lesion voxels falling outside the brain -- a registration smell test."""
    lesion = np.asarray(lesion_img.dataobj) > 0
    brain = np.asarray(brain_mask_img.dataobj) > 0
    return int(np.count_nonzero(lesion & ~brain))
