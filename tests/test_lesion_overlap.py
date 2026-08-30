"""A lesion at a known coordinate must land in the known parcels.

This is the end-to-end alignment test. If a left-putamen sphere reports
right-hemisphere or cortical parcels, the affine handling is wrong and every
disconnection matrix built later would be silently corrupt.
"""

import numpy as np
import pytest

from lesionconn.atlases import build_parcellation
from lesionconn.lesions import (
    lesion_parcel_overlap, lesion_volume_mm3, load_lesion_mask,
    location_feature_vector, make_synthetic_lesion,
)
from lesionconn.spaces import assert_canonical

LEFT_PUTAMEN = (-26.0, 0.0, 8.0)
RIGHT_PUTAMEN = (26.0, 0.0, 8.0)


@pytest.fixture(scope="module")
def parcellation():
    return build_parcellation(verbose=False)


def test_synthetic_lesion_is_canonical_and_binary():
    les = make_synthetic_lesion(LEFT_PUTAMEN, 10.0)
    assert_canonical(les)
    assert set(np.unique(np.asarray(les.dataobj)).tolist()) <= {0, 1}


def test_synthetic_lesion_volume_matches_sphere():
    r = 10.0
    les = make_synthetic_lesion(LEFT_PUTAMEN, r)
    expected = 4 / 3 * np.pi * r**3
    assert lesion_volume_mm3(les) == pytest.approx(expected, rel=0.10)


def test_lesion_lands_in_left_hemisphere(parcellation):
    parc, labels = parcellation
    overlap = lesion_parcel_overlap(make_synthetic_lesion(LEFT_PUTAMEN, 10.0), parc, labels)
    assert not overlap.empty
    top = overlap.head(5)["hemisphere"].tolist()
    assert "RH" not in top, f"left lesion hit right-hemisphere parcels: {top}"


def test_lesion_hits_expected_subcortical_structures(parcellation):
    parc, labels = parcellation
    overlap = lesion_parcel_overlap(make_synthetic_lesion(LEFT_PUTAMEN, 10.0), parc, labels)
    names = " ".join(overlap.head(6)["name"].str.lower())
    assert any(s in names for s in ("putamen", "pallidum", "insula", "caudate")), \
        f"unexpected anatomy at MNI {LEFT_PUTAMEN}: {overlap.head(6)['name'].tolist()}"


def test_left_and_right_lesions_are_mirror_images(parcellation):
    """Catches an L/R flip, which no volume or count check would reveal."""
    parc, labels = parcellation
    left = lesion_parcel_overlap(make_synthetic_lesion(LEFT_PUTAMEN, 10.0), parc, labels)
    right = lesion_parcel_overlap(make_synthetic_lesion(RIGHT_PUTAMEN, 10.0), parc, labels)
    assert "RH" not in left.head(5)["hemisphere"].tolist()
    assert "LH" not in right.head(5)["hemisphere"].tolist()


def test_fractions_are_well_formed(parcellation):
    parc, labels = parcellation
    overlap = lesion_parcel_overlap(make_synthetic_lesion(LEFT_PUTAMEN, 10.0), parc, labels)
    assert (overlap["frac_of_parcel"].between(0, 1)).all()
    assert (overlap["frac_of_lesion"].between(0, 1)).all()
    # frac_of_lesion sums to <= 1; the remainder is lesion outside any parcel.
    assert overlap["frac_of_lesion"].sum() <= 1.0 + 1e-6


def test_location_feature_vector_is_fixed_length(parcellation):
    """Model A's feature vector must be the same length for every subject."""
    parc, labels = parcellation
    a = location_feature_vector(make_synthetic_lesion(LEFT_PUTAMEN, 10.0), parc, labels)
    b = location_feature_vector(make_synthetic_lesion(RIGHT_PUTAMEN, 6.0), parc, labels)
    assert a.shape == b.shape == (len(labels),)
    assert a.dtype == np.float32
    assert not np.array_equal(a, b)


def test_lesion_outside_the_brain_overlaps_nothing(parcellation):
    parc, labels = parcellation
    far = make_synthetic_lesion((0.0, 0.0, 120.0), 5.0)
    assert lesion_parcel_overlap(far, parc, labels).empty


def test_load_lesion_mask_binarizes_and_aligns(tmp_path):
    """A mask arriving on a foreign grid gets resampled, not silently accepted."""
    import nibabel as nib

    odd_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    data = np.zeros((60, 60, 60), np.float32)
    data[28:34, 28:34, 28:34] = 7.5  # non-binary values on purpose
    path = tmp_path / "mask.nii.gz"
    nib.Nifti1Image(data, odd_affine).to_filename(path)

    loaded = load_lesion_mask(path)
    assert_canonical(loaded)
    assert set(np.unique(np.asarray(loaded.dataobj)).tolist()) <= {0, 1}


def test_intake_records_native_volume_not_the_resampled_one(tmp_path):
    """B2: 2mm discretisation inflates small lesions; native volume is the truth.

    No interpolation method fixes this -- linear and nearest give identical
    error, because it is voxel discretisation rather than interpolation.
    """
    import nibabel as nib
    from nibabel.affines import apply_affine

    from lesionconn.lesions import intake_lesion_mask

    affine = np.eye(4)
    affine[0, 3], affine[1, 3], affine[2, 3] = -90, -126, -72   # 1mm, like ARC
    shape = (180, 220, 180)
    ijk = np.stack(np.indices(shape), -1).reshape(-1, 3).astype(float)
    xyz = apply_affine(affine, ijk)
    sphere = (np.linalg.norm(xyz - np.array([-26.0, 0.0, 8.0]), axis=1) <= 3.0).reshape(shape)

    path = tmp_path / "sub-tiny_mask.nii.gz"
    nib.Nifti1Image(sphere.astype(np.int16), affine).to_filename(path)

    rec = intake_lesion_mask(path, quiet=True)
    truth = float(sphere.sum())          # 1mm^3 voxels

    assert rec.native_volume_mm3 == pytest.approx(truth)
    assert rec.canonical_volume_mm3 != pytest.approx(truth, rel=0.05)
    assert rec.volume_delta_pct > 10.0
    assert rec.subject_id == "sub-tiny"
    assert "1x1x1mm" in rec.native_grid


def test_large_lesion_survives_resampling_intact(tmp_path):
    import nibabel as nib
    from nibabel.affines import apply_affine

    from lesionconn.lesions import intake_lesion_mask

    affine = np.eye(4)
    affine[0, 3], affine[1, 3], affine[2, 3] = -90, -126, -72
    shape = (180, 220, 180)
    ijk = np.stack(np.indices(shape), -1).reshape(-1, 3).astype(float)
    xyz = apply_affine(affine, ijk)
    sphere = (np.linalg.norm(xyz - np.array([-26.0, 0.0, 8.0]), axis=1) <= 20.0).reshape(shape)

    path = tmp_path / "sub-big_mask.nii.gz"
    nib.Nifti1Image(sphere.astype(np.int16), affine).to_filename(path)

    rec = intake_lesion_mask(path, quiet=True)
    assert abs(rec.volume_delta_pct) < 2.0
