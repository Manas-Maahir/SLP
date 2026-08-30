"""The guardrail must actually fire. A check that never fails is not a check."""

import nibabel as nib
import numpy as np
import pytest

from lesionconn.spaces import (
    Grid, GridMismatch, assert_canonical, assert_same_grid,
    is_integer_like, to_canonical, to_grid,
)


def _img(shape=None, affine=None, dtype=np.int16):
    canon = Grid.canonical()
    shape = shape or canon.shape
    affine = canon.affine if affine is None else affine
    return nib.Nifti1Image(np.zeros(shape, dtype), affine)


def test_canonical_grid_properties():
    g = Grid.canonical()
    assert g.shape == (91, 109, 91)
    assert g.voxel_sizes == pytest.approx((2.0, 2.0, 2.0))
    assert g.voxel_volume_mm3 == pytest.approx(8.0)
    assert g.orientation == "LAS"


def test_identical_grids_pass():
    assert_same_grid(_img(), _img())
    assert_canonical(_img())


def test_one_voxel_shift_is_caught():
    shifted = Grid.canonical().affine.copy()
    shifted[0, 3] += 2.0
    with pytest.raises(GridMismatch, match="affine"):
        assert_same_grid(_img(), _img(affine=shifted))


def test_subvoxel_shift_is_caught():
    """Even a 0.1mm offset matters -- it is enough to mislabel a lesion border."""
    shifted = Grid.canonical().affine.copy()
    shifted[1, 3] += 0.1
    with pytest.raises(GridMismatch):
        assert_same_grid(_img(), _img(affine=shifted))


def test_shape_mismatch_is_caught():
    with pytest.raises(GridMismatch, match="shape"):
        assert_same_grid(_img(), _img(shape=(91, 109, 90)))


def test_flipped_orientation_is_caught():
    """An L/R flip is the most dangerous error in lesion work: it looks normal."""
    flipped = Grid.canonical().affine.copy()
    flipped[0, 0] *= -1
    with pytest.raises(GridMismatch):
        assert_same_grid(_img(), _img(affine=flipped))


def test_nilearn_template_is_not_on_canonical_grid():
    """Documents the exact trap this module exists for."""
    from nilearn import datasets

    tpl = datasets.load_mni152_template(resolution=2)
    with pytest.raises(GridMismatch):
        assert_canonical(tpl, "nilearn MNI152 template")


def test_to_canonical_moves_template_onto_grid():
    from nilearn import datasets

    tpl = datasets.load_mni152_template(resolution=2)
    moved = to_canonical(tpl, interpolation="continuous")
    assert_canonical(moved)


def test_refuses_continuous_interpolation_on_labels():
    labels = nib.Nifti1Image(
        np.random.randint(0, 5, (10, 10, 10)).astype(np.int16), np.eye(4)
    )
    with pytest.raises(ValueError, match="invent labels"):
        to_grid(labels, Grid.canonical(), interpolation="continuous")


def test_to_grid_preserves_content_when_already_aligned():
    img = _img()
    out = to_grid(img, Grid.canonical())
    assert np.array_equal(np.asarray(out.dataobj), np.asarray(img.dataobj))
    assert Grid.of(out).matches(Grid.of(img))


def test_to_grid_never_returns_the_caller_object():
    """B4: returning the input makes 'output' and 'input' the same array."""
    img = _img()
    out = to_grid(img, Grid.canonical())
    assert out is not img
    np.asarray(out.dataobj)[0, 0, 0] = 7
    assert np.asarray(img.dataobj)[0, 0, 0] == 0


def test_canonical_affine_cannot_be_mutated():
    """B3: the guardrail's own foundation must not be rewritable in place."""
    from lesionconn import config

    with pytest.raises(ValueError):
        Grid.canonical().affine[0, 3] = 999.0
    assert config.CANONICAL_AFFINE[0, 3] == 90.0


def test_grid_of_does_not_alias_the_source_image():
    """B3: a Grid must not be a live view onto an image's affine."""
    img = _img()
    grid = Grid.of(img)
    img.affine[0, 3] = 555.0
    assert grid.affine[0, 3] == 90.0


def test_is_integer_like():
    assert is_integer_like(_img(dtype=np.int16))
    assert not is_integer_like(
        nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32), np.eye(4))
    )


def test_label_detection_distinguishes_templates_from_parcellations():
    """int16 dtype alone must not mark an image as labels.

    MNI template T1s ship as int16 with thousands of intensity levels; treating
    them as label data would wrongly refuse continuous interpolation.
    """
    from lesionconn.spaces import looks_like_labels

    rng = np.random.default_rng(0)
    t1_like = nib.Nifti1Image(
        rng.integers(0, 4000, (40, 40, 40)).astype(np.int16), Grid.canonical().affine
    )
    parc_like = nib.Nifti1Image(
        rng.integers(0, 415, (40, 40, 40)).astype(np.int16), Grid.canonical().affine
    )
    assert not looks_like_labels(t1_like)
    assert looks_like_labels(parc_like)


def test_continuous_interpolation_allowed_on_integer_template():
    rng = np.random.default_rng(1)
    t1_like = nib.Nifti1Image(
        rng.integers(0, 4000, (60, 60, 60)).astype(np.int16), np.eye(4)
    )
    out = to_grid(t1_like, Grid.canonical(), interpolation="continuous")
    assert_canonical(out)
