"""M1 -- the grid and laterality contracts, and the mm-correctness of dilation.

The registration itself is not run here; it takes ~35s per subject and needs
ARC on disk. What is tested is everything that decides whether a warp is
*trusted*, because those are the parts that fail silently. A misregistration is
visible; a mirrored one is not.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from lesionconn import registration as reg
from lesionconn.spaces import Grid

ants = pytest.importorskip("ants", reason="ANTsPy not installed")


# --- the template is the right MNI ------------------------------------------


def test_template_is_the_fsl_mni152_1mm_grid():
    """Two images are legitimately called 'MNI'. Phase 0 caught nilearn returning
    ICBM 2009c (99,117,95); picking the wrong one shifts every warped lesion."""
    t = reg.mni_template()
    assert t.shape == (182, 218, 182)
    assert "".join(nib.aff2axcodes(t.affine)) == Grid.canonical().orientation
    np.testing.assert_allclose(t.affine[:3, 3], Grid.canonical().affine[:3, 3], atol=1e-3)


def test_template_brain_mask_is_plausible_and_centred():
    brain = reg.template_brain_mask()
    frac = brain.mean()
    assert 0.05 < frac < 0.35, f"brain occupies {frac:.2%} of the 1mm box"
    # a brain mask must be roughly left/right symmetric, whatever the threshold
    left, right = brain[: brain.shape[0] // 2].sum(), brain[brain.shape[0] // 2 :].sum()
    assert abs(left - right) / max(left, right) < 0.05


# --- laterality survives the array order ------------------------------------


def _blob_at(world_x: float, affine, shape=(60, 60, 60)) -> nib.Nifti1Image:
    data = np.zeros(shape, np.uint8)
    grid = np.argwhere(np.ones(shape, bool))
    world = nib.affines.apply_affine(affine, grid)
    hit = (np.abs(world[:, 0] - world_x) < 4) & (np.abs(world[:, 1]) < 4) & (np.abs(world[:, 2]) < 4)
    for i, j, k in grid[hit]:
        data[i, j, k] = 1
    assert data.any(), "test blob landed outside the box"
    return nib.Nifti1Image(data, affine)


def test_frac_left_reads_world_coordinates_not_array_order():
    """The same anatomy stored LAS and RAS must give the same answer.

    This is the Phase 0 / Phase 1 bug class in miniature: flipping the array and
    the affine together leaves shape, volume and voxel count identical and moves
    the anatomy to the other hemisphere. Anything that indexes the array instead
    of the world fails here.
    """
    las = np.array([[-1.0, 0, 0, 30], [0, 1.0, 0, -30], [0, 0, 1.0, -30], [0, 0, 0, 1]])
    ras = np.array([[1.0, 0, 0, -30], [0, 1.0, 0, -30], [0, 0, 1.0, -30], [0, 0, 0, 1]])
    for affine in (las, ras):
        assert reg.frac_left_hemisphere(_blob_at(-20.0, affine)) == 1.0
        assert reg.frac_left_hemisphere(_blob_at(+20.0, affine)) == 0.0


def test_frac_left_is_nan_for_an_empty_mask():
    empty = nib.Nifti1Image(np.zeros((10, 10, 10), np.uint8), np.eye(4))
    assert np.isnan(reg.frac_left_hemisphere(empty))


# --- cost-function masking --------------------------------------------------


def test_dilation_is_millimetres_not_voxels_on_an_anisotropic_grid():
    """A 3mm rim must be 3mm in every direction, including the 3mm-voxel axis."""
    mask = np.zeros((40, 40, 40), bool)
    mask[20, 20, 20] = True
    grown = reg._dilate_mm(mask, (1.0, 1.0, 3.0), 3.0)
    x_extent = np.ptp(np.argwhere(grown)[:, 0]) + 1  # voxels, 1mm each
    z_extent = np.ptp(np.argwhere(grown)[:, 2]) + 1  # voxels, 3mm each
    assert x_extent == 7, f"3mm each way at 1mm voxels should span 7, got {x_extent}"
    assert z_extent == 3, f"3mm each way at 3mm voxels should span 3, got {z_extent}"


def test_dilation_only_grows():
    mask = np.zeros((20, 20, 20), bool)
    mask[8:12, 8:12, 8:12] = True
    grown = reg._dilate_mm(mask, (1.0, 1.0, 1.0), 2.0)
    assert (grown | mask == grown).all()
    assert grown.sum() > mask.sum()


def test_dilation_of_an_empty_mask_stays_empty():
    empty = np.zeros((10, 10, 10), bool)
    assert not reg._dilate_mm(empty, (1.0, 1.0, 1.0), 5.0).any()


# --- the QC row -------------------------------------------------------------


def test_mutual_information_is_higher_for_an_image_against_itself():
    """A sanity floor on the metric, not a threshold to tune.

    MI against self is the ceiling; MI against noise is the floor. If a real
    registration does not sit between them, the metric is broken rather than
    the warp.
    """
    template = np.asanyarray(reg.mni_template().dataobj, dtype=np.float32)
    rng = np.random.default_rng(0)
    noise = rng.random(template.shape).astype(np.float32)
    same = reg._mutual_information(template, template)
    versus_noise = reg._mutual_information(noise, template)
    assert same > versus_noise
    assert versus_noise >= 0.0


# --- the QC image dtype (docs/BUGS.md B8) -----------------------------------


def test_qc_image_is_int16_and_carries_no_token_coincidences(tmp_path):
    """float32 payloads byte-grep as printable ASCII; int16 does not.

    Pinned because reverting to float32 would silently reintroduce 26 phantom
    patient-name hits per volume in the repo-wide de-identification scan, and
    the tempting fix at that point is to narrow the scan instead.
    """
    import gzip
    import re

    from lesionconn import deid

    rng = np.random.default_rng(0)
    data = rng.uniform(0, 1200, (40, 40, 40)).astype(np.float32)
    path = reg.save_qc_image(data, np.eye(4), tmp_path / "qc.nii.gz")

    assert nib.load(str(path)).get_data_dtype() == np.int16
    tokens = deid.sensitive_tokens()
    if tokens:  # only meaningful once a subject map exists
        pattern = re.compile(deid.token_pattern(tokens).pattern.encode(), re.IGNORECASE)
        assert not pattern.search(gzip.decompress(path.read_bytes()))

    # The rate is low per byte -- one 28 MB volume produced 26 hits, this 256 KB
    # fixture would usually produce none -- so the contrast is not re-derived
    # here on random data. It is measured, with real numbers, in docs/BUGS.md B8.
    # What this test pins is the dtype, which is the thing a refactor can undo.


def test_qc_image_preserves_anatomy_it_is_meant_to_show(tmp_path):
    """Rounding to int16 must not move or lose the structure being inspected."""
    data = np.zeros((30, 30, 30), np.float32)
    data[10:20, 10:20, 10:20] = 800.5
    path = reg.save_qc_image(data, np.eye(4), tmp_path / "qc.nii.gz")
    out = np.asanyarray(nib.load(str(path)).dataobj)
    assert (out > 0).sum() == (data > 0).sum()
    assert abs(int(out.max()) - 800) <= 1


def test_registration_refuses_a_mask_that_is_not_on_the_t2w_grid(tmp_path):
    """A silent resample here would displace the lesion before registration."""
    from lesionconn.spaces import GridMismatch

    t2w = tmp_path / "t2w.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((50, 50, 50), np.float32), np.eye(4)), t2w)
    off = np.diag([2.0, 2.0, 2.0, 1.0])
    data = np.zeros((25, 25, 25), np.uint8)
    data[10:14, 10:14, 10:14] = 1
    nib.save(nib.Nifti1Image(data, off), mask)
    with pytest.raises(GridMismatch):
        reg.register_lesion(t2w, mask, "sub-TEST")


def test_registration_refuses_an_empty_mask(tmp_path):
    t2w = tmp_path / "t2w.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((50, 50, 50), np.float32), np.eye(4)), t2w)
    nib.save(nib.Nifti1Image(np.zeros((50, 50, 50), np.uint8), np.eye(4)), mask)
    with pytest.raises(ValueError, match="empty lesion"):
        reg.register_lesion(t2w, mask, "sub-TEST")
