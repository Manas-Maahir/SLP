"""The parcellation defines N and the shape of every downstream matrix."""

import numpy as np
import pytest

from lesionconn import config
from lesionconn.atlases import build_parcellation
from lesionconn.spaces import assert_canonical


@pytest.fixture(scope="module")
def parcellation():
    return build_parcellation(verbose=False)


def test_source_atlases_are_already_on_canonical_grid():
    """If this fails the canonical-grid choice was wrong, not the atlases."""
    from lesionconn.atlases import fetch_harvard_oxford_subcortical, fetch_schaefer

    assert_canonical(fetch_schaefer()[0], "Schaefer-400")
    assert_canonical(fetch_harvard_oxford_subcortical()[0], "Harvard-Oxford sub")


def test_parcellation_is_canonical(parcellation):
    assert_canonical(parcellation[0], "parcellation")


def test_labels_are_contiguous_from_one(parcellation):
    _, labels = parcellation
    assert labels["index"].tolist() == list(range(1, len(labels) + 1))
    assert labels["index"].is_unique
    assert labels["name"].is_unique


def test_cortex_occupies_the_first_400(parcellation):
    _, labels = parcellation
    cortical = labels[labels["index"] <= config.SCHAEFER_N_ROIS]
    assert len(cortical) == config.SCHAEFER_N_ROIS
    assert cortical["source_atlas"].str.startswith("Schaefer").all()


def test_tissue_classes_are_dropped(parcellation):
    """Cerebral WM / cortex / ventricle are not parcels; left in they would
    swallow most of a lesion's overlap and wreck the location baseline."""
    _, labels = parcellation
    names = labels["name"].str.lower()
    for pattern in config.HO_DROP_PATTERNS:
        assert not names.str.contains(pattern).any(), f"{pattern!r} survived into the parcellation"


def test_every_declared_label_has_voxels(parcellation):
    img, labels = parcellation
    present = set(np.unique(np.asarray(img.dataobj)).tolist()) - {0}
    declared = set(labels["index"].tolist())
    assert declared - present == set(), f"labels with no voxels: {sorted(declared - present)}"
    assert present - declared == set(), f"voxels with no label: {sorted(present - declared)}"


def test_no_cortical_parcel_was_eaten_by_subcortical(parcellation):
    """Merge rule: subcortical is written only where Schaefer is 0."""
    from lesionconn.atlases import fetch_schaefer

    img, _ = parcellation
    ctx = np.asarray(fetch_schaefer()[0].dataobj).astype(np.int16)
    merged = np.asarray(img.dataobj)
    assert np.array_equal(merged[ctx > 0], ctx[ctx > 0])


def test_labels_sit_inside_the_brain(parcellation):
    from nilearn import datasets

    from lesionconn.spaces import to_canonical

    img, _ = parcellation
    brain = np.asarray(
        to_canonical(datasets.load_mni152_brain_mask(resolution=2), "nearest").dataobj
    ) > 0
    labelled = np.asarray(img.dataobj) > 0
    outside_frac = np.count_nonzero(labelled & ~brain) / np.count_nonzero(labelled)
    assert outside_frac < 0.10, f"{outside_frac:.1%} of labelled voxels fall outside the brain"


def test_dtype_is_integer(parcellation):
    img, _ = parcellation
    assert np.issubdtype(np.asarray(img.dataobj).dtype, np.integer)
