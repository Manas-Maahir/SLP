"""Model A's feature space must never be blind to a plausible stroke.

docs/BUGS.md B1: the disconnection parcellation is grey matter only, so corona
radiata and centrum semiovale lesions -- among the commonest stroke locations --
scored an all-zero location vector. A baseline with no information about exactly
the subjects where disconnection should win makes the Phase 3 comparison
meaningless, and biased in our own favour.
"""

import numpy as np
import pytest
from nilearn import datasets

from lesionconn.atlases import build_location_atlas, build_parcellation
from lesionconn.lesions import lesion_parcel_overlap, location_feature_vector, make_synthetic_lesion
from lesionconn.spaces import assert_canonical, to_canonical

# Deep white matter: no cortical parcel, and common stroke territory.
WHITE_MATTER = {
    "left corona radiata": (-26.0, -16.0, 28.0),
    "left centrum semiovale": (-28.0, -12.0, 36.0),
    "right corona radiata": (26.0, -16.0, 28.0),
}
GREY_MATTER = {"left putamen": (-26.0, 0.0, 8.0)}


@pytest.fixture(scope="module")
def location_atlas():
    parc_img, parc_labels = build_parcellation(verbose=False)
    return build_location_atlas(parc_img, parc_labels, verbose=False)


@pytest.fixture(scope="module")
def parcellation():
    return build_parcellation(verbose=False)


def test_location_atlas_is_canonical(location_atlas):
    assert_canonical(location_atlas[0], "location atlas")


def test_labels_contiguous_and_unique(location_atlas):
    _, labels = location_atlas
    assert labels["index"].tolist() == list(range(1, len(labels) + 1))
    assert labels["name"].is_unique


@pytest.mark.parametrize("name,centre", sorted(WHITE_MATTER.items()))
def test_white_matter_lesion_is_not_invisible(location_atlas, name, centre):
    """The regression test for B1. These three coordinates used to score zero."""
    atlas, labels = location_atlas
    lesion = make_synthetic_lesion(centre, 8.0)
    vec = location_feature_vector(lesion, atlas, labels)
    assert vec.sum() > 0, f"{name} produced an all-zero location vector"
    assert (vec > 0).sum() >= 1


@pytest.mark.parametrize("name,centre", sorted(WHITE_MATTER.items()))
def test_white_matter_lesion_scored_zero_under_the_old_parcellation(
    parcellation, name, centre
):
    """Pins the defect itself, so a regression is unambiguous rather than subtle."""
    parc, parc_labels = parcellation
    vec = location_feature_vector(make_synthetic_lesion(centre, 8.0), parc, parc_labels)
    assert vec.sum() == 0


def test_grey_matter_lesion_still_hits_grey_matter_parcels(location_atlas):
    atlas, labels = location_atlas
    ov = lesion_parcel_overlap(make_synthetic_lesion(GREY_MATTER["left putamen"], 8.0),
                               atlas, labels)
    assert "Left Putamen" in ov.head(5)["name"].tolist()


def test_white_matter_laterality_is_preserved(location_atlas):
    """Juelich tracts are bilateral in one label; unsplit, L and R would tie."""
    atlas, labels = location_atlas
    left = location_feature_vector(
        make_synthetic_lesion(WHITE_MATTER["left corona radiata"], 8.0), atlas, labels)
    right = location_feature_vector(
        make_synthetic_lesion(WHITE_MATTER["right corona radiata"], 8.0), atlas, labels)
    assert not np.array_equal(left, right)

    hemi = labels.sort_values("index")["hemisphere"].to_numpy()
    assert left[hemi == "RH"].sum() == 0
    assert right[hemi == "LH"].sum() == 0


def test_grey_matter_block_is_unchanged_from_the_node_set(location_atlas, parcellation):
    """The disconnection node set must not drift when the baseline gains labels."""
    atlas, labels = location_atlas
    parc, parc_labels = parcellation
    n = len(parc_labels)
    assert labels.iloc[:n]["name"].tolist() == parc_labels["name"].tolist()

    a = np.asarray(atlas.dataobj)
    p = np.asarray(parc.dataobj)
    assert np.array_equal(a[p > 0], p[p > 0])


def test_every_brain_voxel_carries_a_label(location_atlas):
    atlas, _ = location_atlas
    brain = np.asarray(
        to_canonical(datasets.load_mni152_brain_mask(resolution=2), "nearest").dataobj
    ) > 0
    labelled = np.asarray(atlas.dataobj) > 0
    coverage = np.count_nonzero(labelled & brain) / np.count_nonzero(brain)
    assert coverage > 0.98, f"only {coverage:.1%} of brain carries a label"


def test_feature_vector_length_matches_labels(location_atlas):
    atlas, labels = location_atlas
    vec = location_feature_vector(make_synthetic_lesion((-26.0, 0.0, 8.0), 8.0), atlas, labels)
    assert vec.shape == (len(labels),)
    assert vec.dtype == np.float32


CEREBELLUM = {"left cerebellum": (-24.0, -60.0, -40.0),
              "right cerebellum": (24.0, -60.0, -40.0)}


@pytest.mark.parametrize("name,centre", sorted(CEREBELLUM.items()))
def test_cerebellar_lesion_is_not_invisible(location_atlas, name, centre):
    """Second instance of B1: no atlas in the node set labels the cerebellum."""
    atlas, labels = location_atlas
    vec = location_feature_vector(make_synthetic_lesion(centre, 8.0), atlas, labels)
    assert vec.sum() > 0, f"{name} produced an all-zero location vector"


def test_catch_all_is_a_last_resort_not_a_dumping_ground(location_atlas):
    """If the catch-all absorbs most of the brain, the real atlases aren't working."""
    atlas, labels = location_atlas
    data = np.asarray(atlas.dataobj)
    catch = labels[labels["network"] == "Unassigned"]["index"].tolist()
    share = np.count_nonzero(np.isin(data, catch)) / np.count_nonzero(data > 0)
    assert share < 0.30, f"catch-all holds {share:.1%} of labelled voxels"


def test_named_anatomy_beats_catch_all_for_a_white_matter_lesion(location_atlas):
    """A corona radiata lesion should land in a named tract, not 'unassigned'."""
    atlas, labels = location_atlas
    ov = lesion_parcel_overlap(
        make_synthetic_lesion(WHITE_MATTER["left corona radiata"], 8.0), atlas, labels)
    assert ov.iloc[0]["network"] != "Unassigned", ov.head(3)["name"].tolist()
