"""M3a from real streamline endpoints, and the comparison against the inferred one.

The most valuable assertion here is the last group: the two matrices agree at
network level and anti-correlate at parcel level, measured over the cohort. That
is not a bug in either -- it is evidence that the 415-parcel resolution is finer
than either method can resolve, and it is pinned so a future change that appears
to "fix" the disagreement has to explain what it actually changed.
"""

from __future__ import annotations

import numpy as np
import pytest

from lesionconn import config, streamlines as st
from lesionconn.atlases import load_parcellation
from lesionconn.lesions import make_synthetic_lesion
from lesionconn.spaces import Grid, GridMismatch

LEFT_CORONA_RADIATA = (-26.0, -16.0, 28.0)
RIGHT_CORONA_RADIATA = (26.0, -16.0, 28.0)
LEFT_TEMPORAL_POLE = (-30.0, 10.0, -20.0)

pytestmark = pytest.mark.skipif(
    not st.STREAMLINE_INDEX_PATH.exists(),
    reason="streamline index not built -- run scripts/15_build_streamline_index.py",
)


@pytest.fixture(scope="module")
def index():
    return st.load_streamline_index()


@pytest.fixture(scope="module")
def labels():
    _, parc_labels = load_parcellation()
    return parc_labels


def matrix_for(index, centre, radius=12.0):
    return st.streamline_disconnection_matrix(make_synthetic_lesion(centre, radius), index)


# --- contract ---------------------------------------------------------------


def test_matrix_is_n_by_n_symmetric_with_a_zero_diagonal(index):
    m = matrix_for(index, LEFT_CORONA_RADIATA)
    assert m.shape == (config.EXPECTED_N_PARCELS, config.EXPECTED_N_PARCELS)
    assert m.dtype == np.float32
    assert np.allclose(m, m.T)
    assert not m.diagonal().any()


def test_every_entry_is_a_counted_fraction(index):
    m = matrix_for(index, LEFT_CORONA_RADIATA)
    assert m.min() >= 0.0 and m.max() <= 1.0


def test_a_lesion_outside_the_brain_cuts_nothing(index):
    assert not matrix_for(index, (0.0, 0.0, 120.0), radius=5.0).any()
    assert st.severed_streamlines(
        make_synthetic_lesion((0.0, 0.0, 120.0), 5.0), index
    ).size == 0


def test_it_refuses_a_lesion_off_the_canonical_grid(index):
    import nibabel as nib

    off = nib.Nifti1Image(np.zeros((60, 60, 60), np.int16), np.eye(4))
    with pytest.raises(GridMismatch):
        st.streamline_disconnection_matrix(off, index)


def test_the_tractogram_space_check_rejects_a_foreign_template():
    """A tractogram in another template is the streamline version of the grid bug."""
    good = {"dimensions": [182, 218, 182], "voxel_sizes": [1.0, 1.0, 1.0]}
    st._assert_tractogram_space(good, Grid.canonical())
    bad = {"dimensions": [193, 229, 193], "voxel_sizes": [1.0, 1.0, 1.0]}  # ICBM 2009c
    with pytest.raises(AssertionError, match="world box"):
        st._assert_tractogram_space(bad, Grid.canonical())


def test_densify_closes_gaps_larger_than_a_voxel():
    line = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    dense = st._densify(line, step_mm=1.0)
    assert np.linalg.norm(np.diff(dense, axis=0), axis=1).max() <= 1.001
    assert len(dense) > len(line)


# --- anatomy ----------------------------------------------------------------


def test_a_left_lesion_cuts_mostly_left_hemisphere_connections(index, labels):
    m = matrix_for(index, LEFT_TEMPORAL_POLE)
    top = st.top_disconnections(m, labels, top_n=10)
    assert top, "a temporal pole lesion should cut something"
    rh = sum(1 for e in top if "_RH_" in e["parcel_a"] and "_RH_" in e["parcel_b"])
    assert rh == 0, "no right-right pair should lead a left-hemisphere lesion"


def test_corona_radiata_cuts_motor_cortex_from_the_brainstem(index, labels):
    """The corticospinal tract, counted rather than inferred."""
    m = matrix_for(index, LEFT_CORONA_RADIATA)
    top = st.top_disconnections(m, labels, top_n=25)
    assert any(
        "Brain-Stem" in (e["parcel_a"], e["parcel_b"]) and "SomMot" in e["parcel_a"] + e["parcel_b"]
        for e in top
    ), "motor cortex <-> brainstem should be among the strongest well-supported cuts"


def test_mirrored_lesions_give_mirrored_matrices(index, labels):
    left = matrix_for(index, LEFT_CORONA_RADIATA)
    right = matrix_for(index, RIGHT_CORONA_RADIATA)
    lh = labels["name"].str.contains("_LH_").to_numpy()
    rh = labels["name"].str.contains("_RH_").to_numpy()
    assert left[np.ix_(lh, lh)].sum() > left[np.ix_(rh, rh)].sum()
    assert right[np.ix_(rh, rh)].sum() > right[np.ix_(lh, lh)].sum()


# --- the support floor ------------------------------------------------------


def test_ranking_excludes_edges_below_the_fibre_floor(index, labels):
    """Median support is 9 fibres; without a floor the ranking is led by noise."""
    m = matrix_for(index, LEFT_CORONA_RADIATA)
    support = st.edge_support_matrix(index)
    for e in st.top_disconnections(m, labels, support, top_n=15):
        assert e["n_streamlines"] >= st.MIN_EDGE_STREAMLINES


def test_the_floor_filters_the_ranking_not_the_matrix(index, labels):
    """M4 must still receive every edge; the floor governs reporting only."""
    m = matrix_for(index, LEFT_CORONA_RADIATA)
    support = st.edge_support_matrix(index)
    thin = (support > 0) & (support < st.MIN_EDGE_STREAMLINES)
    assert (m[thin] > 0).any(), "thin edges must survive in the matrix itself"


def test_support_matrix_is_symmetric_and_matches_the_index(index):
    support = st.edge_support_matrix(index)
    assert np.allclose(support, support.T)
    assert support.diagonal().sum() == 0
    assert support.max() == index.edge_totals.max()


def test_every_ranked_row_carries_the_fibre_count_behind_it(index, labels):
    """100% of 900 fibres and 100% of 10 are the same number, different findings."""
    m = matrix_for(index, LEFT_CORONA_RADIATA)
    for e in st.top_disconnections(m, labels, top_n=5):
        assert e["n_streamlines"] > 0
        assert 0.0 < e["value"] <= 1.0


# --- the two matrices, compared ---------------------------------------------


@pytest.mark.skipif(
    not config.MANIFEST_PATH.exists(), reason="no cohort manifest"
)
def test_the_two_matrices_agree_by_network_and_disagree_by_parcel():
    """Measured over the cohort: +0.33 by network, -0.32 by parcel.

    Pinned because it is the finding, not an incidental number: both methods
    concur on which systems a lesion disconnects and contradict each other on
    which parcel inside them, which is what says the 415-node resolution is
    finer than either can resolve.
    """
    from scipy.stats import spearmanr

    from lesionconn import cohort

    manifest = cohort.load_manifest()
    _, parc_labels = load_parcellation()
    group = (parc_labels["hemisphere"].astype(str) + "_"
             + parc_labels["network"].astype(str)).to_numpy()
    _, gid = np.unique(group, return_inverse=True)
    n_groups = gid.max() + 1
    n = config.EXPECTED_N_PARCELS
    iu = np.triu_indices(n, 1)
    iu_g = np.triu_indices(n_groups, 1)

    parcel_rho, network_rho = [], []
    for subject_id in manifest["subject_id"].head(25):
        paths = cohort.subject_paths(subject_id)
        if not paths["streamline_matrix_path"].exists():
            continue
        approx = np.load(paths["matrix_path"])
        stream = np.load(paths["streamline_matrix_path"])

        sel = (approx[iu] > 0) | (stream[iu] > 0)
        if sel.sum() >= 20:
            parcel_rho.append(spearmanr(approx[iu][sel], stream[iu][sel]).statistic)

        def collapse(m):
            out = np.zeros((n_groups, n_groups), np.float32)
            np.maximum.at(out, (gid[:, None].repeat(n, 1), gid[None, :].repeat(n, 0)), m)
            return out

        ga, gs = collapse(approx)[iu_g], collapse(stream)[iu_g]
        gsel = (ga > 0) | (gs > 0)
        if gsel.sum() >= 10:
            network_rho.append(spearmanr(ga[gsel], gs[gsel]).statistic)

    assert len(parcel_rho) >= 10 and len(network_rho) >= 10
    assert np.median(network_rho) > 0.15, "they should agree about brain systems"
    assert np.median(parcel_rho) < 0.0, "and disagree parcel by parcel"
    assert np.median(network_rho) > np.median(parcel_rho)
