"""M3a: the parcel x parcel disconnection matrix.

Assertions are on anatomy and on the contract, not on observed numbers. The
anatomical ones are the useful half: a temporal-pole lesion must disconnect the
temporal pole from orbitofrontal cortex, because that is what the uncinate
fasciculus joins. If the tract->parcel mapping were wrong, that would break while
every shape and symmetry check still passed.

docs/DISCONNECTION.md is the spec these tests encode, including the limits.
"""

from __future__ import annotations

import numpy as np
import pytest

from lesionconn import config, disconnectome as dc
from lesionconn.atlases import load_parcellation
from lesionconn.lesions import make_synthetic_lesion

LEFT_CORONA_RADIATA = (-26.0, -16.0, 28.0)
RIGHT_CORONA_RADIATA = (26.0, -16.0, 28.0)
LEFT_TEMPORAL_POLE = (-30.0, 10.0, -20.0)

N = config.EXPECTED_N_PARCELS


@pytest.fixture(scope="module")
def bits():
    for path in (config.TRACT_ATLAS_PATH, config.TRACT_PARCEL_MAP_PATH):
        if not path.exists():
            pytest.skip(f"{path.name} not built -- run scripts/11 and 12")
    tract_img, tract_labels = dc.load_tract_atlas()
    _parc, parc_labels = load_parcellation()
    return tract_img, tract_labels, dc.load_tract_parcel_map(), parc_labels


def matrix_for(bits, centre, radius=10.0):
    tract_img, tract_labels, tp_map, _ = bits
    sev = dc.severed_tracts(make_synthetic_lesion(centre, radius), tract_img, tract_labels)
    return dc.disconnection_matrix(sev, tp_map), sev


def pair(matrix, parc_labels, name_a: str, name_b: str) -> float:
    idx = dict(zip(parc_labels["name"], parc_labels["index"], strict=True))
    return float(matrix[idx[name_a] - 1, idx[name_b] - 1])


# --- the ARCHITECTURE.md M3 -> M4 contract ---------------------------------


def test_matrix_is_n_by_n_float_symmetric_with_a_zero_diagonal(bits):
    matrix, _ = matrix_for(bits, LEFT_CORONA_RADIATA)
    assert matrix.shape == (N, N)
    assert matrix.dtype == np.float32
    assert np.allclose(matrix, matrix.T)
    assert not matrix.diagonal().any()


def test_entries_are_bounded_by_the_worst_tract_severance(bits):
    """D = max f_t * a_i * a_j with a <= 1, so no entry can exceed max f_t."""
    matrix, sev = matrix_for(bits, LEFT_CORONA_RADIATA)
    assert matrix.min() >= 0.0
    assert matrix.max() <= sev.table["frac_severed"].max() + 1e-6


def test_an_intact_brain_disconnects_nothing(bits):
    matrix, _ = matrix_for(bits, (0.0, 0.0, 120.0), radius=5.0)
    assert not matrix.any()


def test_row_i_is_parcel_index_i_plus_one(bits):
    """The index convention the .npy is useless without."""
    _, _, _, parc_labels = bits
    assert list(parc_labels["index"]) == list(range(1, N + 1))


# --- anatomy ---------------------------------------------------------------


def test_temporal_pole_lesion_disconnects_the_uncinate_endpoints(bits):
    """The uncinate joins temporal pole to orbitofrontal cortex."""
    matrix, _ = matrix_for(bits, LEFT_TEMPORAL_POLE)
    _, _, _, parc_labels = bits
    uncinate = pair(
        matrix, parc_labels,
        "7Networks_LH_Limbic_TempPole_7", "7Networks_LH_Limbic_OFC_2",
    )
    assert uncinate > 0
    # and it should be the strongest thing in the matrix, not merely present
    assert uncinate == pytest.approx(matrix.max(), rel=1e-6)


def test_corona_radiata_lesion_disconnects_motor_cortex_from_the_brainstem(bits):
    """What the corticospinal tract is for."""
    matrix, _ = matrix_for(bits, LEFT_CORONA_RADIATA)
    _, _, _, parc_labels = bits
    assert pair(matrix, parc_labels, "7Networks_LH_SomMot_30", "Brain-Stem") > 0


def test_the_strongest_pairs_stay_in_the_lesioned_hemisphere(bits):
    """A left lesion must not produce right-right disconnection as its top finding."""
    matrix, _ = matrix_for(bits, LEFT_TEMPORAL_POLE)
    _, _, _, parc_labels = bits
    top = dc.matrix_summary(matrix, parc_labels, top_n=10)
    assert top, "a temporal pole lesion should disconnect something"
    for entry in top:
        assert "_RH_" not in entry["parcel_a"] or "_RH_" not in entry["parcel_b"]


def test_mirrored_lesions_give_mirrored_matrices(bits):
    """Laterality must survive the tract -> parcel step, not just the tract step."""
    left, _ = matrix_for(bits, LEFT_CORONA_RADIATA)
    right, _ = matrix_for(bits, RIGHT_CORONA_RADIATA)
    _, _, _, parc_labels = bits
    lh = parc_labels["name"].str.contains("_LH_").to_numpy()
    rh = parc_labels["name"].str.contains("_RH_").to_numpy()

    assert left[np.ix_(lh, lh)].max() > left[np.ix_(rh, rh)].max()
    assert right[np.ix_(rh, rh)].max() > right[np.ix_(lh, lh)].max()


# --- the approximation's documented behaviour ------------------------------


def test_an_intact_tract_contributes_nothing(bits):
    """f_t = 0 must drop out entirely, or every matrix would be dense with zeros."""
    tract_img, tract_labels, tp_map, _ = bits
    sev = dc.severed_tracts(
        make_synthetic_lesion(LEFT_TEMPORAL_POLE, 10.0), tract_img, tract_labels
    )
    severed = set(sev.hit["index"])
    matrix = dc.disconnection_matrix(sev, tp_map)
    only_intact = tp_map[~tp_map["tract_index"].isin(severed)]
    assert not dc.disconnection_matrix(sev, only_intact).any()
    assert matrix.any()


def test_affinity_threshold_bounds_how_wide_a_tract_can_reach(bits):
    """Without the floor, Corpus_Callosum alone would claim >90% of the matrix."""
    _, _, tp_map, _ = bits
    per_tract = tp_map.groupby("tract_index").size()
    assert per_tract.max() < N, "no tract may reach every parcel"
    assert tp_map["affinity"].min() >= dc.MIN_AFFINITY
    assert tp_map["affinity"].max() <= 1.0


def test_the_matrix_is_graded_not_binary(bits):
    """The whole point of the chosen rule: rank by value, never count nonzeros."""
    matrix, _ = matrix_for(bits, LEFT_CORONA_RADIATA)
    iu = np.triu_indices(N, k=1)
    nonzero = matrix[iu][matrix[iu] > 0]
    assert len(np.unique(nonzero)) > 100, "a graded matrix, not a thresholded one"
    # most of the mass is far below the top entry -- see the density table in
    # docs/DISCONNECTION.md
    assert np.median(nonzero) < 0.25 * nonzero.max()


def test_matrix_summary_ranks_by_value_and_names_both_parcels(bits):
    matrix, _ = matrix_for(bits, LEFT_TEMPORAL_POLE)
    _, _, _, parc_labels = bits
    top = dc.matrix_summary(matrix, parc_labels, top_n=5)
    values = [e["value"] for e in top]
    assert values == sorted(values, reverse=True)
    assert all(e["parcel_a"] and e["parcel_b"] for e in top)
    assert all(e["value"] > 0 for e in top)


def test_vermis_maps_to_no_parcel_which_is_a_known_gap(bits):
    """docs/DISCONNECTION.md limit 6: the N=415 node set has no cerebellum.

    Pinned so the gap is a recorded fact rather than a surprise in Phase 3.
    """
    _, tract_labels, tp_map, _ = bits
    vermis = int(tract_labels.loc[tract_labels["name"] == "Vermis", "index"].iloc[0])
    assert vermis not in set(tp_map["tract_index"])
