"""M3b: a lesion at a known coordinate must sever the known tracts.

Every assertion here is on anatomy -- "a left corona radiata lesion cuts the left
corticospinal tract and not the right one" -- rather than on a number observed
from a previous run. A test that pins today's output cannot tell a correct change
from a regression; a test that pins anatomy can.

The mirroring tests are the point of the file. The atlas is LPS and the canonical
grid is LAS, and a mirrored resampling produces a disconnection profile that is
perfectly plausible, perfectly canonical, and exactly backwards. So the guardrail
is not merely asserted to pass on good data -- it is shown to FAIL on a
deliberately mirrored atlas.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lesionconn import config, disconnectome as dc
from lesionconn.lesions import make_synthetic_lesion
from lesionconn.spaces import Grid, GridMismatch

# Coordinates are MNI world mm. The white-matter ones match the probes already
# used by tests/test_location_baseline.py, so the two modules are checked against
# the same anatomy.
LEFT_CORONA_RADIATA = (-26.0, -16.0, 28.0)
RIGHT_CORONA_RADIATA = (26.0, -16.0, 28.0)
LEFT_OPTIC_RADIATION = (-32.0, -60.0, 8.0)
LEFT_TEMPORAL_POLE = (-30.0, 10.0, -20.0)
CALLOSAL_GENU = (0.0, 26.0, 4.0)


@pytest.fixture(scope="module")
def atlas():
    if not config.TRACT_ATLAS_PATH.exists():
        pytest.skip(
            "tract atlas not built on this machine (data/ is gitignored) -- run "
            "scripts/11_build_tract_atlas.py"
        )
    return dc.load_tract_atlas()


def severance(atlas, centre, radius=10.0):
    img, labels = atlas
    return dc.severed_tracts(make_synthetic_lesion(centre, radius), img, labels)


def frac(sev, name: str) -> float:
    return float(sev.table.set_index("name").loc[name, "frac_severed"])


# --- laterality: the bug this module exists to prevent ----------------------


def test_left_corona_radiata_severs_the_left_corticospinal_tract(atlas):
    sev = severance(atlas, LEFT_CORONA_RADIATA)
    assert frac(sev, "Cortico_Spinal_Tract_L") > 0.0
    assert frac(sev, "Cortico_Spinal_Tract_R") == 0.0


def test_right_corona_radiata_severs_the_right_corticospinal_tract(atlas):
    sev = severance(atlas, RIGHT_CORONA_RADIATA)
    assert frac(sev, "Cortico_Spinal_Tract_R") > 0.0
    assert frac(sev, "Cortico_Spinal_Tract_L") == 0.0


def test_a_left_hemisphere_lesion_involves_no_right_hemisphere_tract(atlas):
    """Deep left white matter is 16mm from the midline; nothing right reaches it."""
    sev = severance(atlas, LEFT_CORONA_RADIATA)
    assert not sev.hit.empty
    assert set(sev.hit["hemisphere"]) <= {"L", "M"}


def test_a_right_hemisphere_lesion_involves_no_left_hemisphere_tract(atlas):
    sev = severance(atlas, RIGHT_CORONA_RADIATA)
    assert not sev.hit.empty
    assert set(sev.hit["hemisphere"]) <= {"R", "M"}


def test_mirrored_lesions_swap_hemispheres_rather_than_agreeing(atlas):
    """The check that catches a symmetric-but-wrong atlas.

    If laterality were broken by mirroring, BOTH lesions would report the same
    hemisphere. Requiring the profiles to swap is stronger than requiring either
    one alone to look right.
    """
    left = severance(atlas, LEFT_CORONA_RADIATA)
    right = severance(atlas, RIGHT_CORONA_RADIATA)
    assert set(left.hit["hemisphere"]) & {"R"} == set()
    assert set(right.hit["hemisphere"]) & {"L"} == set()
    assert frac(left, "Cortico_Spinal_Tract_L") > frac(right, "Cortico_Spinal_Tract_L")
    assert frac(right, "Cortico_Spinal_Tract_R") > frac(left, "Cortico_Spinal_Tract_R")


# --- anatomy beyond the corticospinal tract --------------------------------


def test_optic_radiation_lesion_severs_the_optic_radiation(atlas):
    sev = severance(atlas, LEFT_OPTIC_RADIATION)
    assert sev.hit.iloc[0]["name"] == "Optic_Radiation_L"
    assert frac(sev, "Optic_Radiation_R") == 0.0


def test_temporal_pole_lesion_severs_the_uncinate_fasciculus(atlas):
    sev = severance(atlas, LEFT_TEMPORAL_POLE)
    assert sev.hit.iloc[0]["name"] == "Uncinate_Fasciculus_L"


def test_midline_callosal_lesion_severs_the_corpus_callosum(atlas):
    sev = severance(atlas, CALLOSAL_GENU)
    assert sev.hit.iloc[0]["name"] == "Corpus_Callosum"


def test_a_lesion_outside_the_brain_severs_nothing(atlas):
    sev = severance(atlas, (0.0, 0.0, 120.0), radius=5.0)
    assert sev.hit.empty
    assert sev.vector().sum() == 0.0


# --- the guardrails, shown failing -----------------------------------------


def test_laterality_check_rejects_a_mirrored_atlas(atlas):
    """Flip the atlas across the midline; the guardrail must refuse it.

    A guardrail never observed to fire is an assumption, not a guarantee.
    """
    img, labels = atlas
    mirrored = np.asarray(img.dataobj)[::-1, ...]
    with pytest.raises(AssertionError, match="laterality is wrong"):
        dc.assert_tract_laterality(mirrored, labels, Grid.canonical())


def test_native_comparison_rejects_a_mirrored_atlas():
    """The anatomy-free half of the guardrail: tracts must land where they started."""
    native = {"Arcuate_Fasciculus_L": (-44.6, -13.5, 14.0)}
    dc.assert_matches_native({"Arcuate_Fasciculus_L": (-44.4, -13.6, 14.2)}, native)
    with pytest.raises(AssertionError, match="centre of mass moved"):
        dc.assert_matches_native({"Arcuate_Fasciculus_L": (44.6, -13.5, 14.0)}, native)


def test_native_comparison_rejects_an_anterior_posterior_flip():
    """LPS -> LAS flips y. A y-mirror is as plausible a bug as an x-mirror."""
    native = {"Cortico_Spinal_Tract_L": (-24.4, -15.8, 29.9)}
    with pytest.raises(AssertionError, match="centre of mass moved"):
        dc.assert_matches_native({"Cortico_Spinal_Tract_L": (-24.4, 15.8, 29.9)}, native)


def test_severed_tracts_refuses_a_lesion_off_the_canonical_grid(atlas):
    import nibabel as nib

    img, labels = atlas
    off_grid = nib.Nifti1Image(np.zeros((60, 60, 60), np.int16), np.eye(4))
    with pytest.raises(GridMismatch):
        dc.severed_tracts(off_grid, img, labels)


def test_the_built_atlas_passes_its_own_laterality_check(atlas):
    img, labels = atlas
    dc.assert_tract_laterality(np.asarray(img.dataobj), labels, Grid.canonical())


# --- shape of the output ---------------------------------------------------


def test_atlas_has_eighty_tracts_split_evenly_by_hemisphere(atlas):
    _, labels = atlas
    assert len(labels) == config.EXPECTED_N_TRACTS
    counts = labels["hemisphere"].value_counts()
    assert counts["L"] == counts["R"]


def test_tract_names_carry_no_stray_whitespace(atlas):
    _, labels = atlas
    assert all(n == n.strip() for n in labels["name"])
    assert "Posterior_Commissure" in set(labels["name"])


def test_frac_severed_is_a_fraction(atlas):
    sev = severance(atlas, LEFT_CORONA_RADIATA)
    assert sev.table["frac_severed"].between(0.0, 1.0).all()
    assert sev.table["frac_of_lesion"].between(0.0, 1.0).all()


def test_table_is_sorted_worst_first(atlas):
    sev = severance(atlas, LEFT_CORONA_RADIATA)
    f = sev.table["frac_severed"].to_numpy()
    assert np.all(np.diff(f) <= 0)


def test_feature_vector_is_eighty_dim_and_index_ordered(atlas):
    _img, labels = atlas
    sev = severance(atlas, LEFT_CORONA_RADIATA)
    v = sev.vector()
    assert v.shape == (config.EXPECTED_N_TRACTS,)
    cst_l = int(labels.loc[labels["name"] == "Cortico_Spinal_Tract_L", "index"].iloc[0])
    assert v[cst_l] == pytest.approx(frac(sev, "Cortico_Spinal_Tract_L"))


def test_severed_tracts_json_meets_the_architecture_contract(atlas, tmp_path):
    sev = severance(atlas, LEFT_CORONA_RADIATA)
    sev.subject_id = "sub-0001"
    path = dc.write_severed_tracts(sev, tmp_path / "severed_tracts.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["subject_id"] == "sub-0001"
    assert payload["atlas"]["n_tracts"] == config.EXPECTED_N_TRACTS
    assert payload["space"]["orientation"] == "LAS"
    assert payload["space"]["shape"] == list(config.CANONICAL_SHAPE)
    assert payload["n_tracts_involved"] == len(sev.hit)
    assert "not a diagnostic" in payload["disclaimer"].lower()
    assert "lower bound" in payload["measure"]

    names = [t["name"] for t in payload["tracts"]]
    assert "Cortico_Spinal_Tract_L" in names
    assert "Cortico_Spinal_Tract_R" not in names
    for t in payload["tracts"]:
        assert 0.0 <= t["frac_severed"] <= 1.0
        assert t["n_voxels_hit"] > 0


def test_thin_tracts_are_flagged_not_dropped(atlas):
    """2mm sampling under-represents the cranial nerves. Recorded, never hidden."""
    _, labels = atlas
    assert labels["thin"].any()
    assert len(labels) == config.EXPECTED_N_TRACTS
    assert (labels["n_voxels"] > 0).all()
