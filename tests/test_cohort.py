"""Milestone 3: the cohort run, its manifest, and the de-identification contract.

The manifest is where a patient name would most plausibly leak -- it is the one
artifact that exists to index the cohort, and the obvious implementation stores
the source path in it. So most of this file is about what is NOT in the manifest.
"""

from __future__ import annotations

import numpy as np
import pytest

from lesionconn import cohort, config, deid

pytestmark = pytest.mark.skipif(
    not config.MANIFEST_PATH.exists(),
    reason="no cohort manifest on this machine -- run scripts/13_build_cohort.py",
)


@pytest.fixture(scope="module")
def manifest():
    return cohort.load_manifest()


# --- de-identification -----------------------------------------------------


def test_manifest_holds_no_source_path_or_filename_column(manifest):
    """The source path is the obvious thing to store, and it names the patient."""
    forbidden = {"source_path", "source_filename", "filename", "path", "name"}
    assert not (set(manifest.columns) & forbidden)


def test_every_manifest_subject_id_is_de_identified(manifest):
    assert all(deid.SUBJECT_ID_RE.match(s) for s in manifest["subject_id"])
    assert manifest["subject_id"].is_unique


def test_no_manifest_cell_contains_a_patient_token(manifest):
    """Belt and braces over the whole-repo grep: check the values directly."""
    if not config.SUBJECT_MAP_PATH.exists():
        pytest.skip("no subject map")
    pattern = deid.token_pattern(deid.sensitive_tokens())
    hits = [
        (col, str(value))
        for col in manifest.columns
        for value in manifest[col].tolist()
        if pattern.search(str(value))
    ]
    assert not hits, f"{len(hits)} manifest cell(s) name a patient"


def test_every_cached_artifact_is_named_by_subject_id_alone(manifest):
    for subject_id in manifest["subject_id"]:
        for path in cohort.subject_paths(subject_id).values():
            assert path.name.startswith(subject_id)
            assert not deid.filename_tokens(path.name) & deid.sensitive_tokens()


def test_save_manifest_refuses_a_frame_carrying_a_patient_name(tmp_path):
    import pandas as pd

    if not config.SUBJECT_MAP_PATH.exists():
        pytest.skip("no subject map")
    leaked = deid.load_subject_map().iloc[0]["source_filename"]
    bad = pd.DataFrame([{"subject_id": "sub-0001", "note": leaked}])
    with pytest.raises(ValueError, match="patient-identifying token"):
        cohort.save_manifest(bad, tmp_path / "manifest.csv")


# --- the manifest contract -------------------------------------------------


def test_manifest_has_the_columns_milestone_3_asked_for(manifest):
    required = {
        "subject_id", "qc_passed", "qc_reasons", "native_volume_mm3",
        "lesion_path", "severed_tracts_path", "tract_vector_path",
        "matrix_path", "location_path", "location_nonzero",
    }
    assert required <= set(manifest.columns)


def test_native_volume_is_recorded_and_differs_from_canonical(manifest):
    """docs/BUGS.md B2: the 2mm grid is the substrate, never the volume source."""
    assert (manifest["native_volume_mm3"] > 0).all()
    assert (manifest["native_volume_mm3"] != manifest["canonical_volume_mm3"]).any()


def test_qc_status_and_reason_are_both_present(manifest):
    assert manifest["qc_passed"].isin([True, False]).all()
    failed = manifest[~manifest["qc_passed"]]
    assert (failed["qc_reasons"].astype(str).str.len() > 0).all(), (
        "an exclusion without a recorded reason is exactly what qc.py forbids"
    )


def test_no_subject_has_an_all_zero_location_vector(manifest):
    """docs/BUGS.md B1, on real data rather than synthetic probes."""
    assert (manifest["location_nonzero"] > 0).all()


def test_every_cached_artifact_exists_and_has_the_contracted_shape(manifest):
    for subject_id in manifest["subject_id"].head(5):
        paths = cohort.subject_paths(subject_id)
        for path in paths.values():
            assert path.exists(), f"{path.name} missing"
        assert np.load(paths["tract_vector_path"]).shape == (config.EXPECTED_N_TRACTS,)
        assert np.load(paths["matrix_path"]).shape == (
            config.EXPECTED_N_PARCELS, config.EXPECTED_N_PARCELS,
        )
        assert np.load(paths["location_path"]).shape[0] > config.EXPECTED_N_PARCELS


def test_the_location_baseline_uses_the_463_atlas_not_the_415_node_set(manifest):
    """docs/BUGS.md B1: mixing these up is the bug that made the baseline degenerate."""
    from lesionconn.atlases import load_location_atlas

    _, labels = load_location_atlas()
    n_location = len(labels)
    assert n_location > config.EXPECTED_N_PARCELS
    for subject_id in manifest["subject_id"].head(3):
        vector = np.load(cohort.subject_paths(subject_id)["location_path"])
        assert vector.shape == (n_location,)


def test_features_are_finite_and_in_range(manifest):
    for subject_id in manifest["subject_id"].head(5):
        paths = cohort.subject_paths(subject_id)
        for key in ("tract_vector_path", "matrix_path", "location_path"):
            v = np.load(paths[key])
            assert np.isfinite(v).all()
            assert v.min() >= 0.0 and v.max() <= 1.0


def test_cohort_report_names_every_exclusion(manifest):
    text = cohort.cohort_report(manifest)
    assert "QC excluded" in text
    for row in manifest[~manifest["qc_passed"]].itertuples():
        assert row.subject_id in text
