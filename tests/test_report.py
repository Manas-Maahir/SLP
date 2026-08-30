"""M5: the report a human reads.

The report is where a wrong number does the most damage, because it is the only
artifact anyone reads without also reading the code. Two defects found by looking
at real output are pinned here so they cannot come back:

  * the anatomy section must come from the N=463 location atlas. Built from the
    N=415 node set it led with LEFT occipital parcels for a lesion that is 70%
    RIGHT cerebellum, because the node set has no cerebellar labels and saw only
    the 17.8% of that lesion lying outside them.
  * the volume labelled "as acquired" must be measured on the source grid. Taken
    from the cached 2mm mask it read 424mm3 against a true 496mm3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lesionconn import config, deid, disconnectome as dc, report as rp
from lesionconn.atlases import load_location_atlas, load_parcellation
from lesionconn.lesions import lesion_parcel_overlap, make_synthetic_lesion

LEFT_TEMPORAL_POLE = (-30.0, 10.0, -20.0)
RIGHT_CEREBELLUM = (16.0, -55.0, -23.0)


@pytest.fixture(scope="module")
def kit():
    for path in (config.TRACT_ATLAS_PATH, config.TRACT_PARCEL_MAP_PATH):
        if not path.exists():
            pytest.skip(f"{path.name} not built -- run scripts/11 and 12")
    tract_img, tract_labels = dc.load_tract_atlas()
    parc_img, parc_labels = load_parcellation()
    loc_img, loc_labels = load_location_atlas()
    return tract_img, tract_labels, parc_img, parc_labels, loc_img, loc_labels


def make_report(kit, centre, radius=12.0, subject_id="sub-9999", **kwargs) -> str:
    tract_img, tract_labels, parc_img, parc_labels, loc_img, loc_labels = kit
    lesion = make_synthetic_lesion(centre, radius)
    sev = dc.severed_tracts(lesion, tract_img, tract_labels, subject_id=subject_id)
    return rp.render_text(
        subject_id, lesion, sev,
        lesion_parcel_overlap(lesion, parc_img, parc_labels),
        location_overlap=lesion_parcel_overlap(lesion, loc_img, loc_labels),
        matrix=dc.disconnection_matrix(sev, dc.load_tract_parcel_map()),
        parc_labels=parc_labels,
        **kwargs,
    )


# --- the framing travels with the output -----------------------------------


def test_the_disclaimer_appears_in_the_report_itself(kit):
    text = make_report(kit, LEFT_TEMPORAL_POLE)
    assert text.lower().count("not a diagnostic device") >= 2, "header and footer"
    assert "normative" in text.lower()
    assert "no diffusion imaging of this patient" in text.lower()


def test_the_report_says_the_matrix_is_derived_not_measured(kit):
    text = make_report(kit, LEFT_TEMPORAL_POLE)
    assert "docs/DISCONNECTION.md" in text
    assert "not severed-streamline" in text.lower()
    assert "lower bound" in text.lower()


# --- the two defects found by reading real output --------------------------


def test_a_cerebellar_lesion_is_described_as_cerebellar(kit):
    """The N=415 node set has no cerebellum; the report must not hide that."""
    text = make_report(kit, RIGHT_CEREBELLUM)
    anatomy = text.split("DISCONNECTION NODE SET")[0]
    assert "Cerebellum" in anatomy, (
        "the anatomy section must come from the N=463 location atlas, or a "
        "cerebellar lesion is described by whatever non-cerebellar tissue it clips"
    )
    assert "MOST OF THIS LESION IS INVISIBLE TO THE NODE SET" in text
    assert "node-set coverage of this lesion" in text


def test_node_set_coverage_is_always_stated(kit):
    for centre in (LEFT_TEMPORAL_POLE, RIGHT_CEREBELLUM):
        assert "node-set coverage of this lesion" in make_report(kit, centre)


def test_a_well_covered_lesion_does_not_get_the_coverage_warning(kit):
    """The warning must mean something, so it cannot fire on every report."""
    text = make_report(kit, LEFT_TEMPORAL_POLE)
    coverage_line = next(
        line for line in text.splitlines() if "node-set coverage" in line
    )
    pct = float(coverage_line.split(":")[1].strip().split("%")[0])
    if pct >= 50:
        assert "MOST OF THIS LESION IS INVISIBLE" not in text


def test_the_acquired_volume_names_the_grid_it_was_measured_on(kit):
    text = make_report(
        kit, LEFT_TEMPORAL_POLE,
        native_volume_mm3=496.0,
        native_grid="shape=(182, 218, 182) voxel=1x1x1mm orient=LAS",
    )
    assert "volume (as acquired)" in text
    assert "496" in text
    assert "1x1x1mm" in text
    assert "docs/BUGS.md B2" in text


# --- severity shown two ways ------------------------------------------------


def test_tracts_carry_absolute_volume_beside_the_percentage(kit):
    """docs/DISCONNECTION.md limit 7: the percentage alone favours small tracts."""
    text = make_report(kit, LEFT_TEMPORAL_POLE)
    tracts = text.split("WHITE-MATTER TRACTS SEVERED")[1]
    assert "% severed" in tracts
    assert "severed mm3" in tracts
    assert "Neither column alone" in tracts


def test_thin_tracts_are_marked_in_the_report(kit):
    text = make_report(kit, (0.0, -30.0, -26.0), radius=14.0)  # brainstem, thin tracts
    if "*" in text.split("WHITE-MATTER TRACTS SEVERED")[1].split("---")[0]:
        assert "thin at 2mm" in text


# --- de-identification ------------------------------------------------------


def test_the_report_names_the_subject_only_by_id(kit):
    text = make_report(kit, LEFT_TEMPORAL_POLE, subject_id="sub-0042")
    assert "sub-0042" in text
    if config.SUBJECT_MAP_PATH.exists():
        assert not deid.token_pattern(deid.sensitive_tokens()).search(text)


def test_write_report_refuses_text_carrying_a_patient_name(tmp_path):
    if not config.SUBJECT_MAP_PATH.exists():
        pytest.skip("no subject map")
    leaked = deid.load_subject_map().iloc[0]["source_filename"]
    report = rp.Report(subject_id="sub-0001", text=f"Report for {leaked}")
    with pytest.raises(ValueError, match="patient-identifying token"):
        rp.write_report(report, tmp_path / "report.txt")


# --- labels -----------------------------------------------------------------


def test_short_name_keeps_parcels_distinguishable():
    """Four axis labels once truncated to the same string on a real figure."""
    names = [f"7Networks_LH_SalVentAttn_FrOperIns_{i}" for i in (1, 4, 8, 9)]
    shortened = [rp._short_name(n) for n in names]
    assert len(set(shortened)) == len(names)
    assert all(not s.startswith("7Networks_") for s in shortened)


# --- figures ----------------------------------------------------------------


def test_all_four_figures_render_and_are_named_by_subject_id(kit, tmp_path):
    tract_img, tract_labels, _, parc_labels, _, _ = kit
    lesion = make_synthetic_lesion(LEFT_TEMPORAL_POLE, 12.0)
    sev = dc.severed_tracts(lesion, tract_img, tract_labels, subject_id="sub-0042")
    matrix = dc.disconnection_matrix(sev, dc.load_tract_parcel_map())

    figures = rp.render_figures("sub-0042", lesion, sev, matrix, parc_labels, tmp_path)
    assert len(figures) == 4
    kinds = {"glass_brain", "lesion_on_template", "severed_tracts", "disconnection_matrix"}
    assert {f.stem.split("sub-0042_")[1] for f in figures} == kinds
    for path in figures:
        assert path.name.startswith("sub-0042")
        assert path.stat().st_size > 5_000, f"{path.name} looks blank"


def test_figures_render_for_a_lesion_that_severs_nothing(kit, tmp_path):
    """The degenerate case still has to produce a figure, not a traceback."""
    tract_img, tract_labels, _, parc_labels, _, _ = kit
    lesion = make_synthetic_lesion((0.0, 0.0, 120.0), 5.0)
    sev = dc.severed_tracts(lesion, tract_img, tract_labels, subject_id="sub-0000")
    matrix = np.zeros((config.EXPECTED_N_PARCELS, config.EXPECTED_N_PARCELS), np.float32)
    figures = rp.render_figures("sub-0000", lesion, sev, matrix, parc_labels, tmp_path)
    assert all(p.exists() for p in figures)


def test_report_handles_a_lesion_with_no_tracts_and_no_parcels(kit):
    text = make_report(kit, (0.0, 0.0, 120.0), radius=5.0, subject_id="sub-0000")
    assert "sub-0000" in text
    assert "none" in text.lower()


def test_render_text_works_without_the_optional_sections(kit):
    """The entry point always passes them, but the function must not require them."""
    tract_img, tract_labels, parc_img, parc_labels, _, _ = kit
    lesion = make_synthetic_lesion(LEFT_TEMPORAL_POLE, 12.0)
    sev = dc.severed_tracts(lesion, tract_img, tract_labels)
    text = rp.render_text(
        "sub-0042", lesion, sev, lesion_parcel_overlap(lesion, parc_img, parc_labels)
    )
    assert "LESION DISCONNECTION REPORT" in text
    assert isinstance(text, str) and len(text) > 500


def test_empty_overlap_frames_do_not_crash_the_report(kit):
    tract_img, tract_labels, _, _parc_labels, _, _ = kit
    lesion = make_synthetic_lesion(LEFT_TEMPORAL_POLE, 12.0)
    sev = dc.severed_tracts(lesion, tract_img, tract_labels)
    empty = pd.DataFrame(columns=["name", "frac_of_parcel", "frac_of_lesion", "volume_mm3"])
    text = rp.render_text("sub-0042", lesion, sev, empty, location_overlap=empty)
    assert "no node-set parcel is involved at all." in text
