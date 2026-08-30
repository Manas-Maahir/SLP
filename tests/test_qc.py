"""QC must exclude the masks ATLAS is known to ship broken -- and say why."""

import nibabel as nib
import numpy as np
import pytest

from lesionconn import qc
from lesionconn.lesions import make_synthetic_lesion
from lesionconn.spaces import Grid


def _empty():
    g = Grid.canonical()
    return nib.Nifti1Image(np.zeros(g.shape, np.int16), g.affine)


def _whole_brain():
    g = Grid.canonical()
    d = np.zeros(g.shape, np.int16)
    d[10:80, 10:100, 10:80] = 1
    return nib.Nifti1Image(d, g.affine)


@pytest.fixture(scope="module")
def brain_mask():
    from nilearn import datasets

    from lesionconn.spaces import to_canonical

    return to_canonical(datasets.load_mni152_brain_mask(resolution=2), "nearest")


def test_normal_lesion_passes(brain_mask):
    r = qc.check_lesion_mask(make_synthetic_lesion((-26, 0, 8), 10), "sub-x001", brain_mask)
    assert r.passed and not r.reasons


def test_empty_mask_is_excluded():
    r = qc.check_lesion_mask(_empty(), "sub-x002")
    assert not r.passed
    assert "empty" in r.reasons[0]


def test_known_bad_subjects_are_excluded_with_a_cited_reason():
    for sid in ("sub-r039s002", "r005s081", "sub-r009-s003", "sub-r040-s043"):
        r = qc.check_lesion_mask(make_synthetic_lesion((-26, 0, 8), 10), sid)
        assert not r.passed, f"{sid} should be excluded"
        assert "known-bad upstream" in r.reasons[0]
        assert "issue #" in r.reasons[0], "an exclusion must cite its source"


def test_tissue_mask_masquerading_as_lesion_is_caught(brain_mask):
    r = qc.check_lesion_mask(_whole_brain(), "sub-x003", brain_mask)
    assert not r.passed
    assert any("of the brain" in x for x in r.reasons)


def test_mask_outside_the_brain_is_caught(brain_mask):
    r = qc.check_lesion_mask(make_synthetic_lesion((0, 0, 105), 8), "sub-x004", brain_mask)
    assert not r.passed
    assert any("outside the brain" in x for x in r.reasons)


def test_tiny_mask_is_caught():
    r = qc.check_lesion_mask(make_synthetic_lesion((-26, 0, 8), 1.5), "sub-x005")
    assert not r.passed
    assert any("implausibly small" in x for x in r.reasons)


def test_cohort_volume_outlier_is_flagged_not_excluded():
    """A very large stroke is a real patient, not a data error."""
    rs = [
        qc.QCResult(subject_id=f"s{i}", volume_mm3=v)
        for i, v in enumerate([2000, 2100, 1900, 2050, 1950, 2000, 1980, 900_000])
    ]
    df = qc.check_cohort(rs)
    assert int(df["volume_outlier"].sum()) == 1
    assert df.loc[df["volume_outlier"], "subject_id"].iloc[0] == "s7"
    assert df["passed"].all()


def test_report_names_every_exclusion():
    rs = [
        qc.check_lesion_mask(_empty(), "sub-bad"),
        qc.check_lesion_mask(make_synthetic_lesion((-26, 0, 8), 10), "sub-good"),
    ]
    text = qc.qc_report(qc.check_cohort(rs))
    assert "sub-bad" in text and "EXCLUDED" in text
    assert "1 pass" in text


def test_subject_id_normalisation_matches_both_naming_styles():
    assert qc.normalise_subject_id("sub-r039s002") == "r039s002"
    assert qc.normalise_subject_id("sub-r040-s043") == "r040s043"
    assert qc.normalise_subject_id("R039S002") == "r039s002"
