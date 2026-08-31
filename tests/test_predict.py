"""The prediction layer -- mostly tests that it REFUSES when it should.

A prototype that emits a plausible WAB-AQ for a right-hemisphere lesion is worse
than one that emits nothing: the number looks like the others, and whoever reads
the report has no way to tell it apart. So the load-bearing tests here are the
negative ones, and the strictest is that a refused prediction must not leak a
number into the text anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from lesionconn import predict as pr


def _blob(world_x: float, shape=(60, 60, 60)) -> "object":
    """A small mask centred at a given world x, on an LAS-like affine."""
    import nibabel as nib

    affine = np.array([[-1.0, 0, 0, 30], [0, 1.0, 0, -30], [0, 0, 1.0, -30], [0, 0, 0, 1]])
    data = np.zeros(shape, np.uint8)
    grid = np.argwhere(np.ones(shape, bool))
    world = nib.affines.apply_affine(affine, grid)
    hit = (
        (np.abs(world[:, 0] - world_x) < 5)
        & (np.abs(world[:, 1]) < 5)
        & (np.abs(world[:, 2]) < 5)
    )
    for i, j, k in grid[hit]:
        data[i, j, k] = 1
    assert data.any()
    return nib.Nifti1Image(data, affine)


def _model(**kw) -> pr.TrainedModel:
    defaults = dict(
        feature_set="volume",
        model_name="hist_gb",
        n_train=228,
        n_features=1,
        cohort="ARC",
        target="WAB-AQ",
        cv_r2=0.42,
        cv_mae=17.2,
        residual_lo=-35.0,
        residual_hi=31.0,
        train_volume_mm3=(150.0, 400_000.0),
    )
    defaults.update(kw)
    return pr.TrainedModel(**defaults)


# --- applicability ----------------------------------------------------------


def test_a_right_hemisphere_lesion_is_refused():
    """ARC is 228/228 left-dominant and WAB-AQ is a language score.

    67 of this project's own 108 lesions are right-dominant, so this is the
    common case, not an edge case.
    """
    reasons = pr.applicability(_blob(+20.0), 5000.0, {"volume": _model()})
    assert reasons
    assert "LEFT hemisphere" in reasons[0]


def test_a_left_hemisphere_lesion_in_range_is_accepted():
    assert pr.applicability(_blob(-20.0), 5000.0, {"volume": _model()}) == []


def test_an_empty_mask_is_refused():
    import nibabel as nib

    empty = nib.Nifti1Image(np.zeros((10, 10, 10), np.uint8), np.eye(4))
    assert pr.applicability(empty, 0.0, {"volume": _model()}) == ["the mask is empty"]


def test_a_volume_outside_the_training_range_is_refused():
    reasons = pr.applicability(_blob(-20.0), 5.0, {"volume": _model()})
    assert any("training range" in r for r in reasons)


# --- the interval -----------------------------------------------------------


def test_the_interval_is_clipped_to_the_wab_aq_scale():
    """WAB-AQ is bounded 0-100; a prediction outside it is an arithmetic artifact."""
    lo, hi = _model().interval(95.0)
    assert lo >= pr.TARGET_LO and hi <= pr.TARGET_HI
    lo, hi = _model().interval(5.0)
    assert lo >= pr.TARGET_LO and hi <= pr.TARGET_HI


def test_the_interval_brackets_the_point_estimate():
    lo, hi = _model().interval(50.0)
    assert lo < 50.0 < hi


# --- feature assembly -------------------------------------------------------


def test_features_for_returns_none_when_a_piece_is_missing():
    """Silently substituting zeros would score a lesion against absent evidence."""
    assert pr.features_for("location", volume_mm3=100.0) is None
    assert pr.features_for("network_edges", volume_mm3=100.0) is None


def test_volume_feature_is_log_volume():
    x = pr.features_for("volume", volume_mm3=1000.0)
    assert x.shape == (1, 1)
    np.testing.assert_allclose(x[0, 0], np.log(1000.0))


def test_combined_arms_concatenate():
    x = pr.features_for(
        "volume+location", volume_mm3=1000.0, location_vector=np.ones(463)
    )
    assert x.shape == (1, 464)


# --- the report block: a refusal must never carry a number ------------------


def test_a_refusal_shows_no_predicted_value():
    """The whole point. A number in a refused block would be read as a prediction."""
    text = pr.render_block([], ["only 0% of this lesion is in the LEFT hemisphere"], width=78)
    assert "NO PREDICTION" in text
    assert "deliberately not shown" in text
    # The section heading still says PREDICTED OUTCOME -- what must be absent is
    # the results TABLE, which is what a reader would take a number from.
    assert "90% interval" not in text
    assert "CV R2" not in text


def test_a_refusal_wins_even_if_rows_are_somehow_passed():
    rows = [{"feature_set": "volume", "predicted_wab_aq": 80.0, "interval_lo": 50.0,
             "interval_hi": 100.0, "cv_r2": 0.42, "cv_mae": 17.2}]
    text = pr.render_block(rows, ["out of distribution"], width=78)
    assert "80" not in text
    assert "NO PREDICTION" in text


def test_the_block_always_names_the_volume_baseline():
    """Reporting the disconnection number alone would sell a coin flip."""
    rows = [
        {"feature_set": "volume", "predicted_wab_aq": 78.9, "interval_lo": 44.0,
         "interval_hi": 100.0, "cv_r2": 0.424, "cv_mae": 17.2},
        {"feature_set": "network_edges_streamline", "predicted_wab_aq": 95.6,
         "interval_lo": 66.0, "interval_hi": 100.0, "cv_r2": 0.519, "cv_mae": 15.0},
    ]
    text = pr.render_block(rows, [], width=78)
    assert "VOLUME ALONE" in text
    assert "not a better answer" in text
    assert "78.9" in text and "95.6" in text
    assert "16.7" in text, "the spread between arms must be stated"


def test_the_block_says_it_is_not_a_prognosis():
    rows = [{"feature_set": "volume", "predicted_wab_aq": 60.0, "interval_lo": 30.0,
             "interval_hi": 90.0, "cv_r2": 0.42, "cv_mae": 17.2}]
    assert "not a prognosis" in pr.render_block(rows, [], width=78)


def test_no_model_fitted_is_stated_rather_than_silently_empty():
    assert "run scripts/20_fit_model.py" in pr.render_block([], [], width=78)


# --- the persisted model, if one exists -------------------------------------


@pytest.mark.skipif(not pr.MODEL_PATH.exists(), reason="no model fitted")
def test_the_persisted_model_card_records_its_training_cohort():
    models = pr.load_models()
    assert models
    for m in models.values():
        assert "left-hemisphere-dominant" in m.cohort
        assert m.n_train > 0
        assert m.cv_mae > 0
        card = m.card()
        assert "estimator" not in card, "the card is metadata, not the fitted object"
