"""Phase 3 harness -- does it manufacture signal that is not there?

The tests that matter most here are the two negative ones. A cross-validation
harness that reports a positive R^2 on pure noise, or a permutation test that
returns a small p for a shuffled target, would produce a thesis result out of
nothing -- and with 80 to 85,905 features against ~100-228 subjects, that is the
default failure, not an exotic one.
"""

from __future__ import annotations

import numpy as np
import pytest

from lesionconn import config, modeling as md

pytestmark = pytest.mark.skipif(
    not config.PARCELLATION_PATH.exists(), reason="parcellation not built"
)


# --- network collapse -------------------------------------------------------


def test_groups_cover_every_parcel_exactly_once():
    gid, n_groups = md.network_groups()
    assert gid.shape == (config.EXPECTED_N_PARCELS,)
    assert n_groups == 17
    assert set(np.unique(gid)) == set(range(n_groups))


@pytest.mark.parametrize("how", ["mean", "max"])
def test_collapse_is_symmetric_and_the_right_shape(how):
    n = config.EXPECTED_N_PARCELS
    rng = np.random.default_rng(0)
    m = rng.random((n, n))
    m = (m + m.T) / 2
    np.fill_diagonal(m, 0)
    out = md.collapse_to_networks(m, how)
    _, n_groups = md.network_groups()
    assert out.shape == (n_groups, n_groups)
    assert np.allclose(out, out.T)


def test_collapse_sends_a_within_group_edge_to_the_diagonal():
    """An edge between two parcels of the same network is a within-network edge."""
    gid, _ = md.network_groups()
    a, b = np.where(gid == gid[0])[0][:2]
    n = config.EXPECTED_N_PARCELS
    m = np.zeros((n, n))
    m[a, b] = m[b, a] = 1.0
    out = md.collapse_to_networks(m, "max")
    assert out[gid[a], gid[a]] == 1.0
    off = out.copy()
    off[gid[a], gid[a]] = 0
    assert not off.any(), "a within-network edge must not leak into another group"


def test_max_collapse_never_below_mean_collapse():
    n = config.EXPECTED_N_PARCELS
    rng = np.random.default_rng(1)
    m = rng.random((n, n))
    m = (m + m.T) / 2
    assert (md.collapse_to_networks(m, "max") >= md.collapse_to_networks(m, "mean") - 1e-12).all()


# --- feature construction ---------------------------------------------------


def test_unknown_feature_set_raises_rather_than_guessing():
    with pytest.raises(KeyError):
        md.build_features(["sub-0001"], "not_a_feature_set")


def test_volume_arm_refuses_missing_or_impossible_volumes():
    with pytest.raises(ValueError, match="needs the volume"):
        md.build_features(["sub-0001"], "volume")
    with pytest.raises(ValueError, match="non-positive"):
        md.build_features(["sub-0001", "sub-0002"], "volume", volumes=[100.0, 0.0])


def test_volume_arm_is_log_volume():
    x = md.build_features(["a", "b"], "volume", volumes=[1000.0, 8000.0])
    assert x.shape == (2, 1)
    np.testing.assert_allclose(x.ravel(), np.log([1000.0, 8000.0]))


def test_combined_feature_sets_concatenate_columns(monkeypatch):
    """`volume+tracts` is the incremental arm, and it must actually carry both.

    It exists because volume is recoverable from the disconnection features at
    R^2 0.47-0.66 on the 108 -- so an arm that beats volume head-to-head has not
    yet shown it carries anything volume does not.
    """
    real = md.build_features

    def stub(subject_ids, which, feature_dir=None, volumes=None):
        if which == "tracts":
            return np.ones((2, 5))
        return real(subject_ids, which, feature_dir, volumes)

    monkeypatch.setattr(md, "build_features", stub)
    combined = real(["a", "b"], "volume+tracts", volumes=[1000.0, 8000.0])
    assert combined.shape == (2, 6), "the volume column and all 5 tract columns"
    np.testing.assert_allclose(combined[:, 0], np.log([1000.0, 8000.0]))
    assert (combined[:, 1:] == 1.0).all()


# --- the harness does not invent signal -------------------------------------


def test_pure_noise_scores_at_or_below_zero():
    """R^2 of a model that has learned nothing must not come back positive.

    With 200 columns and 60 rows an unregularised fit reaches R^2 = 1.0 on the
    training data every time. Held-out folds are the only thing standing between
    that and a reported result.
    """
    rng = np.random.default_rng(7)
    X = rng.standard_normal((60, 200))
    y = rng.standard_normal(60)
    scores = md.evaluate(X, y, "noise", "ridge", n_splits=5, n_repeats=2)
    assert np.median(scores.r2) <= 0.05, f"noise scored {np.median(scores.r2):+.3f}"


def test_a_real_linear_relationship_is_recovered():
    """The complement of the test above -- the harness must not be merely deaf."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((80, 20))
    y = X @ rng.standard_normal(20) + 0.1 * rng.standard_normal(80)
    scores = md.evaluate(X, y, "signal", "ridge", n_splits=5, n_repeats=2)
    assert np.median(scores.r2) > 0.7


def test_permutation_p_is_large_when_there_is_nothing_to_find():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((50, 30))
    y = rng.standard_normal(50)
    _, p, null = md.permutation_p(X, y, n_permutations=30)
    assert p > 0.05, f"noise reached p={p:.3f}"
    assert null.size == 30


def test_permutation_p_is_small_for_a_strong_relationship():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((50, 5))
    y = X @ np.array([3.0, -2.0, 1.0, 0.5, -1.5]) + 0.05 * rng.standard_normal(50)
    observed, p, _ = md.permutation_p(X, y, n_permutations=30)
    assert observed > 0.8 and p <= 1 / 31 + 1e-9


# --- the volume control is not optional -------------------------------------


def test_compare_always_includes_the_volume_arm(monkeypatch):
    """Every feature set here encodes lesion size; the comparison has to say so.

    Asked for `tracts` alone, `compare` must still run the volume control --
    otherwise a result that is purely lesion size reads as a disconnection
    finding, which is the single most likely way this project produces a wrong
    positive.
    """
    rng = np.random.default_rng(0)
    fake = {"tracts": rng.standard_normal((40, 6))}

    def stub(subject_ids, which, feature_dir=None, volumes=None):
        if which == "volume":
            return np.log(np.asarray(volumes, float)).reshape(-1, 1)
        return fake[which]

    monkeypatch.setattr(md, "build_features", stub)
    out = md.compare(
        [f"s{i}" for i in range(40)],
        rng.standard_normal(40),
        volumes=rng.uniform(500, 50_000, 40),
        feature_sets=["tracts"],
        models=["ridge"],
        n_splits=3,
        n_repeats=1,
    )
    assert set(out["feature_set"]) == {"volume", "tracts"}


def test_wide_feature_sets_are_restricted_to_linear_models():
    narrow = md._model_grid(100)
    wide = md._model_grid(md.DENSE_MODEL_MAX_FEATURES + 1)
    assert {"ridge", "elasticnet", "hist_gb"} <= set(narrow)
    assert set(wide) == {"ridge"}


def test_scores_row_reports_an_interval_not_just_a_number():
    rng = np.random.default_rng(3)
    s = md.Scores("f", "ridge", 50, 10, rng.random(20), rng.random(20))
    row = s.row()
    assert row["r2_lo"] <= row["r2_median"] <= row["r2_hi"]
    assert row["n_folds"] == 20
