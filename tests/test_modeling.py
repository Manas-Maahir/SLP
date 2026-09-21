"""Phase 3 harness -- does it manufacture signal that is not there?

The tests that matter most here are the two negative ones. A cross-validation
harness that reports a positive R^2 on pure noise, or a permutation test that
returns a small p for a shuffled target, would produce a thesis result out of
nothing -- and with 80 to 85,905 features against ~100-228 subjects, that is the
default failure, not an exotic one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
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

    def stub(subject_ids, which, feature_dir=None, volumes=None, covariates=None):
        if which == "tracts":
            return np.ones((2, 5))
        return real(subject_ids, which, feature_dir, volumes, covariates)

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

    def stub(subject_ids, which, feature_dir=None, volumes=None, covariates=None):
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


# --- covariates: a control arm, never imputed -------------------------------


def test_covariates_encode_to_age_sex_indicator_and_log_days():
    table = pd.DataFrame(
        {"age_at_stroke": [50, 61], "sex": ["M", "f"], "wab_days": [100.0, 1000.0]}
    )
    X = md.encode_covariates(table)
    assert X.shape == (2, 3)
    np.testing.assert_allclose(X[:, 0], [50.0, 61.0])
    np.testing.assert_allclose(X[:, 1], [0.0, 1.0], err_msg="sex is the F indicator, case-folded")
    np.testing.assert_allclose(X[:, 2], np.log([100.0, 1000.0]))


@pytest.mark.parametrize(
    ("table", "match"),
    [
        ({"age_at_stroke": [50], "sex": ["M"]}, "missing"),
        ({"age_at_stroke": [np.nan], "sex": ["M"], "wab_days": [10]}, "never imputed"),
        ({"age_at_stroke": [50], "sex": ["?"], "wab_days": [10]}, "M or F"),
        ({"age_at_stroke": [50], "sex": ["M"], "wab_days": [0]}, "non-positive"),
    ],
)
def test_covariates_raise_rather_than_impute(table, match):
    """Every other arm reads a cached array and fails loudly if it is absent.

    A covariate quietly filled with the cohort mean would be the one column in
    the comparison that had been invented -- and it would sit in the arm whose
    entire job is to be a control.
    """
    with pytest.raises((KeyError, ValueError), match=match):
        md.encode_covariates(pd.DataFrame(table))


def test_covariate_arm_needs_the_table_passed_in():
    with pytest.raises(ValueError, match="needs the covariate table"):
        md.build_features(["sub-0001"], "covariates")


def test_covariates_are_a_control_not_a_disconnection_arm():
    """`verdict()` reads `kind == "disconnection"` to decide what the thesis is.

    A covariate arm landing in that set would let age and sex be reported as a
    disconnection result.
    """
    assert md.FEATURE_SETS["covariates"].kind == "control"
    assert md.is_control_arm("covariates") and md.is_control_arm("volume+covariates")
    assert not md.is_control_arm("volume+tracts")


def test_all_control_arms_stay_out_of_the_thesis_correction_family():
    """Asking a nuisance question must not weaken the thesis comparisons.

    `paired_table` Bonferroni-corrects over the rows it produces. If
    `volume+covariates` were admitted as an arm, the published Phase 3
    comparisons would get *worse* p-values because a different question was
    also asked in the same run.
    """
    rng = np.random.default_rng(5)
    folds = {
        (name, "ridge"): rng.random(25)
        for name in ("volume", "location", "tracts", "covariates", "volume+covariates")
    }
    results = pd.DataFrame(
        [
            {"feature_set": name, "model": "ridge", "r2_median": 0.4}
            for name, _ in folds
        ]
    )
    table = md.paired_table(folds, results)
    # `location` stays -- "location vs volume" is one of the published rows. The
    # two all-control arms are the ones that must not be here.
    assert set(table["arm"]) == {"tracts", "location"}
    assert not {"covariates", "volume+covariates"} & set(table["arm"])


# --- residualising a confound out of the target -----------------------------


def _confounded(seed, x_signal):
    """y = f(z) + optional g(X) + noise, with z a single confound column."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(-2, 2, 120)
    X = rng.standard_normal((120, 8))
    y = 3.0 * z**2 - z + 0.3 * rng.standard_normal(120)
    if x_signal:
        y = y + 4.0 * X[:, 0] - 2.0 * X[:, 1]
    return X, y, z.reshape(-1, 1)


def test_confounds_default_to_off_and_change_nothing():
    """The default path must be bit-identical, or every published number moves."""
    X, y, _ = _confounded(2, x_signal=True)
    a = md.evaluate(X, y, "f", "ridge", n_splits=5, n_repeats=1)
    b = md.evaluate(X, y, "f", "ridge", n_splits=5, n_repeats=1, confounds=None)
    np.testing.assert_array_equal(a.r2, b.r2)


def test_residualising_does_not_manufacture_signal():
    """X is pure noise and y is entirely a function of the confound.

    Scoring above zero here would mean the residualisation itself was leaking --
    the failure that would turn 'disconnection adds nothing' into a finding.
    """
    X, y, z = _confounded(3, x_signal=False)
    scores = md.evaluate(X, y, "noise", "ridge", n_splits=5, n_repeats=2, confounds=z)
    assert np.median(scores.r2) <= 0.05, f"noise scored {np.median(scores.r2):+.3f} on the residual"


def test_residualising_still_finds_what_the_confound_could_not_explain():
    """The complement -- it must not be merely deaf to everything."""
    X, y, z = _confounded(3, x_signal=True)
    scores = md.evaluate(X, y, "signal", "ridge", n_splits=5, n_repeats=2, confounds=z)
    assert np.median(scores.r2) > 0.5, f"real residual signal scored {np.median(scores.r2):+.3f}"


def test_the_confound_predicting_its_own_residual_is_the_control():
    """z against the z-residual of y must collapse to nothing.

    This is the check that the nuisance model actually removed what it claimed
    to, and it is why `compare` keeps the volume arm in a residualised run.
    """
    _, y, z = _confounded(4, x_signal=False)
    scores = md.evaluate(z, y, "volume", "hist_gb", n_splits=5, n_repeats=2, confounds=z)
    assert np.median(scores.r2) <= 0.05, f"the confound kept {np.median(scores.r2):+.3f}"


def test_confound_row_count_is_checked():
    X, y, z = _confounded(6, x_signal=True)
    with pytest.raises(ValueError, match="confound rows"):
        md.evaluate(X, y, "f", "ridge", n_splits=3, n_repeats=1, confounds=z[:10])


def test_parallel_permutation_null_is_identical_to_the_serial_one():
    """Speed must not move the number.

    The null is parallelised over shuffles, so a worker-visible difference --
    a permutation drawn in a different order, or a score that depends on
    anything but its own target -- would change a p-value silently. There is
    no GPU variant of this: sklearn's HistGradientBoostingRegressor is CPU
    only, and swapping in a CUDA booster would mean the null validated a
    different model than the one under test.
    """
    rng = np.random.default_rng(13)
    X = rng.standard_normal((60, 6))
    y = X @ np.array([2.0, -1.0, 0.5, 0.0, 0.0, 0.0]) + 0.5 * rng.standard_normal(60)

    o_serial, p_serial, null_serial = md.permutation_p(
        X, y, model="ridge", n_permutations=8, n_jobs=1
    )
    o_par, p_par, null_par = md.permutation_p(
        X, y, model="ridge", n_permutations=8, n_jobs=-1
    )
    np.testing.assert_array_equal(null_serial, null_par)
    assert o_serial == o_par
    assert p_serial == p_par


# --- thresholded arms (docs/THRESHOLDING.md) --------------------------------


def _symmetric(rng, n, zero_frac=0.5):
    m = rng.random((n, n)).astype(np.float32)
    m[rng.random((n, n)) < zero_frac] = 0
    m = np.triu(m, 1)
    return m + m.T


def _support(rng, n):
    s = rng.integers(0, 120, size=(n, n)).astype(np.int32)
    s = np.triu(s, 1)
    return s + s.T


def test_threshold_matrix_zeroes_exactly_the_rejected_edges_and_copies():
    rng = np.random.default_rng(10)
    m = _symmetric(rng, 30)
    before = m.copy()
    keep = _support(rng, 30) >= 50
    out = md.threshold_matrix(m, keep=keep, min_value=0.3)
    np.testing.assert_array_equal(m, before)  # input untouched
    expected = np.where(keep & (m >= 0.3), m, 0)
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == m.dtype


def test_permissive_thresholds_reproduce_todays_features_bit_for_bit():
    """sup0 / tau0 must BE the unthresholded arm -- the published arms cannot move."""
    n = config.EXPECTED_N_PARCELS
    rng = np.random.default_rng(11)
    m = _symmetric(rng, n, zero_frac=0.9)
    support = _support(rng, n)
    today = md._upper(md.collapse_to_networks(m))
    for op in ("sup0", "tau0"):
        np.testing.assert_array_equal(md.threshold_features(m, op, support), today)


def test_higher_support_threshold_keeps_a_subset_of_edges():
    rng = np.random.default_rng(12)
    support = _support(rng, 40)
    previous = None
    for k in md.SUPPORT_SWEEP:
        kept = md.edge_mask(f"sup{k}", support)
        if previous is not None:
            assert not (kept & ~previous).any(), f"sup{k} kept an edge a lower floor dropped"
        previous = kept


def test_random_and_complement_controls_have_the_right_edges():
    rng = np.random.default_rng(13)
    support = _support(rng, 40)
    kept = md.edge_mask("sup10", support)
    rand = md.edge_mask("sup10rand", support)
    comp = md.edge_mask("sup10comp", support)
    iu = np.triu_indices(40, 1)
    assert rand[iu].sum() == kept[iu].sum(), "size-matched: same number of edges"
    assert (support[rand] > 0).all(), "random edges come from the connected set"
    assert not (comp & kept).any(), "the complement is disjoint from the kept set"
    assert ((support[comp] > 0) & (support[comp] < 10)).all()
    for mask in (kept, rand, comp):
        np.testing.assert_array_equal(mask, mask.T)
    np.testing.assert_array_equal(rand, md.edge_mask("sup10rand", support))  # seeded


def test_prevalence_filter_drops_rare_columns_only():
    X = np.zeros((10, 4))
    X[:, 0] = 1.0          # present in 10/10
    X[:1, 1] = 1.0         # present in 1/10
    X[:3, 2] = 1.0         # present in 3/10
    out = md.prevalence_filter(X, 0.2)
    np.testing.assert_array_equal(out, X[:, [0, 2]])
    np.testing.assert_array_equal(md.prevalence_filter(X, 0.0), X)


def test_every_threshold_arm_is_registered_as_disconnection():
    """An unregistered arm is silently dropped by verdict() -- the worst failure here."""
    arms = [f for f in md.FEATURE_SETS if "@" in f]
    assert "network_edges_streamline@sup10" in arms
    assert "network_edges@tau0.02" in arms
    assert "network_edges_streamline@sup10rand" in arms
    assert not any(a.startswith("network_edges@sup") for a in arms), (
        "the inferred matrix has no fibre counts -- a support arm on it would be fiction"
    )
    for a in arms:
        assert md.FEATURE_SETS[a].kind == "disconnection"
        assert not md.is_control_arm(a)


def test_unregistered_threshold_arm_raises():
    with pytest.raises(KeyError):
        md.build_features(["sub-0001"], "network_edges_streamline@sup7")
