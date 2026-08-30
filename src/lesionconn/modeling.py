"""Phase 3 -- does disconnection predict outcome better than location?

This module is the harness, not the answer. It is deliberately cohort-agnostic:
it takes subject ids, a target vector and a feature-set name, so the same code
runs on the 108 (with lesion volume as target, to debug the machinery and to
size the confound) and on ARC (with WAB-AQ, for the real test).

Four commitments, each of which exists because the obvious alternative would
manufacture a positive result.

**Nested cross-validation.** Features outnumber subjects in every arm here --
80 features against 228 subjects at best, 85,905 against 228 at worst. Choosing
a regularisation strength on the same data that reports the score is how a
lesion-symptom paper gets an R^2 that nobody can reproduce. The inner loop picks
alpha, the outer loop scores, and they never see each other's data.

**A lesion-volume baseline in every comparison.** Bigger lesion, worse outcome:
it is the oldest confound in lesion-symptom mapping, and every feature set here
encodes volume implicitly. A disconnection model that beats location but not
volume has discovered that big strokes are bad. `compare` therefore always
includes the one-column volume arm, whether asked for or not.

**Distributions, not point scores.** Every score is reported as a median across
outer folds and repeats with a percentile interval. A single number from a
single split of 228 subjects is noise with a decimal point.

**A permutation null.** R^2 on small n can be positive by luck; the p-value is
computed against labels shuffled through the same pipeline.

Feature-set choice is constrained by a measured result, not by taste. The two
disconnection matrices anti-correlate parcel by parcel (median Spearman -0.318)
and agree at network level (+0.331) -- see docs/DISCONNECTION.md -- so
`matrix_edges` is offered but should be read as a negative control rather than
a serious arm, and `network_edges` is the honest matrix-derived feature set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .atlases import load_parcellation

# Anything wider than this gets linear models only -- a boosted tree on 85k
# columns is hours of compute to confirm what the regularised path already says.
DENSE_MODEL_MAX_FEATURES = 5_000

DEFAULT_SPLITS = 5
DEFAULT_REPEATS = 5
DEFAULT_SEED = 20260830

_GROUP_CACHE: dict[str, np.ndarray] = {}


# --------------------------------------------------------------------------
# feature construction
# --------------------------------------------------------------------------


def network_groups() -> tuple[np.ndarray, int]:
    """Parcel index -> one of 17 hemisphere x network groups."""
    if "gid" not in _GROUP_CACHE:
        _, labels = load_parcellation()
        key = labels["hemisphere"].astype(str) + "_" + labels["network"].astype(str)
        _, gid = np.unique(key.to_numpy(), return_inverse=True)
        _GROUP_CACHE["gid"] = gid
    gid = _GROUP_CACHE["gid"]
    return gid, int(gid.max()) + 1


def collapse_to_networks(matrix: np.ndarray, how: str = "mean") -> np.ndarray:
    """415x415 -> 17x17. Upper triangle of the result is the feature vector.

    `mean` is the default because a max-collapse is decided by whichever single
    parcel pair happens to be highest, and the parcel level is exactly where the
    two matrices were shown not to agree. `max` is kept because it is what the
    +0.331 network-level agreement in docs/DISCONNECTION.md was measured with,
    so the two are comparable on request.
    """
    gid, n_groups = network_groups()
    n = len(gid)
    if matrix.shape != (n, n):
        raise ValueError(
            f"matrix is {matrix.shape}, expected ({n}, {n}) -- the N={config.EXPECTED_N_PARCELS} "
            "node set. The N=463 location atlas is a different object (docs/BUGS.md B1)."
        )
    rows = gid[:, None].repeat(n, 1)
    cols = gid[None, :].repeat(n, 0)
    out = np.zeros((n_groups, n_groups), dtype=np.float64)
    if how == "max":
        np.maximum.at(out, (rows, cols), matrix)
        return out
    counts = np.zeros_like(out)
    np.add.at(out, (rows, cols), matrix)
    np.add.at(counts, (rows, cols), 1.0)
    return out / np.maximum(counts, 1.0)


def _upper(matrix: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(matrix.shape[0], 1)
    return matrix[iu]


@dataclass(frozen=True)
class FeatureSet:
    name: str
    description: str
    kind: str  # "location" | "disconnection" | "control"


FEATURE_SETS: dict[str, FeatureSet] = {
    "volume": FeatureSet("volume", "log lesion volume, one column", "control"),
    "location": FeatureSet("location", "463-dim location atlas overlap (Model A)", "location"),
    "tracts": FeatureSet("tracts", "80-dim tract severance (Model B, primary)", "disconnection"),
    "network_edges": FeatureSet(
        "network_edges", "136 network-pair edges from the inferred matrix", "disconnection"
    ),
    "network_edges_streamline": FeatureSet(
        "network_edges_streamline", "136 network-pair edges, counted matrix", "disconnection"
    ),
    "matrix_edges": FeatureSet(
        "matrix_edges", "85,905 parcel-pair edges -- known unreliable, a control", "disconnection"
    ),
}


def build_features(
    subject_ids, which: str, feature_dir: Path | None = None, volumes=None
) -> np.ndarray:
    """Stack one feature set across subjects. Missing files raise, never impute.

    `"volume+tracts"` concatenates arms. That is not a convenience: measured on
    the 108, log lesion volume is recoverable from the disconnection features at
    R^2 = 0.47 (tracts) to 0.66 (network edges), so an arm that beats volume
    head-to-head has not shown it carries anything volume does not. The
    incremental arm is the one that answers the question.
    """
    from . import cohort

    if "+" in which:
        parts = [p.strip() for p in which.split("+")]
        return np.hstack(
            [build_features(subject_ids, p, feature_dir, volumes) for p in parts]
        )

    if which not in FEATURE_SETS:
        raise KeyError(f"unknown feature set {which!r}; have {sorted(FEATURE_SETS)}")

    if which == "volume":
        if volumes is None:
            raise ValueError("the volume arm needs the volume column passed in")
        v = np.asarray(volumes, dtype=np.float64)
        if (v <= 0).any():
            raise ValueError("non-positive lesion volume")
        return np.log(v).reshape(-1, 1)

    rows = []
    for sid in subject_ids:
        paths = cohort.subject_paths(str(sid))
        if which == "location":
            rows.append(np.load(paths["location_path"]).astype(np.float64))
        elif which == "tracts":
            rows.append(np.load(paths["tract_vector_path"]).astype(np.float64))
        elif which == "matrix_edges":
            rows.append(_upper(np.load(paths["matrix_path"]).astype(np.float64)))
        elif which == "network_edges":
            rows.append(_upper(collapse_to_networks(np.load(paths["matrix_path"]))))
        elif which == "network_edges_streamline":
            rows.append(_upper(collapse_to_networks(np.load(paths["streamline_matrix_path"]))))
    return np.vstack(rows)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def _model_grid(n_features: int) -> dict[str, tuple]:
    """(estimator, param grid) per model family, pruned by feature count."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    grids: dict[str, tuple] = {
        "ridge": (
            Pipeline([("sc", StandardScaler()), ("m", Ridge(max_iter=10_000))]),
            {"m__alpha": np.logspace(-1, 5, 13)},
        )
    }
    if n_features <= DENSE_MODEL_MAX_FEATURES:
        grids["elasticnet"] = (
            Pipeline([("sc", StandardScaler()), ("m", ElasticNet(max_iter=20_000))]),
            {"m__alpha": np.logspace(-3, 1, 9), "m__l1_ratio": [0.1, 0.5, 0.9]},
        )
        grids["hist_gb"] = (
            HistGradientBoostingRegressor(random_state=DEFAULT_SEED),
            {"max_leaf_nodes": [7, 15, 31], "learning_rate": [0.05, 0.1]},
        )
    return grids


@dataclass
class Scores:
    feature_set: str
    model: str
    n_subjects: int
    n_features: int
    r2: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    mae: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    def row(self) -> dict:
        # A fold whose test split has one sample has an undefined R^2. nan-aware
        # summaries keep one such fold from erasing the other twenty-four, and
        # n_folds_undefined is reported rather than quietly dropped.
        finite = self.r2[np.isfinite(self.r2)]
        if finite.size:
            lo, hi = np.nanpercentile(self.r2, [2.5, 97.5])
            median, positive = float(np.nanmedian(self.r2)), int((finite > 0).sum())
        else:
            lo = hi = median = float("nan")
            positive = 0
        return {
            "feature_set": self.feature_set,
            "model": self.model,
            "n_subjects": self.n_subjects,
            "n_features": self.n_features,
            "r2_median": round(median, 4),
            "r2_lo": round(float(lo), 4),
            "r2_hi": round(float(hi), 4),
            "mae_median": round(float(np.nanmedian(self.mae)), 3) if self.mae.size else float("nan"),
            "r2_folds_positive": positive,
            "n_folds": int(self.r2.size),
            "n_folds_undefined": int(self.r2.size - finite.size),
        }


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    feature_set: str,
    model: str,
    n_splits: int = DEFAULT_SPLITS,
    n_repeats: int = DEFAULT_REPEATS,
    seed: int = DEFAULT_SEED,
) -> Scores:
    """Nested CV. The inner loop tunes; the outer loop scores. Never the same data."""
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold

    # R^2 on a single held-out sample is undefined. Say so here rather than
    # returning a column of nan that only surfaces after every fit has run.
    smallest_test = X.shape[0] // n_splits
    if smallest_test < 2:
        raise ValueError(
            f"{X.shape[0]} subjects across {n_splits} folds leaves {smallest_test} in a test "
            f"split, and R^2 needs at least 2. Lower n_splits, or use a larger cohort."
        )

    estimator, grid = _model_grid(X.shape[1])[model]
    outer = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    r2s, maes = [], []
    for train, test in outer.split(X):
        search = GridSearchCV(
            estimator,
            grid,
            cv=KFold(n_splits=n_splits, shuffle=True, random_state=seed),
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        search.fit(X[train], y[train])
        pred = search.best_estimator_.predict(X[test])
        r2s.append(r2_score(y[test], pred))
        maes.append(mean_absolute_error(y[test], pred))
    return Scores(feature_set, model, X.shape[0], X.shape[1], np.array(r2s), np.array(maes))


def permutation_p(
    X: np.ndarray,
    y: np.ndarray,
    model: str = "ridge",
    n_permutations: int = 200,
    n_splits: int = DEFAULT_SPLITS,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, np.ndarray]:
    """p-value for the observed R^2 against shuffled labels.

    Single-level CV with the inner grid folded in, not the full nested loop --
    200 nested runs is hours for a number whose job is to say "this could have
    happened by chance". The observed score is computed the same way, so the
    comparison is like-for-like even though it is not the headline R^2.
    """
    from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

    estimator, grid = _model_grid(X.shape[1])[model]
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    search = GridSearchCV(estimator, grid, cv=cv, scoring="r2", n_jobs=-1)

    def score(target):
        return float(np.mean(cross_val_score(search, X, target, cv=cv, scoring="r2", n_jobs=1)))

    observed = score(y)
    rng = np.random.default_rng(seed)
    null = np.array([score(rng.permutation(y)) for _ in range(n_permutations)])
    p = float((1 + (null >= observed).sum()) / (1 + n_permutations))
    return observed, p, null


def compare(
    subject_ids,
    y: np.ndarray,
    volumes,
    feature_sets: list[str] | None = None,
    models: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Every (feature set x model) arm, with the volume control always included."""
    wanted = list(feature_sets or ["location", "tracts", "network_edges"])
    if "volume" not in wanted:
        wanted.insert(0, "volume")  # the confound arm is not optional

    rows = []
    for fs in wanted:
        X = build_features(subject_ids, fs, volumes=volumes)
        available = _model_grid(X.shape[1])
        for model in models or list(available):
            if model not in available:
                continue
            rows.append(evaluate(X, y, fs, model, **kwargs).row())
            print(f"  {fs:26s} {model:12s} {rows[-1]['r2_median']:+.4f}")
    return pd.DataFrame(rows)


def verdict(results: pd.DataFrame) -> str:
    """State the Phase 3 comparison in words, including against the volume arm."""
    scored = results[results["r2_median"].notna()]
    dropped = sorted(set(results["feature_set"]) - set(scored["feature_set"]))
    if scored.empty:
        return (
            "no arm produced a defined R^2 -- every outer fold was undefined, which means "
            "the cohort is too small for this many splits, not that the features are useless"
        )
    best = scored.loc[scored.groupby("feature_set")["r2_median"].idxmax()]
    best = best.set_index("feature_set")
    lines = ["", "best arm per feature set (median R^2, 95% interval across outer folds):"]
    if dropped:
        lines.append(f"  (no defined score for: {', '.join(dropped)})")
    for fs, r in best.sort_values("r2_median", ascending=False).iterrows():
        lines.append(
            f"  {fs:26s} {r['model']:12s} {r['r2_median']:+.4f}  "
            f"[{r['r2_lo']:+.3f}, {r['r2_hi']:+.3f}]  {r['n_features']} features"
        )
    vol = best.loc["volume", "r2_median"] if "volume" in best.index else None
    loc = best.loc["location", "r2_median"] if "location" in best.index else None
    disc = [
        best.loc[f, "r2_median"]
        for f in best.index
        if FEATURE_SETS.get(f) and FEATURE_SETS[f].kind == "disconnection"
    ]

    # The incremental question, asked separately: does disconnection add anything
    # ON TOP of lesion size? Measured on the 108, volume is recoverable from the
    # disconnection features at R^2 0.47-0.66, so beating volume head-to-head is
    # not the same as carrying information volume does not already have.
    combined = [f for f in best.index if "+" in str(f) and str(f).startswith("volume+")]
    if combined and vol is not None:
        lines.append("")
        lines.append("  incremental over lesion volume alone:")
        for f in combined:
            delta = best.loc[f, "r2_median"] - vol
            verdict_word = "adds nothing" if delta <= 0.01 else f"adds {delta:+.4f}"
            lines.append(
                f"    {f:32s} {best.loc[f, 'r2_median']:+.4f}  vs volume {vol:+.4f}  -> {verdict_word}"
            )

    if vol is None or not disc:
        return "\n".join(lines)
    b = max(disc)
    lines.append("")
    if b <= vol:
        lines.append(
            f"  Disconnection ({b:+.4f}) does NOT beat lesion volume alone ({vol:+.4f}). "
            "Whatever signal is present is size, not topology."
        )
    elif loc is None:
        lines.append(
            f"  Disconnection ({b:+.4f}) beats volume ({vol:+.4f}), but the location "
            "baseline was not run -- that comparison IS the thesis test, so this says nothing "
            "about it yet."
        )
    elif b <= loc:
        lines.append(
            f"  Disconnection ({b:+.4f}) beats volume ({vol:+.4f}) but not location "
            f"({loc:+.4f}). That is B < A -- the thesis does not hold here."
        )
    else:
        lines.append(
            f"  Disconnection ({b:+.4f}) beats both volume ({vol:+.4f}) and location "
            f"({loc:+.4f}). Confirm with the permutation null before believing it."
        )
    return "\n".join(lines)
