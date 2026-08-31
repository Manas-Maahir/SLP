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

import time
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
    checkpoint_path: Path | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Every (feature set x model) arm, with the volume control always included.

    `checkpoint_path` writes the table after every arm. This run is hours long
    and unattended, and an arm that finishes at 01:30 should not be lost because
    a later one fails at 02:00.
    """
    wanted = list(feature_sets or ["location", "tracts", "network_edges"])
    if "volume" not in wanted:
        wanted.insert(0, "volume")  # the confound arm is not optional

    rows = []
    folds: dict[tuple[str, str], np.ndarray] = {}
    for fs in wanted:
        X = build_features(subject_ids, fs, volumes=volumes)
        available = _model_grid(X.shape[1])
        for model in models or list(available):
            if model not in available:
                continue
            started = time.perf_counter()
            scores = evaluate(X, y, fs, model, **kwargs)
            folds[(fs, model)] = scores.r2
            rows.append(scores.row())
            rows[-1]["fit_seconds"] = round(time.perf_counter() - started, 1)
            print(
                f"  {fs:26s} {model:12s} {rows[-1]['r2_median']:+.4f}"
                f"   ({rows[-1]['fit_seconds']:.0f}s)",
                flush=True,
            )
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
    out = pd.DataFrame(rows)
    out.attrs["fold_scores"] = folds  # for the paired comparison; see paired_table
    return out


def paired_table(
    fold_scores: dict[tuple[str, str], np.ndarray],
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Compare arms fold by fold, which is the comparison the design supports.

    `RepeatedKFold` runs with a fixed seed, so every arm is scored on the
    *identical* 25 outer splits. That makes the comparison paired, and a paired
    test is far more powerful -- and far more honest -- than noting that two
    marginal 95% intervals happen to be ordered. On 228 subjects those intervals
    span roughly [0.2, 0.65] for every arm and overlap almost completely, so
    ranking medians alone will call a coin flip a finding.

    One row per (arm vs baseline) pair, using each feature set's best model.
    """
    from scipy.stats import wilcoxon

    scored = results[results["r2_median"].notna()]
    if scored.empty or not fold_scores:
        return pd.DataFrame()
    best_model = (
        scored.loc[scored.groupby("feature_set")["r2_median"].idxmax()]
        .set_index("feature_set")["model"]
        .to_dict()
    )
    arms = [f for f in best_model if (f, best_model[f]) in fold_scores]
    baselines = [b for b in ("volume", "location") if b in arms]

    rows = []
    for arm in arms:
        for base in baselines:
            # "volume vs location" is the mirror of "location vs volume" and adds
            # nothing; volume is the floor every arm is measured against, not an
            # arm in its own right.
            if arm == base or arm == "volume":
                continue
            a = fold_scores[(arm, best_model[arm])]
            b = fold_scores[(base, best_model[base])]
            keep = np.isfinite(a) & np.isfinite(b)
            if keep.sum() < 5:
                continue
            d = a[keep] - b[keep]
            rows.append(
                {
                    "arm": arm,
                    "baseline": base,
                    "model": best_model[arm],
                    "median_diff": round(float(np.median(d)), 4),
                    "folds_better": int((d > 0).sum()),
                    "n_folds": int(keep.sum()),
                    "p_wilcoxon": round(float(wilcoxon(a[keep], b[keep]).pvalue), 4),
                }
            )
    table = pd.DataFrame(rows)
    if not table.empty:
        # Bonferroni over the comparisons actually made, so a nominally
        # significant arm is not read as a result on its own.
        table["p_bonferroni"] = (table["p_wilcoxon"] * len(table)).clip(upper=1.0).round(4)
    return table


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
            f"  Point estimates order disconnection ({b:+.4f}) above location ({loc:+.4f}) "
            f"and volume ({vol:+.4f}) -- but see the paired test below before reading that "
            "as a result."
        )

    # The ordering above compares medians of heavily overlapping distributions.
    # The folds are identical across arms, so the paired test is the one that
    # decides, and it is printed last because it is what should be quoted.
    paired = paired_table(results.attrs.get("fold_scores", {}), results)
    if paired.empty:
        lines.append("")
        lines.append(
            "  No paired comparison available (fold scores not retained), so the ordering "
            "above is a ranking of point estimates and nothing more."
        )
        return "\n".join(lines)

    lines += ["", "  paired across the identical outer folds -- THIS is the comparison:", ""]
    lines.append(
        f"    {'arm vs baseline':52s} {'diff':>8s} {'better':>9s} {'p':>8s} {'p(Bonf)':>9s}"
    )
    for _, r in paired.sort_values("p_wilcoxon").iterrows():
        lines.append(
            f"    {r['arm'] + ' vs ' + r['baseline']:52s} {r['median_diff']:+8.4f} "
            f"{r['folds_better']:>4d}/{r['n_folds']:<4d} {r['p_wilcoxon']:8.4f} "
            f"{r['p_bonferroni']:9.4f}"
        )

    survives = paired[(paired["p_bonferroni"] < 0.05) & (paired["median_diff"] > 0)]
    beats_location = survives[survives["baseline"] == "location"]
    lines.append("")
    if beats_location.empty:
        lines.append(
            "  **B is not distinguishable from A.** No disconnection arm beats the location "
            "baseline once the comparison is paired and corrected. The roadmap's Phase 3 exit "
            "for this is 'B ~= A -- dig into representation before declaring anything', not a "
            "win."
        )
    else:
        arms = ", ".join(sorted(set(beats_location["arm"])))
        lines.append(
            f"  **B > A survives the paired, corrected test** for: {arms}. That is the thesis "
            "holding, on this cohort, with this registration."
        )
    return "\n".join(lines)
