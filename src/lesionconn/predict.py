"""Close the loop: a lesion mask in, a predicted WAB-AQ out.

Everything up to here fit models inside cross-validation and threw them away.
That answers "does disconnection predict outcome?" and produces nothing you can
run on a new lesion. This module fits a final model on the whole training cohort,
persists it, and scores an arbitrary mask.

Three things are non-negotiable, and each exists because the prototype would
otherwise imply more than the evidence supports.

**Applicability is enforced, not warned about.** The training cohort is ARC:
228 subjects, **228 of them left-hemisphere-dominant**, all with aphasia. WAB-AQ
is a language score. Asking this model about a right-hemisphere or cerebellar
lesion is not a degraded prediction, it is a meaningless one -- and 67 of the
project's own 108 lesions are right-dominant. `applicability` refuses those
rather than returning a plausible-looking number.

**The interval comes from held-out predictions.** Residuals are taken from
`cross_val_predict` over the training cohort, so the spread quoted is
out-of-sample error, not how well the model fits data it has already seen.

**Three models are always reported together.** Lesion volume alone reaches
R^2 0.40 on this cohort, against 0.44 for location and 0.48 for the best
disconnection arm -- and a paired test across identical folds says none of those
gaps survives correction (docs/PHASE3.md). Printing the disconnection number on
its own would present a coin flip as a capability. Printing all three makes the
margin visible to whoever reads the report.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, modeling as md

MODEL_DIR = config.DERIVED_DIR / "models"
MODEL_PATH = MODEL_DIR / "wab_aq_models.joblib"
CARD_PATH = MODEL_DIR / "wab_aq_model_card.json"

# WAB-AQ is bounded; a prediction outside it is an arithmetic artifact.
TARGET_LO, TARGET_HI = 0.0, 100.0

# Refuse below this. ARC is 228/228 left-dominant, so anything approaching the
# midline is already an extrapolation and anything right of it is a different
# question entirely.
MIN_FRAC_LEFT = 0.50

# The arms reported side by side. Order matters: the control is first so the
# reader meets the baseline before the sophisticated number.
REPORTED_ARMS = ("volume", "location", "network_edges_streamline")
DEFAULT_MODEL = "hist_gb"


@dataclass
class TrainedModel:
    """A fitted estimator plus everything needed to read its output honestly."""

    feature_set: str
    model_name: str
    n_train: int
    n_features: int
    cohort: str
    target: str
    cv_r2: float
    cv_mae: float
    residual_lo: float
    residual_hi: float
    train_volume_mm3: tuple[float, float]
    estimator: object = field(default=None, repr=False)

    def card(self) -> dict:
        d = asdict(self)
        d.pop("estimator", None)
        return d

    def interval(self, point: float) -> tuple[float, float]:
        """Held-out residual spread around a point estimate, clipped to the scale."""
        return (
            float(np.clip(point + self.residual_lo, TARGET_LO, TARGET_HI)),
            float(np.clip(point + self.residual_hi, TARGET_LO, TARGET_HI)),
        )


def frac_left(lesion_img) -> float:
    from .registration import frac_left_hemisphere

    return frac_left_hemisphere(lesion_img)


def applicability(lesion_img, volume_mm3: float, models: dict[str, TrainedModel]) -> list[str]:
    """Reasons this lesion is outside the training distribution. Empty means go.

    Refusal beats a hedged number. A right-hemisphere lesion scored by a
    left-hemisphere aphasia model produces a confident-looking WAB-AQ that means
    nothing, and a reader who sees a number will use it.
    """
    reasons = []
    fl = frac_left(lesion_img)
    if not np.isfinite(fl):
        return ["the mask is empty"]
    if fl < MIN_FRAC_LEFT:
        reasons.append(
            f"only {100 * fl:.0f}% of this lesion is in the LEFT hemisphere. The model was "
            f"trained on ARC, which is 228/228 left-dominant aphasia. WAB-AQ is a language "
            f"score and this lesion is not in the territory it was learned from."
        )
    if models:
        lo, hi = next(iter(models.values())).train_volume_mm3
        if volume_mm3 < lo or volume_mm3 > hi:
            reasons.append(
                f"lesion volume {volume_mm3:,.0f} mm3 is outside the training range "
                f"{lo:,.0f}-{hi:,.0f} mm3"
            )
    return reasons


def fit_final(
    manifest: pd.DataFrame,
    arms=REPORTED_ARMS,
    model_name: str = DEFAULT_MODEL,
    n_splits: int = md.DEFAULT_SPLITS,
    seed: int = md.DEFAULT_SEED,
) -> dict[str, TrainedModel]:
    """Fit each arm on the whole cohort, with held-out error measured first."""
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

    ids = manifest["subject_id"].tolist()
    volumes = manifest["native_volume_mm3"].to_numpy(float)
    y = manifest["wab_aq"].to_numpy(float)

    out: dict[str, TrainedModel] = {}
    for arm in arms:
        X = md.build_features(ids, arm, volumes=volumes)
        estimator, grid = md._model_grid(X.shape[1])[model_name]
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        search = GridSearchCV(
            estimator, grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
        )

        # Held-out predictions FIRST: the interval must describe out-of-sample
        # error, not how well the final fit reproduces its own training data.
        held_out = cross_val_predict(search, X, y, cv=cv, n_jobs=1)
        residuals = y - held_out
        lo, hi = np.percentile(residuals, [5, 95])

        search.fit(X, y)
        out[arm] = TrainedModel(
            feature_set=arm,
            model_name=model_name,
            n_train=len(ids),
            n_features=X.shape[1],
            cohort="ARC ds004884, 228 subjects, 228/228 left-hemisphere-dominant",
            target="WAB Aphasia Quotient (0-100)",
            cv_r2=round(float(r2_score(y, held_out)), 4),
            cv_mae=round(float(mean_absolute_error(y, held_out)), 3),
            residual_lo=round(float(lo), 3),
            residual_hi=round(float(hi), 3),
            train_volume_mm3=(float(volumes.min()), float(volumes.max())),
            estimator=search.best_estimator_,
        )
        print(
            f"  {arm:26s} R^2 {out[arm].cv_r2:+.4f}  MAE {out[arm].cv_mae:5.2f}"
            f"  90% residual [{lo:+.1f}, {hi:+.1f}]",
            flush=True,
        )
    return out


def save_models(models: dict[str, TrainedModel], path: Path | None = None) -> Path:
    import joblib

    path = Path(path or MODEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, path)
    CARD_PATH.write_text(
        json.dumps({k: v.card() for k, v in models.items()}, indent=2), encoding="utf-8"
    )
    return path


def load_models(path: Path | None = None) -> dict[str, TrainedModel]:
    import joblib

    path = Path(path or MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/20_fit_model.py")
    return joblib.load(path)


def features_for(
    arm: str,
    volume_mm3: float,
    location_vector: np.ndarray | None = None,
    tract_vector: np.ndarray | None = None,
    matrix: np.ndarray | None = None,
    streamline_matrix: np.ndarray | None = None,
) -> np.ndarray | None:
    """One feature row for a single lesion, from pieces the report already has."""
    parts = []
    for name in arm.split("+"):
        if name == "volume":
            parts.append(np.log([max(volume_mm3, 1e-6)]))
        elif name == "location" and location_vector is not None:
            parts.append(np.asarray(location_vector, float))
        elif name == "tracts" and tract_vector is not None:
            parts.append(np.asarray(tract_vector, float))
        elif name == "network_edges" and matrix is not None:
            parts.append(md._upper(md.collapse_to_networks(matrix)))
        elif name == "network_edges_streamline" and streamline_matrix is not None:
            parts.append(md._upper(md.collapse_to_networks(streamline_matrix)))
        else:
            return None
    return np.concatenate(parts).reshape(1, -1)


def predict(models: dict[str, TrainedModel], **pieces) -> list[dict]:
    """Point estimate and held-out interval per arm. Skips arms it cannot build."""
    rows = []
    for arm, model in models.items():
        X = features_for(arm, **pieces)
        if X is None or X.shape[1] != model.n_features:
            continue
        point = float(np.clip(model.estimator.predict(X)[0], TARGET_LO, TARGET_HI))
        lo, hi = model.interval(point)
        rows.append(
            {
                "feature_set": arm,
                "predicted_wab_aq": round(point, 1),
                "interval_lo": round(lo, 1),
                "interval_hi": round(hi, 1),
                "cv_r2": model.cv_r2,
                "cv_mae": model.cv_mae,
            }
        )
    return rows


def render_block(
    rows: list[dict], refusals: list[str], width: int = 78
) -> str:
    """The report section. Refusal is loud; agreement between arms is visible."""
    thin = "-" * width
    out = ["", thin, " PREDICTED OUTCOME   (WAB Aphasia Quotient, 0-100)", thin]

    if refusals:
        out += ["", "  NO PREDICTION -- this lesion is outside the training distribution:"]
        out += [f"    - {r}" for r in refusals]
        out += [
            "",
            "  A number is deliberately not shown. The model would return one, and it",
            "  would be meaningless.",
            "",
        ]
        return "\n".join(out)

    if not rows:
        out += ["", "  No model available -- run scripts/20_fit_model.py.", ""]
        return "\n".join(out)

    out += [
        "",
        f"  {'features':28s} {'predicted':>10s} {'90% interval':>16s} {'CV R2':>7s} {'MAE':>6s}",
    ]
    for r in rows:
        out.append(
            f"  {r['feature_set']:28s} {r['predicted_wab_aq']:>10.1f} "
            f"{'[' + format(r['interval_lo'], '.0f') + ', ' + format(r['interval_hi'], '.0f') + ']':>16s} "
            f"{r['cv_r2']:>7.3f} {r['cv_mae']:>6.1f}"
        )

    spread = max(r["predicted_wab_aq"] for r in rows) - min(r["predicted_wab_aq"] for r in rows)
    out += [
        "",
        "  READ THIS BEFORE THE NUMBERS. All three rows are shown because lesion",
        "  VOLUME ALONE reaches most of the accuracy here. A paired test across",
        "  identical cross-validation folds found no disconnection arm beating the",
        "  location baseline once corrected for multiple comparisons, so the",
        "  disconnection row is not a better answer -- it is a third opinion.",
        f"  The three arms differ by {spread:.1f} WAB-AQ points on this lesion.",
        "",
        f"  Typical error is about {min(r['cv_mae'] for r in rows):.0f} points on a 0-100 scale,",
        "  against a cohort standard deviation of 28. This is a research prototype,",
        "  not a prognosis.",
        "",
    ]
    return "\n".join(out)
