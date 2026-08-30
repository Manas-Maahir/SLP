"""Cohort QC: decide which lesion masks are fit to enter the feature store.

ATLAS is known to ship masks that are empty, non-binary, mislabelled, or that
fold microbleeds into the lesion (docs/ATLAS-notes.md). Before this module,
`load_lesion_mask` printed a warning and carried on, so a bad subject flowed
silently into the cohort.

Two rules here:
  - Exclusions are always recorded with a reason, never silent.
  - Nothing is dropped on a single automated heuristic alone; the automated
    checks flag for review, the explicit list excludes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .spaces import Grid, assert_canonical

# Subjects reported broken upstream. Keys are bare ids; "sub-" and "-" are
# stripped before matching. Sources are ATLAS issue numbers -- see
# docs/ATLAS-notes.md. An exclusion without a citation is not allowed.
KNOWN_BAD: dict[str, str] = {
    "r039s002": "mask not binary, max value 0.0099 (ATLAS issue #13)",
    "r005s081": "empty file (ATLAS issue #19)",
    "r009s003": "mask does not match the visible lesion; possible mislabel (ATLAS issue #21)",
    "r040s043": "microbleeds included in the lesion mask (ATLAS issue #23)",
}

MIN_VOLUME_MM3 = 100.0
MAX_BRAIN_FRACTION = 0.50
MAX_OUTSIDE_BRAIN_FRACTION = 0.10
VOLUME_OUTLIER_MAD = 5.0


def normalise_subject_id(subject_id: str) -> str:
    return str(subject_id).replace("sub-", "").replace("-", "").strip().lower()


@dataclass
class QCResult:
    subject_id: str
    passed: bool = True
    reasons: list[str] = field(default_factory=list)
    volume_mm3: float = 0.0
    n_voxels: int = 0
    frac_outside_brain: float = 0.0
    n_distinct_values: int = 0
    # Measured before resampling. Trust this over volume_mm3 wherever lesion
    # size is a feature or covariate -- see docs/BUGS.md B2.
    native_volume_mm3: float = 0.0
    volume_delta_pct: float = 0.0

    def fail(self, reason: str) -> "QCResult":
        self.passed = False
        self.reasons.append(reason)
        return self


def check_lesion_mask(img, subject_id: str, brain_mask=None) -> QCResult:
    """Per-subject checks. Call before the mask enters the feature store."""
    r = QCResult(subject_id=str(subject_id))

    key = normalise_subject_id(subject_id)
    if key in KNOWN_BAD:
        r.fail(f"known-bad upstream: {KNOWN_BAD[key]}")

    assert_canonical(img, f"{subject_id} lesion mask")
    data = np.asarray(img.dataobj)
    r.n_distinct_values = len(np.unique(data))

    mask = data > 0
    r.n_voxels = int(mask.sum())
    r.volume_mm3 = float(r.n_voxels * Grid.of(img).voxel_volume_mm3)

    if r.n_voxels == 0:
        return r.fail("empty mask")
    if r.volume_mm3 < MIN_VOLUME_MM3:
        r.fail(f"implausibly small: {r.volume_mm3:.0f} mm3 below {MIN_VOLUME_MM3:.0f}")

    if brain_mask is not None:
        assert_canonical(brain_mask, "brain mask")
        brain = np.asarray(brain_mask.dataobj) > 0
        r.frac_outside_brain = float(np.count_nonzero(mask & ~brain) / r.n_voxels)
        if r.frac_outside_brain > MAX_OUTSIDE_BRAIN_FRACTION:
            pct = round(100 * r.frac_outside_brain)
            r.fail(
                f"{pct}% of the mask falls outside the brain "
                "-- suspect registration or a wrong-space mask"
            )
        brain_frac = np.count_nonzero(mask & brain) / np.count_nonzero(brain)
        if brain_frac > MAX_BRAIN_FRACTION:
            pct = round(100 * brain_frac)
            r.fail(f"covers {pct}% of the brain -- suspect a tissue mask, not a lesion")

    return r


def check_cohort(results, mad_threshold: float = VOLUME_OUTLIER_MAD) -> pd.DataFrame:
    """Add cohort-relative volume outlier flags to per-subject results.

    Median absolute deviation on log volume: lesion volumes are heavily
    right-skewed, so a mean/SD rule would flag most genuinely large strokes.

    Outliers are flagged for review, never auto-excluded -- a very large stroke
    is a real patient, not a data error.
    """
    df = pd.DataFrame([vars(r) for r in results])
    if df.empty:
        return df

    df["reasons"] = df["reasons"].apply(lambda xs: "; ".join(xs))
    ok = df["volume_mm3"] > 0
    df["volume_outlier"] = False
    if int(ok.sum()) >= 3:
        logv = np.log10(df.loc[ok, "volume_mm3"])
        med = logv.median()
        mad = (logv - med).abs().median()
        if mad > 0:
            z = 0.6745 * (logv - med) / mad
            df.loc[ok, "volume_outlier"] = (z.abs() > mad_threshold).to_numpy()

    flagged = df["volume_outlier"].to_numpy()
    if flagged.any():
        existing = df.loc[flagged, "reasons"]
        df.loc[flagged, "reasons"] = [
            (e + "; volume outlier vs cohort").lstrip("; ") for e in existing
        ]
    return df


def qc_report(df: pd.DataFrame) -> str:
    if df.empty:
        return "no subjects"
    n_fail = int((~df["passed"]).sum())
    n_flag = int(df["volume_outlier"].sum()) if "volume_outlier" in df else 0
    lines = [
        f"{len(df)} subjects: {len(df) - n_fail} pass, {n_fail} excluded, "
        f"{n_flag} volume-flagged"
    ]
    for row in df[~df["passed"]].itertuples():
        lines.append(f"  EXCLUDED {row.subject_id}: {row.reasons}")
    return "\n".join(lines)


def load_and_check(path, subject_id: str | None = None, brain_mask=None):
    """Load a mask through the standard intake path, then QC it."""
    from .lesions import intake_lesion_mask

    path = Path(path)
    subject_id = subject_id or path.name.split("_")[0]
    rec = intake_lesion_mask(path, subject_id=subject_id)
    result = check_lesion_mask(rec.img, rec.subject_id, brain_mask=brain_mask)
    result.native_volume_mm3 = rec.native_volume_mm3
    result.volume_delta_pct = rec.volume_delta_pct
    return rec.img, result
