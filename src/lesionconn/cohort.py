"""Milestone 3 -- run the whole cohort through intake, QC and M3.

One row per subject in ``manifest.csv``, one cached artifact set per subject under
``data/derived/cohort/``. Nothing recomputes: a subject whose cache is present and
complete is skipped.

**Every subject is addressed by its de-identified id and nothing else.** The
source path is looked up from ``data/private/subject_map.csv`` at the last
possible moment, passed straight to ``nibabel``, and never stored, printed or
written. ``intake_lesion_mask`` is given the id explicitly so that even its
warnings carry no filename.

QC exclusions are reported, never silent. A subject that fails QC still gets its
row in the manifest with the reason, and still gets its features computed -- the
manifest is the record of what happened to the whole cohort, not a filtered list
of the survivors. Filtering is M4's decision to make, from the `qc_passed` column.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets

from . import config, deid, disconnectome as dc, qc, streamlines as st
from .atlases import load_location_atlas, load_parcellation
from .lesions import intake_lesion_mask, lesion_parcel_overlap, location_feature_vector
from .spaces import to_canonical

FEATURES_DIR = config.COHORT_DIR / "features"


def subject_paths(subject_id: str) -> dict[str, Path]:
    """Where every cached artifact for one subject lives. Ids only, no names."""
    return {
        "lesion_path": config.LESION_CACHE_DIR / f"{subject_id}_lesion.nii.gz",
        "severed_tracts_path": config.DISCONNECTOME_DIR / f"{subject_id}_severed_tracts.json",
        "tract_vector_path": FEATURES_DIR / f"{subject_id}_tract_severance.npy",
        "matrix_path": FEATURES_DIR / f"{subject_id}_disconnection_matrix.npy",
        "streamline_matrix_path": FEATURES_DIR / f"{subject_id}_streamline_matrix.npy",
        "location_path": FEATURES_DIR / f"{subject_id}_location.npy",
    }


def _ensure_dirs() -> None:
    config.ensure_dirs()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Atlases:
    """Everything the cohort loop needs, loaded once rather than 108 times."""

    tract_img: object
    tract_labels: pd.DataFrame
    tract_parcel_map: pd.DataFrame
    parc_img: object
    parc_labels: pd.DataFrame
    location_img: object
    location_labels: pd.DataFrame
    brain_mask: object
    # None when the tractogram index has not been built; the streamline matrix is
    # then skipped and the rest of the cohort run proceeds unchanged.
    streamline_index: object = None

    @classmethod
    def load(cls) -> "Atlases":
        tract_img, tract_labels = dc.load_tract_atlas()
        parc_img, parc_labels = load_parcellation()
        location_img, location_labels = load_location_atlas()
        brain = to_canonical(
            datasets.load_mni152_brain_mask(resolution=2), interpolation="nearest"
        )
        return cls(
            tract_img=tract_img, tract_labels=tract_labels,
            tract_parcel_map=dc.load_tract_parcel_map(),
            parc_img=parc_img, parc_labels=parc_labels,
            location_img=location_img, location_labels=location_labels,
            brain_mask=brain,
            streamline_index=(
                st.load_streamline_index()
                if st.STREAMLINE_INDEX_PATH.exists() else None
            ),
        )


def is_cached(subject_id: str) -> bool:
    """All artifacts present. The streamline matrix is optional -- it only exists
    when a tractogram index has been built, so its absence must not force a
    recompute of everything else."""
    optional = {"streamline_matrix_path"}
    return all(
        path.exists()
        for key, path in subject_paths(subject_id).items()
        if key not in optional
    )


def process_subject(
    subject_id: str,
    source_path: str | Path,
    atlases: Atlases,
    quiet: bool = True,
) -> dict:
    """Intake -> QC -> tract severance -> matrix -> location features, all cached.

    Returns the manifest row. `source_path` is used and discarded; it never
    reaches the returned dict.
    """
    _ensure_dirs()
    paths = subject_paths(subject_id)

    # 1. intake, on the standard path so native volume is captured before the
    #    2mm resampling inflates small lesions (docs/BUGS.md B2)
    intake = intake_lesion_mask(source_path, subject_id=subject_id, quiet=quiet)
    lesion = intake.img

    # 2. QC gate
    result = qc.check_lesion_mask(lesion, subject_id, brain_mask=atlases.brain_mask)
    result.native_volume_mm3 = intake.native_volume_mm3
    result.volume_delta_pct = intake.volume_delta_pct

    # 3. M3b -- tract severance (the primary feature object)
    sev = dc.severed_tracts(
        lesion, atlases.tract_img, atlases.tract_labels,
        subject_id=subject_id, native_volume_mm3=intake.native_volume_mm3,
    )
    tract_vector = sev.vector()

    # 4. M3a -- the parcel-pair matrix (see docs/DISCONNECTION.md)
    matrix = dc.disconnection_matrix(sev, atlases.tract_parcel_map)

    # 4b. M3a again, from real streamline endpoints rather than an inferred
    #     parcel mapping. A second, independent estimate of the same object --
    #     see src/lesionconn/streamlines.py for why neither is "the" answer.
    streamline_matrix = None
    if atlases.streamline_index is not None:
        streamline_matrix = st.streamline_disconnection_matrix(
            lesion, atlases.streamline_index
        )

    # 5. the location baseline, from the N=463 atlas -- NOT the N=415 node set,
    #    or white-matter lesions score all zeros (docs/BUGS.md B1)
    location = location_feature_vector(
        lesion, atlases.location_img, atlases.location_labels
    )
    location_overlap = lesion_parcel_overlap(
        lesion, atlases.location_img, atlases.location_labels
    )
    parcel_overlap = lesion_parcel_overlap(lesion, atlases.parc_img, atlases.parc_labels)

    # 6. cache
    nib.save(lesion, paths["lesion_path"])
    dc.write_severed_tracts(sev, paths["severed_tracts_path"])
    np.save(paths["tract_vector_path"], tract_vector)
    np.save(paths["matrix_path"], matrix)
    if streamline_matrix is not None:
        np.save(paths["streamline_matrix_path"], streamline_matrix)
    np.save(paths["location_path"], location)

    top_tract = sev.hit.iloc[0] if len(sev.hit) else None
    top_region = location_overlap.iloc[0] if len(location_overlap) else None
    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]

    row = {
        "subject_id": subject_id,
        # QC
        "qc_passed": result.passed,
        "qc_reasons": "; ".join(result.reasons),
        # volume: native is the number to trust
        "native_volume_mm3": round(intake.native_volume_mm3, 1),
        "canonical_volume_mm3": round(intake.canonical_volume_mm3, 1),
        "volume_delta_pct": round(intake.volume_delta_pct, 2),
        "n_voxels": result.n_voxels,
        "n_distinct_values": intake.n_distinct_values,
        "native_grid": intake.native_grid,
        "frac_outside_brain": round(result.frac_outside_brain, 4),
        # M3b
        "n_tracts_involved": len(sev.hit),
        "top_tract": "" if top_tract is None else str(top_tract["name"]),
        "top_tract_frac_severed": (
            0.0 if top_tract is None else round(float(top_tract["frac_severed"]), 4)
        ),
        "max_tract_frac_severed": round(float(tract_vector.max()), 4),
        # M3a
        "matrix_nonzero_pairs": int((upper > 0).sum()),
        "matrix_max": round(float(matrix.max()), 6),
        # M3a, streamline-derived
        "streamline_matrix_nonzero_pairs": (
            0 if streamline_matrix is None
            else int((streamline_matrix[np.triu_indices(streamline_matrix.shape[0], 1)] > 0).sum())
        ),
        "n_streamlines_cut": (
            0 if atlases.streamline_index is None
            else int(st.severed_streamlines(lesion, atlases.streamline_index).size)
        ),
        # location baseline (N=463)
        "n_location_regions": len(location_overlap),
        "location_nonzero": int((location > 0).sum()),
        "top_location_region": "" if top_region is None else str(top_region["name"]),
        # the N=415 node set, for reference -- B1's degenerate case is visible here
        "n_parcels_involved": len(parcel_overlap),
        **{k: str(v) for k, v in paths.items()},
    }
    return row


def build_cohort(
    subject_map: pd.DataFrame | None = None,
    atlases: Atlases | None = None,
    force: bool = False,
    limit: int | None = None,
    progress_every: int = 10,
    manifest_path: Path | None = None,
) -> pd.DataFrame:
    """Process every subject in the map. Cached subjects are re-read, not redone.

    A subject is reused only when BOTH its five cached artifacts are on disk AND
    its row survives in the previous manifest -- the artifacts alone cannot
    reconstruct the summary columns, and half-reusing would silently mix a stale
    row with fresh files.

    `manifest_path` selects which prior manifest the cache check reads. A second
    cohort (ARC) shares the per-subject caches, whose ids cannot collide with
    `sub-NNNN`, but must not share the manifest -- one file cannot hold two
    cohorts' QC without the outlier rule pooling their volume distributions.
    """
    subject_map = deid.load_subject_map() if subject_map is None else subject_map
    atlases = Atlases.load() if atlases is None else atlases
    if limit:
        subject_map = subject_map.head(limit)

    manifest_path = Path(manifest_path or config.MANIFEST_PATH)
    previous: dict[str, dict] = {}
    if not force and manifest_path.exists():
        prior = load_manifest(manifest_path)
        previous = {str(r["subject_id"]): dict(r) for _, r in prior.iterrows()}

    rows, failures, n_cached = [], [], 0
    for i, rec in enumerate(subject_map.itertuples(), start=1):
        subject_id = str(rec.subject_id)
        try:
            cached = previous.get(subject_id)
            if not force and cached is not None and is_cached(subject_id):
                rows.append(cached)
                n_cached += 1
            else:
                rows.append(process_subject(subject_id, rec.source_path, atlases))
        except Exception as exc:
            # The message may name the source file, so only the exception TYPE is
            # recorded. The id is enough to find the subject in the private map.
            failures.append({"subject_id": subject_id, "error_type": type(exc).__name__})
        if progress_every and i % progress_every == 0:
            print(f"  {i}/{len(subject_map)}")

    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        manifest = qc.flag_volume_outliers(
            manifest, volume_col="native_volume_mm3", reason_col="qc_reasons"
        )
    manifest.attrs["failures"] = failures
    manifest.attrs["n_cached"] = n_cached
    return manifest


def save_manifest(manifest: pd.DataFrame, path: Path | None = None) -> Path:
    """Write manifest.csv, refusing to emit anything carrying a patient name."""
    path = Path(path or config.MANIFEST_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = manifest.to_csv(index=False)
    deid.guard_output(text, "manifest.csv")
    path.write_text(text, encoding="utf-8", newline="")
    return path


def load_manifest(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or config.MANIFEST_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/13_build_cohort.py first"
        )
    return pd.read_csv(path)


def cohort_report(manifest: pd.DataFrame) -> str:
    """What happened to the cohort, exclusions named explicitly."""
    if manifest.empty:
        return "no subjects processed"

    n = len(manifest)
    failed = manifest[~manifest["qc_passed"]]
    lines = [
        f"{n} subjects processed",
        f"  QC pass    : {n - len(failed)}",
        f"  QC excluded: {len(failed)}",
    ]
    if "volume_outlier" in manifest:
        lines.append(f"  volume-flagged (kept, review): {int(manifest['volume_outlier'].sum())}")

    if len(failed):
        lines.append("")
        lines.append("  exclusions:")
        for row in failed.itertuples():
            lines.append(f"    {row.subject_id}: {row.qc_reasons}")

    vol = manifest["native_volume_mm3"]
    lines += [
        "",
        f"  native lesion volume mm3 : median {vol.median():,.0f}  "
        f"range {vol.min():,.0f}-{vol.max():,.0f}",
        f"  tracts involved          : median {manifest['n_tracts_involved'].median():.0f}  "
        f"range {manifest['n_tracts_involved'].min()}-{manifest['n_tracts_involved'].max()}",
        f"  subjects with no tract   : {int((manifest['n_tracts_involved'] == 0).sum())}",
        f"  all-zero location vector : {int((manifest['location_nonzero'] == 0).sum())} "
        "(docs/BUGS.md B1 -- must be 0)",
        f"  all-zero N=415 overlap   : {int((manifest['n_parcels_involved'] == 0).sum())} "
        "(expected > 0; the node set is grey matter only)",
    ]
    failures = manifest.attrs.get("failures") or []
    if failures:
        lines.append("")
        lines.append(f"  {len(failures)} subject(s) raised during processing:")
        for f in failures:
            lines.append(f"    {f['subject_id']}: {f['error_type']}")
    return "\n".join(lines)
