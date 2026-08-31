"""M5 -- the one entry point. A lesion mask in, a report and figures out.

    ../.venv/Scripts/python.exe 05_report.py <lesion.nii.gz>
    ../.venv/Scripts/python.exe 05_report.py sub-0007        # a cohort subject

Writes data/derived/reports/<subject_id>/ containing report.txt,
severed_tracts.json and four figures.

De-identification: if the file is one of the cohort masks, its de-identified id
is looked up from data/private/subject_map.csv and used throughout -- the source
filename never reaches the report, the figures or their names. For a file outside
the cohort the stem is used, and the write still refuses if it contains any
cohort identifier.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
from nilearn import datasets

from lesionconn import (
    config, deid, disconnectome as dc, predict as pr, qc, report as rp,
    streamlines as st,
)
from lesionconn.atlases import load_location_atlas, load_parcellation
from lesionconn.lesions import (
    intake_lesion_mask, lesion_parcel_overlap, location_feature_vector,
)
from lesionconn.spaces import to_canonical


def resolve(target: str) -> tuple[str, Path]:
    """(subject_id, path) for either a cohort id or a path to a mask."""
    if deid.SUBJECT_ID_RE.match(target):
        # A cohort id resolves to the SOURCE mask, not the cached canonical one.
        #
        # The cache is already on the 2mm grid, so intake would measure its
        # "native" volume there and report a resampled figure under the label the
        # report tells the reader to trust. Measured on sub-0035 that reads
        # 424 mm3 against a true native 496 mm3 -- a 14.5% error on the one
        # number docs/BUGS.md B2 exists to protect. Only the source has it.
        from lesionconn.cohort import subject_paths

        try:
            return target, deid.source_path_for(target)
        except (FileNotFoundError, KeyError):
            cached = subject_paths(target)["lesion_path"]
            if cached.exists():
                return target, cached
            raise

    path = Path(target).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    if config.SUBJECT_MAP_PATH.exists():
        mapping = deid.load_subject_map()
        hit = mapping.loc[mapping["source_path"].map(lambda p: Path(p) == path)]
        if len(hit):
            # A cohort mask given by its real path -- switch to the id immediately.
            return str(hit.iloc[0]["subject_id"]), path

    stem = path.name
    for suffix in (".nii.gz", ".nii"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem, path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lesion", help="path to a binary lesion mask, or a sub-NNNN id")
    ap.add_argument("--subject-id", default=None, help="override the label used in the report")
    ap.add_argument("--out", default=None, help="output directory")
    args = ap.parse_args()

    subject_id, path = resolve(args.lesion)
    subject_id = args.subject_id or subject_id
    out_dir = Path(args.out) if args.out else config.REPORTS_DIR / subject_id

    print(f"subject : {subject_id}")

    intake = intake_lesion_mask(path, subject_id=subject_id, quiet=False)
    lesion = intake.img

    tract_img, tract_labels = dc.load_tract_atlas()
    parc_img, parc_labels = load_parcellation()
    location_img, location_labels = load_location_atlas()

    brain = to_canonical(
        datasets.load_mni152_brain_mask(resolution=2), interpolation="nearest"
    )
    result = qc.check_lesion_mask(lesion, subject_id, brain_mask=brain)
    qc_status = "PASS" if result.passed else "EXCLUDED -- " + "; ".join(result.reasons)
    print(f"QC      : {qc_status}")

    severance = dc.severed_tracts(
        lesion, tract_img, tract_labels,
        subject_id=subject_id, native_volume_mm3=intake.native_volume_mm3,
    )
    matrix = dc.disconnection_matrix(severance, dc.load_tract_parcel_map())
    streamline_matrix = None
    if st.STREAMLINE_INDEX_PATH.exists():
        streamline_matrix = st.streamline_disconnection_matrix(lesion)
    parcel_overlap = lesion_parcel_overlap(lesion, parc_img, parc_labels)
    # Anatomy is described from the coverage-complete N=463 atlas; the N=415
    # node set only ever describes what the matrix can see. See docs/BUGS.md B1.
    location_overlap = lesion_parcel_overlap(lesion, location_img, location_labels)

    print(f"tracts  : {len(severance.hit)} involved")
    print("figures : rendering...")
    figures = rp.render_figures(
        subject_id, lesion, severance, matrix, parc_labels, out_dir
    )

    # --- outcome prediction, if a model has been fitted -------------------
    #
    # Appended to the report rather than woven into it: the disconnection
    # findings above stand on their own measurement, while this is a learned
    # model whose applicability is narrow and whose error is large. Keeping them
    # separate means a refused prediction does not invalidate the rest.
    location_vector = location_feature_vector(lesion, location_img, location_labels)
    prediction_block = ""
    try:
        models = pr.load_models()
    except FileNotFoundError:
        print("outcome : no model fitted (run scripts/20_fit_model.py)")
    else:
        refusals = pr.applicability(lesion, intake.native_volume_mm3, models)
        rows = [] if refusals else pr.predict(
            models,
            volume_mm3=intake.native_volume_mm3,
            location_vector=location_vector,
            tract_vector=severance.vector(),
            matrix=matrix,
            streamline_matrix=streamline_matrix,
        )
        prediction_block = pr.render_block(rows, refusals)
        if refusals:
            print(f"outcome : REFUSED -- {refusals[0][:70]}...")
        else:
            best = min(rows, key=lambda r: r["cv_mae"])
            print(
                f"outcome : WAB-AQ ~{best['predicted_wab_aq']:.0f} "
                f"[{best['interval_lo']:.0f}, {best['interval_hi']:.0f}] "
                f"({best['feature_set']})"
            )

    text = rp.render_text(
        subject_id, lesion, severance, parcel_overlap,
        location_overlap=location_overlap,
        matrix=matrix, streamline_matrix=streamline_matrix, parc_labels=parc_labels,
        native_volume_mm3=intake.native_volume_mm3,
        native_grid=intake.native_grid, qc_status=qc_status,
        figures=figures,
    )
    if prediction_block:
        text = text.rstrip() + chr(10) + prediction_block
    report = rp.Report(subject_id=subject_id, text=text, figures=figures, out_dir=out_dir)
    report_path = rp.write_report(report, out_dir / "report.txt")

    dc.write_severed_tracts(severance, out_dir / "severed_tracts.json")
    np.save(out_dir / f"{subject_id}_disconnection_matrix.npy", matrix)
    if streamline_matrix is not None:
        np.save(out_dir / f"{subject_id}_streamline_matrix.npy", streamline_matrix)

    # The location baseline is Model A's feature space, not a disconnection
    # measure -- emitted alongside, never mixed into the report's findings.
    np.save(out_dir / f"{subject_id}_location.npy", location_vector)

    print()
    print(f"  report : {report_path}")
    for fig in figures:
        print(f"  figure : {fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
