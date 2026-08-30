"""Run the registered ARC masks through the existing M3 pipeline.

Nothing about the disconnectome changes for ARC. The masks arrive on the
canonical grid from 17_register_arc.py, so this is the same `build_cohort` the
108 go through, pointed at a different subject map and a different manifest.

The registration QC is joined on, and its exclusions are applied HERE rather
than inside the registration script, so that a warp can be re-judged without
re-registering 228 subjects. Exclusions are reported explicitly and never
silently dropped.
"""

import _bootstrap  # noqa: F401
import argparse
import sys
from pathlib import Path

import pandas as pd

from lesionconn import cohort, config

ARC_DIR = config.DERIVED_DIR / "arc"
LESION_DIR = ARC_DIR / "lesions"
QC_PATH = ARC_DIR / "registration_qc.csv"
MANIFEST_PATH = ARC_DIR / "manifest.csv"

# A warp is rejected on anatomy, not on a similarity score. Aphasia is a
# left-hemisphere syndrome, so a lesion that lands mostly right has either been
# mirrored or badly warped; and a lesion sitting largely outside the template
# brain has been pushed out of the head. Both thresholds are deliberately loose
# -- they exist to catch failure, not to prune the cohort toward a nicer answer.
MIN_FRAC_LEFT_FOR_APHASIA = 0.20
MAX_FRAC_OUTSIDE_BRAIN = 0.20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not QC_PATH.exists():
        raise SystemExit(f"{QC_PATH} not found -- run 17_register_arc.py first")

    reg_qc = pd.read_csv(QC_PATH)
    labels = pd.read_csv(config.RAW_DIR / "arc" / "participants.tsv", sep="\t")
    labels = labels.rename(columns={"participant_id": "subject_id"})

    ok = reg_qc[reg_qc["ok"].astype(bool)].copy()
    print(f"{len(reg_qc)} registered, {len(ok)} without error")

    right = ok["frac_left"] < MIN_FRAC_LEFT_FOR_APHASIA
    outside = ok["frac_outside_brain"] > MAX_FRAC_OUTSIDE_BRAIN
    for name, mask in (("right-dominant lesion", right), ("outside template brain", outside)):
        if mask.any():
            print(f"  excluded, {name}: {mask.sum()}")
            for sid in ok.loc[mask, "subject_id"]:
                print(f"      {sid}")
    ok = ok[~(right | outside)]
    print(f"{len(ok)} pass registration QC\n")

    subject_map = pd.DataFrame(
        {
            "subject_id": ok["subject_id"],
            "source_path": [str(LESION_DIR / f"{s}_lesion.nii.gz") for s in ok["subject_id"]],
        }
    )
    missing = [p for p in subject_map["source_path"] if not Path(p).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} warped masks named in the QC file are not on disk, e.g. {missing[0]}"
        )

    manifest = cohort.build_cohort(
        subject_map=subject_map,
        force=args.force,
        limit=args.limit,
        manifest_path=MANIFEST_PATH,
    )
    if manifest.empty:
        raise SystemExit("no subjects processed")

    # build_cohort reuses rows from the previous manifest, which already carries
    # these columns -- so drop them before re-attaching, or the second run merges
    # a duplicate onto a duplicate and pandas refuses.
    label_cols = ["wab_aq", "wab_type", "age_at_stroke", "sex", "wab_days"]
    reg_cols = {
        "frac_left": "reg_frac_left",
        "frac_outside_brain": "reg_frac_outside_brain",
        "mutual_information": "reg_mutual_information",
        "transform": "reg_transform",
    }
    manifest = manifest.drop(columns=label_cols + list(reg_cols.values()), errors="ignore")
    manifest = manifest.merge(labels[["subject_id", *label_cols]], on="subject_id", how="left")
    manifest = manifest.merge(
        ok[["subject_id", *reg_cols]].rename(columns=reg_cols),
        on="subject_id",
        how="left",
    )

    cohort.save_manifest(manifest, MANIFEST_PATH)
    # cohort_report is the manifest-shaped summary; qc.qc_report takes the
    # per-subject QC frame (`passed`), which is a different object.
    print(cohort.cohort_report(manifest))
    labelled = manifest["wab_aq"].notna().sum()
    print(f"\nsubjects with a WAB-AQ: {labelled}/{len(manifest)}")
    print(f"-> {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
