"""Fit the final outcome models on ARC and persist them.

Everything before this fit models inside cross-validation and discarded them.
This produces the artifact `05_report.py` needs to say anything about outcome.

    ../.venv/Scripts/python.exe 20_fit_model.py

Writes data/derived/models/wab_aq_models.joblib and a JSON model card beside it.
Three arms are fitted and all three are reported together, because lesion volume
alone reaches most of the accuracy and the report must show that.
"""

import _bootstrap  # noqa: F401
import argparse
import sys

import pandas as pd

from lesionconn import config, predict

ARC_MANIFEST = config.DERIVED_DIR / "arc" / "manifest.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=predict.DEFAULT_MODEL)
    ap.add_argument("--arms", nargs="+", default=list(predict.REPORTED_ARMS))
    args = ap.parse_args()

    if not ARC_MANIFEST.exists():
        raise SystemExit(f"{ARC_MANIFEST} not found -- run 18_build_arc_cohort.py first")
    manifest = pd.read_csv(ARC_MANIFEST)
    manifest = manifest[manifest["qc_passed"].astype(bool) & manifest["wab_aq"].notna()]
    print(f"training on {len(manifest)} ARC subjects with a WAB-AQ\n")

    models = predict.fit_final(manifest, arms=args.arms, model_name=args.model)
    path = predict.save_models(models)
    print(f"\n-> {path}")
    print(f"-> {predict.CARD_PATH}")
    print(
        f"\nApplicability: refuses any lesion under {100 * predict.MIN_FRAC_LEFT:.0f}% "
        "left-hemisphere. ARC is 228/228 left-dominant; 67 of the project's own 108 "
        "lesions are right-dominant and will be refused."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
