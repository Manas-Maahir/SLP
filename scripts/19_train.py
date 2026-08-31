"""Phase 3 -- Model A (location) vs Model B (disconnection), honestly scored.

Two modes, and the difference matters.

    --target volume       the harness check, runnable tonight on the 108
    --target wab_aq       the thesis test, needs the registered ARC cohort

**The volume run is not the thesis test and must never be reported as one.**
The 108 have no outcome labels and none are being invented. Lesion volume is a
real measured quantity, and asking how well each feature set predicts it answers
a question worth answering before the real target arrives: *how much of each
feature set is just lesion size?* A disconnection vector that reconstructs
volume at R^2 = 0.9 will predict any size-correlated outcome for the wrong
reason, and WAB-AQ is size-correlated. It also serves as an end-to-end check of
the machinery -- the volume arm predicting log volume from log volume must come
back at R^2 = 1.0, and if it does not, the harness is broken, not the biology.

    ../.venv/Scripts/python.exe 19_train.py --target volume
    ../.venv/Scripts/python.exe 19_train.py --target wab_aq --cohort arc
"""

import _bootstrap  # noqa: F401
import argparse
import sys

import numpy as np
import pandas as pd

from lesionconn import cohort, config, modeling as md

ARC_MANIFEST = config.DERIVED_DIR / "arc" / "manifest.csv"
RESULTS_DIR = config.DERIVED_DIR / "modeling"


def load_cohort(name: str) -> pd.DataFrame:
    if name == "arc":
        if not ARC_MANIFEST.exists():
            raise SystemExit(
                f"{ARC_MANIFEST} not found -- run 17_register_arc.py then 18_build_arc_cohort.py"
            )
        return pd.read_csv(ARC_MANIFEST)
    return cohort.load_manifest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="108", choices=["108", "arc"])
    ap.add_argument("--target", default="volume", choices=["volume", "wab_aq"])
    ap.add_argument(
        "--feature-sets",
        nargs="+",
        default=[
            "location",
            "tracts",
            "network_edges",
            "network_edges_streamline",
            # the incremental arms -- does disconnection add anything over size?
            "volume+location",
            "volume+tracts",
            "volume+network_edges",
        ],
    )
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--repeats", type=int, default=md.DEFAULT_REPEATS)
    ap.add_argument("--splits", type=int, default=md.DEFAULT_SPLITS)
    ap.add_argument("--permutations", type=int, default=0)
    args = ap.parse_args()

    manifest = load_cohort(args.cohort)
    manifest = manifest[manifest["qc_passed"].astype(bool)] if "qc_passed" in manifest else manifest
    ids = manifest["subject_id"].tolist()
    volumes = manifest["native_volume_mm3"].to_numpy(dtype=float)

    if args.target == "wab_aq":
        if "wab_aq" not in manifest:
            raise SystemExit(
                "no wab_aq column -- this cohort has no outcome labels, and none will be invented"
            )
        y = manifest["wab_aq"].to_numpy(dtype=float)
        header = "TARGET: WAB Aphasia Quotient  --  the Phase 3 thesis test"
    else:
        y = np.log(volumes)
        header = (
            "TARGET: log lesion volume  --  HARNESS CHECK, NOT THE THESIS TEST.\n"
            "  Reads as: how much of each feature set is simply lesion size?\n"
            "  The volume arm is the positive control and must return R^2 = 1.000."
        )

    print(f"cohort {args.cohort}: {len(ids)} subjects")
    print(header)
    print(f"\ny: mean {y.mean():.2f}  sd {y.std():.2f}  range [{y.min():.2f}, {y.max():.2f}]\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.cohort}_{args.target}_results.csv"

    results = md.compare(
        ids,
        y,
        volumes,
        checkpoint_path=out,  # every arm is written as it lands
        feature_sets=args.feature_sets,
        models=args.models,
        n_repeats=args.repeats,
        n_splits=args.splits,
    )

    print(md.verdict(results))

    if args.permutations:
        print(f"\npermutation null, {args.permutations} shuffles, ridge:")
        for fs in results["feature_set"].unique():
            X = md.build_features(ids, fs, volumes=volumes)
            obs, p, null = md.permutation_p(X, y, n_permutations=args.permutations)
            print(
                f"  {fs:26s} observed R^2 {obs:+.4f}   null median {np.median(null):+.4f}   p={p:.4f}"
            )

    text = results.to_csv(index=False)
    from lesionconn import deid

    deid.guard_output(text, out.name)
    out.write_text(text, encoding="utf-8", newline="")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
