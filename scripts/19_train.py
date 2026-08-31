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


def run_permutations(args, ids, y, volumes, covariates, results, confounds) -> None:
    """The null, on the model family that produced the headline numbers.

    Checkpointed after every arm for the same reason `compare` is: this is a
    multi-hour unattended run (measured: ~11s per scored fit x 201 shuffles is
    ~36 min for ONE arm), and an arm that lands at 01:30 must not be lost
    because a later one fails at 02:00.
    """
    import time

    from lesionconn import deid

    out = RESULTS_DIR / f"{args.cohort}_{args.target}_permutation.csv"
    print(
        f"\npermutation null, {args.permutations} shuffles, {args.permutation_model}"
        f"  ->  {out}"
    )
    if confounds is not None:
        print("  NOTE: the null runs on the RAW target, not the residual one.")

    # Resume. This loop is hours long and has already been killed once by a
    # session ending under it. An arm's row is a complete, self-contained
    # result, so a re-run reuses what is on disk rather than paying 3 minutes
    # to recompute a number it already has -- the same rule build_cohort
    # follows for cached subjects.
    rows, done = [], set()
    if out.exists():
        prior = pd.read_csv(out)
        # Every prior row is carried into `rows`, not just the matching ones --
        # this file is rewritten wholesale after each arm, so filtering here
        # would silently delete results computed under different settings.
        rows = prior.to_dict("records")
        match = prior[
            (prior["model"] == args.permutation_model)
            & (prior["n_permutations"] == args.permutations)
        ]
        done = set(match["feature_set"])
        if done:
            print(f"  resuming -- {len(done)} arm(s) already on disk: {', '.join(sorted(done))}")
    print(
        f"  {'feature set':26s} {'observed':>9s} {'null med':>9s} "
        f"{'null p95':>9s} {'p':>8s}  mins"
    )

    for fs in results["feature_set"].unique():
        if fs in done:
            continue
        X = md.build_features(ids, fs, volumes=volumes, covariates=covariates)
        if args.permutation_model not in md._model_grid(X.shape[1]):
            print(
                f"  {fs:26s} skipped -- {args.permutation_model} is not offered "
                f"at {X.shape[1]} features"
            )
            continue
        started = time.perf_counter()
        obs, p, null = md.permutation_p(
            X,
            y,
            model=args.permutation_model,
            n_permutations=args.permutations,
            n_splits=args.splits,
        )
        mins = (time.perf_counter() - started) / 60
        rows.append(
            {
                "feature_set": fs,
                "model": args.permutation_model,
                "n_features": int(X.shape[1]),
                "n_permutations": int(args.permutations),
                "observed_r2": round(float(obs), 4),
                "null_median": round(float(np.median(null)), 4),
                "null_p95": round(float(np.percentile(null, 95)), 4),
                "p": round(float(p), 4),
                "minutes": round(mins, 1),
            }
        )
        print(
            f"  {fs:26s} {obs:+9.4f} {np.median(null):+9.4f} "
            f"{np.percentile(null, 95):+9.4f} {p:8.4f}  {mins:.1f}",
            flush=True,
        )
        text = pd.DataFrame(rows).to_csv(index=False)
        deid.guard_output(text, out.name)
        out.write_text(text, encoding="utf-8", newline="")
    print(f"\n-> {out}")


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
    ap.add_argument(
        "--permutation-model",
        default=md.CONFOUND_MODEL,
        help=(
            "model family for the permutation null. Defaults to hist_gb, which is the "
            "family the headline numbers come from -- the original run left this on ridge, "
            "so the null validated arms that were not being quoted (docs/PHASE3.md)."
        ),
    )
    ap.add_argument(
        "--out-suffix",
        default="",
        help=(
            "tag appended to the results filename. Use it when running a restricted arm "
            "list -- otherwise a narrow run overwrites the full comparison table."
        ),
    )
    ap.add_argument(
        "--residualise",
        default=None,
        help=(
            "remove a confound from the TARGET, fold-safe, and score every arm on what is "
            "left. e.g. --residualise volume  or  --residualise volume+covariates"
        ),
    )
    args = ap.parse_args()

    manifest = load_cohort(args.cohort)
    manifest = manifest[manifest["qc_passed"].astype(bool)] if "qc_passed" in manifest else manifest
    ids = manifest["subject_id"].tolist()
    volumes = manifest["native_volume_mm3"].to_numpy(dtype=float)

    # Covariates are offered only when the cohort actually carries all three.
    # The 108 have no labels and no participants.tsv, so asking for a covariate
    # arm there must fail loudly rather than silently score something else.
    covariates = None
    if all(c in manifest for c in md.COVARIATE_COLUMNS):
        covariates = manifest[list(md.COVARIATE_COLUMNS)].reset_index(drop=True)

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
    suffix = f"_resid_{args.residualise.replace('+', '_')}" if args.residualise else ""
    suffix += f"_{args.out_suffix}" if args.out_suffix else ""
    out = RESULTS_DIR / f"{args.cohort}_{args.target}{suffix}_results.csv"

    confounds = None
    if args.residualise:
        confounds = md.build_features(
            ids, args.residualise, volumes=volumes, covariates=covariates
        )
        print(
            f"residualising the target on {args.residualise} "
            f"({confounds.shape[1]} columns, {md.CONFOUND_MODEL}, refitted per training fold)\n"
        )

    results = md.compare(
        ids,
        y,
        volumes,
        checkpoint_path=out,  # every arm is written as it lands
        feature_sets=args.feature_sets,
        models=args.models,
        covariates=covariates,
        confounds=confounds,
        residualised_on=args.residualise or "",
        n_repeats=args.repeats,
        n_splits=args.splits,
    )

    if args.residualise:
        print(md.residual_verdict(results, args.residualise))
    else:
        print(md.verdict(results))

    if args.permutations:
        run_permutations(args, ids, y, volumes, covariates, results, confounds)

    text = results.to_csv(index=False)
    from lesionconn import deid

    deid.guard_output(text, out.name)
    out.write_text(text, encoding="utf-8", newline="")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
