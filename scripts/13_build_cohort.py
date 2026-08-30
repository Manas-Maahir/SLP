"""Milestone 3 -- run the whole cohort through intake, QC and M3.

    ../.venv/Scripts/python.exe 13_build_cohort.py [--force] [--limit N]

Emits data/derived/cohort/manifest.csv plus, per subject, a canonical lesion
mask, severed_tracts.json, the 80-dim tract-severance vector, the 415x415
disconnection matrix and the 463-dim location baseline -- all named by
de-identified id. QC exclusions are printed explicitly and kept in the manifest.
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from lesionconn import cohort, config, deid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    ap.add_argument("--limit", type=int, default=None, help="first N subjects only")
    args = ap.parse_args()

    if not config.SUBJECT_MAP_PATH.exists():
        print("no subject map -- run 10_deidentify.py first", file=sys.stderr)
        return 1

    subject_map = deid.load_subject_map()
    print(f"cohort: {len(subject_map)} subjects")
    print("loading atlases...")
    atlases = cohort.Atlases.load()
    print(f"  tracts {len(atlases.tract_labels)}  parcels {len(atlases.parc_labels)}  "
          f"location regions {len(atlases.location_labels)}")
    print()

    manifest = cohort.build_cohort(
        subject_map, atlases, force=args.force, limit=args.limit
    )
    path = cohort.save_manifest(manifest)

    print()
    print(cohort.cohort_report(manifest))
    print()
    n_cached = manifest.attrs.get("n_cached", 0)
    if n_cached:
        print(f"  {n_cached} subject(s) reused from cache")
    print(f"  manifest: {path}")
    print(f"  features: {cohort.FEATURES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
