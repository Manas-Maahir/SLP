"""Milestone 0 -- assign de-identified ids to the 108-lesion cohort.

Run this before anything else that touches the cohort. It writes exactly one
artifact, ``data/private/subject_map.csv``, and prints only counts -- never a
source filename, because stdout ends up in logs.

    ../.venv/Scripts/python.exe 10_deidentify.py [--source DIR]
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  -- puts src/ on sys.path

from lesionconn import config, deid

DEFAULT_SOURCE = (
    r"C:/Users/manas/Desktop/Projects/Cophe-NeuroVue-1-bug-small-fixes"
    r"/Cophe-NeuroVue-1-bug-small-fixes/Lesion_Resliced_MNI_108_patients"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="directory of lesion masks")
    ap.add_argument("--force", action="store_true", help="overwrite an existing map")
    args = ap.parse_args()

    config.ensure_dirs()
    out = config.SUBJECT_MAP_PATH

    if out.exists() and not args.force:
        existing = deid.load_subject_map()
        print(f"subject map already exists: {out}")
        print(f"  {len(existing)} subjects, ids {existing.subject_id.iloc[0]}"
              f"..{existing.subject_id.iloc[-1]}")
        print("  re-run with --force to rebuild (ids are deterministic, so it is a no-op)")
        return 0

    df = deid.build_subject_map(args.source)
    deid.save_subject_map(df)

    tokens = deid.sensitive_tokens(df)
    print(f"source directory : {args.source}")
    print(f"subjects         : {len(df)}  ({df.subject_id.iloc[0]}..{df.subject_id.iloc[-1]})")
    print(f"map written      : {out}")
    print(f"identifying tokens now under guard : {len(tokens)}")
    print()
    print("From here on, nothing outside data/private/ may contain any of them.")
    print("tests/test_deid.py greps every emitted artifact and fails on a hit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
