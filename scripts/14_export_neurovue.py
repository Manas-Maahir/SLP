"""STRETCH -- hand the cohort's lesion masks to the NeuroVue viewer.

NeuroVue is not modified. Either point its LESION_DIR at a bundle written here,
or upload through the POST /api/lesions route it already has.

    # file drop (no server, no database needed)
    ../.venv/Scripts/python.exe 14_export_neurovue.py --dest ../data/derived/neurovue
    ../.venv/Scripts/python.exe 14_export_neurovue.py --dest DIR --subjects sub-0007 sub-0105

    # then, in NeuroVue:
    #   LESION_DIR=<DIR> uvicorn server:app --reload --port 8001

    # or upload to a running backend
    ../.venv/Scripts/python.exe 14_export_neurovue.py --post http://localhost:8001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from lesionconn import cohort, config, neurovue


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=str(config.DERIVED_DIR / "neurovue"))
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="de-identified ids; default is every QC-passing subject")
    ap.add_argument("--post", default=None, metavar="URL",
                    help="also upload to a running NeuroVue backend")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    manifest = cohort.load_manifest()
    if args.subjects:
        subjects = list(args.subjects)
    else:
        subjects = manifest.loc[manifest["qc_passed"], "subject_id"].tolist()
    if args.limit:
        subjects = subjects[: args.limit]

    print(f"exporting {len(subjects)} subject(s)")
    dest = neurovue.export_bundle(subjects, args.dest)
    total = sum(p.stat().st_size for p in Path(dest).rglob("*") if p.is_file())
    size = f"{total / 1e6:.1f} MB" if total >= 1e6 else f"{total / 1e3:.0f} KB"
    print(f"  bundle : {dest}  ({size})")
    print()
    print("  point NeuroVue at it without changing a line of NeuroVue:")
    print(f"    LESION_DIR={dest} uvicorn server:app --reload --port 8001")

    if args.post:
        print()
        print(f"uploading to {args.post}{neurovue.API_PATH}")
        ok = 0
        for subject_id in subjects:
            path = cohort.subject_paths(subject_id)["lesion_path"]
            try:
                neurovue.post_lesion(args.post, subject_id, path)
                ok += 1
            except Exception as exc:  # one bad upload must not stop the rest
                print(f"  {subject_id}: {type(exc).__name__}")
        print(f"  uploaded {ok}/{len(subjects)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
