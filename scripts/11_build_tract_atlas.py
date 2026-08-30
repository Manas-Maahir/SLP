"""Milestone 1 -- put the HCP-842 tractography atlas on the canonical grid.

80 named tracts, 1mm LPS -> 2mm LAS. Laterality is asserted at build time, so a
mirrored atlas cannot be written to disk. One-time; cached in data/atlases/.

    ../.venv/Scripts/python.exe 11_build_tract_atlas.py [--source DIR]
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from lesionconn import config, disconnectome as dc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(dc.DEFAULT_TRACT_SOURCE))
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--native-com-tol-mm", type=float, default=dc.NATIVE_COM_TOL_MM,
        help="max centre-of-mass shift allowed during resampling; raise it only to "
             "MEASURE the distribution, never to make a real mirroring failure pass",
    )
    args = ap.parse_args()

    if config.TRACT_ATLAS_PATH.exists() and not args.force:
        img, labels = dc.load_tract_atlas()
        print(f"already built: {config.TRACT_ATLAS_PATH}")
        print(f"  {img.shape}  {len(labels)} tracts   (--force to rebuild)")
        return 0

    print("building tract atlas")
    img, labels = dc.build_tract_atlas(
        args.source, native_com_tol_mm=args.native_com_tol_mm
    )
    dc.save_tract_atlas(img, labels)

    print()
    print(f"  wrote {config.TRACT_ATLAS_PATH}  {img.shape} {img.get_data_dtype()}")
    print(f"  wrote {config.TRACT_LABELS_PATH}")
    n = labels["hemisphere"].value_counts()
    print(f"  hemispheres: L={n.get('L', 0)} R={n.get('R', 0)} midline={n.get('M', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
