"""Milestone 2 -- which parcels each tract connects, and how strongly.

This builds the approximation the N x N disconnection matrix rests on. Read
docs/DISCONNECTION.md before trusting anything it produces: we have tract
VOLUMES, not streamline endpoints, so "connects" is inferred from "abuts".

    ../.venv/Scripts/python.exe 12_tract_parcel_map.py [--min-affinity 0.15]
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from lesionconn import config, disconnectome as dc
from lesionconn.atlases import load_parcellation


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-affinity", type=float, default=dc.MIN_AFFINITY)
    ap.add_argument("--dilation", type=int, default=dc.CONTACT_DILATION_VOXELS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if config.TRACT_PARCEL_MAP_PATH.exists() and not args.force:
        df = dc.load_tract_parcel_map()
        print(f"already built: {config.TRACT_PARCEL_MAP_PATH}  ({len(df):,} pairs)")
        print("  --force to rebuild")
        return 0

    tract_img, tract_labels = dc.load_tract_atlas()
    parc_img, parc_labels = load_parcellation()
    print(f"tracts {len(tract_labels)}   parcels {len(parc_labels)}")

    df = dc.build_tract_parcel_map(
        tract_img, tract_labels, parc_img, parc_labels,
        dilation=args.dilation, min_affinity=args.min_affinity,
    )
    dc.save_tract_parcel_map(df)
    print(f"  wrote {config.TRACT_PARCEL_MAP_PATH}")

    # A few named tracts, so the mapping can be eyeballed against anatomy rather
    # than only against its own summary statistics.
    name_of = dict(zip(parc_labels["index"], parc_labels["name"], strict=True))
    print()
    print("spot check -- strongest parcels per tract:")
    for tract in ("Uncinate_Fasciculus_L", "Arcuate_Fasciculus_L", "Cortico_Spinal_Tract_L"):
        sub = df[df["tract_name"] == tract].nlargest(4, "affinity")
        print(f"  {tract}")
        for r in sub.itertuples():
            print(f"    {name_of.get(r.parcel_index, '?')[:46]:48s} affinity {r.affinity:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
