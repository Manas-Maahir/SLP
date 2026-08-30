"""Reduce a tractogram to the index the streamline matrix needs. One-time.

Reads S35_1mm.trk in place (479,457 streamlines, FSL MNI152 1mm LAS) and caches
endpoint->parcel edges plus a voxel->streamline lookup, so each lesion afterwards
is a fast array operation rather than a re-read of 705 MB.

    ../.venv/Scripts/python.exe 15_build_streamline_index.py [--search-mm 4.0]
"""

from __future__ import annotations

import argparse
import sys
import time

import _bootstrap  # noqa: F401

import numpy as np

from lesionconn import streamlines as st
from lesionconn.atlases import load_parcellation


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tractogram", default=str(st.DEFAULT_TRACTOGRAM))
    ap.add_argument("--search-mm", type=float, default=st.ENDPOINT_SEARCH_MM,
                    help="how far from an endpoint to look for a parcel")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if st.STREAMLINE_INDEX_PATH.exists() and not args.force:
        index = st.load_streamline_index()
        print(f"already built: {st.STREAMLINE_INDEX_PATH}")
        print(f"  {index.source}  {index.n_streamlines:,} streamlines, "
              f"{index.n_assigned:,} with both ends assigned "
              f"({index.search_mm:g}mm)   --force to rebuild")
        return 0

    parc_img, parc_labels = load_parcellation()
    print(f"parcellation: {len(parc_labels)} parcels")
    t0 = time.time()
    index = st.build_streamline_index(
        args.tractogram, parc_img, parc_labels, search_mm=args.search_mm
    )
    path = st.save_streamline_index(index)
    print()
    print(f"  built in {time.time() - t0:.0f}s")
    print(f"  wrote {path}  ({path.stat().st_size / 1e6:.0f} MB)")
    print(f"  edges with at least one streamline: "
          f"{int((index.edge_totals > 0).sum()):,} of "
          f"{index.n_parcels * (index.n_parcels - 1) // 2:,} possible")
    live = index.edge_totals[index.edge_totals > 0]
    print(f"  streamlines per edge: median {np.median(live):.0f}  max {live.max():,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
