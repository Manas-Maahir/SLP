"""Phase 0 check 2: build and persist the combined parcellation.

Emits data/atlases/parcellation.nii.gz (int16) + parcellation_labels.csv.
This fixes N -- the shape of every disconnection matrix from here on.
"""

import _bootstrap  # noqa: F401
import sys

import numpy as np
from nilearn import datasets

from lesionconn import config
from lesionconn.atlases import (
    build_location_atlas, build_parcellation, save_location_atlas, save_parcellation,
)
from lesionconn.spaces import describe, to_canonical


def main() -> int:
    print("=" * 72)
    print("BUILDING PARCELLATION (Schaefer-400 + Harvard-Oxford subcortical)")
    print("=" * 72)
    parc_img, labels = build_parcellation(verbose=True)

    data = np.asarray(parc_img.dataobj)
    present = np.unique(data)
    present = present[present > 0]
    expected = np.arange(1, len(labels) + 1)

    print()
    print(describe(parc_img, "parcellation"))
    print(f"  labels declared in CSV : {len(labels)}")
    print(f"  labels present in image: {len(present)}")

    missing = sorted(set(expected.tolist()) - set(present.tolist()))
    if missing:
        names = labels.set_index("index").loc[missing, "name"].tolist()
        pairs = list(zip(missing, names, strict=True))
        print(f"  WARNING: {len(missing)} declared labels have no voxels: {pairs[:8]}")

    # Every labelled voxel should sit inside the brain.
    brain = to_canonical(datasets.load_mni152_brain_mask(resolution=2), interpolation="nearest")
    brain_arr = np.asarray(brain.dataobj) > 0
    outside = int(np.count_nonzero((data > 0) & ~brain_arr))
    total = int(np.count_nonzero(data > 0))
    print(f"  labelled voxels        : {total:,}")
    print(f"  outside MNI brain mask : {outside:,} ({100 * outside / max(total, 1):.1f}%)")

    save_parcellation(parc_img, labels)
    print()
    print(f"  wrote {config.PARCELLATION_PATH}")
    print(f"  wrote {config.PARCELLATION_LABELS_PATH}")

    if len(labels) != config.EXPECTED_N_PARCELS:
        print()
        print(f"  NOTE: N={len(labels)} but config.EXPECTED_N_PARCELS={config.EXPECTED_N_PARCELS}.")
        print("        The CSV is the source of truth -- update the constant to match.")
    print()
    print("=" * 72)
    print("BUILDING LOCATION ATLAS (adds lateralised Juelich tracts + catch-all)")
    print("=" * 72)
    loc_img, loc_labels = build_location_atlas(parc_img, labels, verbose=True)
    save_location_atlas(loc_img, loc_labels)
    print(f"  wrote {config.LOCATION_ATLAS_PATH}")
    print(f"  wrote {config.LOCATION_ATLAS_LABELS_PATH}")

    print()
    print(f"RESULT: disconnection node set N = {len(labels)}; "
          f"location baseline N = {len(loc_labels)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
