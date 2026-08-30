"""Phase 0 check 3: prove the alignment path with a lesion of known location.

ATLAS access is human-gated, so instead of waiting we place a synthetic sphere at
a coordinate whose anatomy we know and assert it lands where it should. If the
affine handling were wrong this is where it shows up -- a left-putamen sphere
reporting right-hemisphere or cortical parcels means everything downstream would
have been silently corrupt.
"""

import _bootstrap  # noqa: F401
import sys

from lesionconn import config, viz
from lesionconn.atlases import load_location_atlas, load_parcellation
from lesionconn.lesions import (
    lesion_parcel_overlap,
    lesion_volume_mm3,
    location_feature_vector,
    make_synthetic_lesion,
)

# Left putamen / adjacent white matter. Left hemisphere => negative x in MNI.
LESION_CENTER = (-26.0, 0.0, 8.0)
LESION_RADIUS_MM = 10.0


def main() -> int:
    print("=" * 72)
    print("SANITY CHECK: synthetic lesion -> parcel overlap")
    print("=" * 72)

    parc_img, labels = load_parcellation()
    lesion = make_synthetic_lesion(LESION_CENTER, LESION_RADIUS_MM)

    print(f"  centre (MNI mm) : {LESION_CENTER}   radius: {LESION_RADIUS_MM} mm")
    print(f"  lesion volume   : {lesion_volume_mm3(lesion):,.0f} mm^3")

    overlap = lesion_parcel_overlap(lesion, parc_img, labels)
    print()
    print("  top overlapping parcels:")
    if overlap.empty:
        print("    NONE -- the lesion hit no labelled voxel. Alignment is wrong.")
        return 1
    cols = ["index", "name", "hemisphere", "n_voxels", "volume_mm3", "frac_of_parcel"]
    print(overlap[cols].head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    hemis = overlap.head(5)["hemisphere"].tolist()
    print()
    print(f"  hemispheres of top-5 parcels: {hemis}")
    if "RH" in hemis:
        print("  FAIL: a left-hemisphere lesion is hitting right-hemisphere parcels.")
        print("        This is an affine/orientation bug -- do not proceed.")
        return 1
    print("  OK: overlap is confined to the left hemisphere, as expected.")

    loc_img, loc_labels = load_location_atlas()
    vec = location_feature_vector(lesion, loc_img, loc_labels)
    print(f"  location feature vector: shape={vec.shape} nonzero={int((vec > 0).sum())}")

    # B1 regression, checked on every run rather than only in the test suite.
    print()
    print("  white-matter lesions must not be invisible to the baseline:")
    for wm_name, wm_centre in (("L corona radiata", (-26.0, -16.0, 28.0)),
                               ("L centrum semiovale", (-28.0, -12.0, 36.0))):
        wm_vec = location_feature_vector(
            make_synthetic_lesion(wm_centre, LESION_RADIUS_MM), loc_img, loc_labels)
        status = "OK" if wm_vec.sum() > 0 else "FAIL -- all-zero feature vector"
        print(f"    {wm_name:22s} nonzero={int((wm_vec > 0).sum()):2d}  {status}")
        if wm_vec.sum() == 0:
            return 1

    config.ensure_dirs()
    p1 = viz.plot_lesion_overlay(
        lesion, config.DERIVED_DIR / "sanity_overlay.png",
        title=f"synthetic lesion @ MNI {LESION_CENTER}",
    )
    p2 = viz.plot_lesion_on_parcellation(
        lesion, parc_img, config.DERIVED_DIR / "sanity_lesion_on_parcellation.png"
    )
    p3 = viz.plot_parcellation(
        parc_img, config.DERIVED_DIR / "parcellation.png",
        title="Schaefer-400 + Harvard-Oxford subcortical",
    )
    print()
    for p in (p1, p2, p3):
        print(f"  wrote {p}")
    print()
    print("RESULT: alignment verified. Eyeball the PNGs to confirm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
