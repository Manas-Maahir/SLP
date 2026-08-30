"""Phase 0 check 1: report the environment and verify every atlas shares a grid.

This is where the canonical-grid constant in config.py gets tested against
reality. If a fetched atlas disagrees, that is reported loudly rather than
silently resampled away.
"""

import _bootstrap  # noqa: F401
import importlib
import sys

from lesionconn import config
from lesionconn.spaces import Grid, GridMismatch, describe

PACKAGES = [
    "nibabel", "numpy", "nilearn", "pandas", "scipy",
    "sklearn", "matplotlib", "torch", "monai",
]


def check_packages() -> bool:
    print("=" * 72)
    print("ENVIRONMENT")
    print("=" * 72)
    print(f"python {sys.version.split()[0]}  ({sys.executable})")
    ok = True
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"  {name:<12} {getattr(mod, '__version__', 'unknown')}")
        except ImportError as exc:
            print(f"  {name:<12} MISSING ({exc})")
            ok = False
    try:
        import torch
        print(f"  torch CUDA available: {torch.cuda.is_available()} "
              f"(CPU build is intentional for Phases 0-3)")
    except ImportError:
        pass
    return ok


def _world_box(img):
    import numpy as np
    from nibabel.affines import apply_affine

    corner = np.array(img.shape[:3]) - 1
    a = apply_affine(img.affine, [0, 0, 0])
    b = apply_affine(img.affine, corner)
    return np.minimum(a, b), np.maximum(a, b)


def check_grids() -> bool:
    import numpy as np
    from nilearn import datasets

    from lesionconn.atlases import _as_img
    from lesionconn.spaces import to_canonical

    print()
    print("=" * 72)
    print("COORDINATE SPACES")
    print("=" * 72)
    canon = Grid.canonical()
    print(f"canonical (config.py): {canon}")
    print()

    print("fetching atlases (first run downloads; cached afterwards)...")
    images = {
        "Schaefer-400 2mm": _as_img(
            datasets.fetch_atlas_schaefer_2018(
                n_rois=config.SCHAEFER_N_ROIS,
                yeo_networks=config.SCHAEFER_YEO_NETWORKS,
                resolution_mm=config.SCHAEFER_RESOLUTION_MM,
            ).maps
        ),
        "Harvard-Oxford sub": _as_img(
            datasets.fetch_atlas_harvard_oxford(config.HARVARD_OXFORD_SUB_ATLAS).maps
        ),
        "MNI152 template 2mm": datasets.load_mni152_template(resolution=2),
        "MNI152 brain mask 2mm": datasets.load_mni152_brain_mask(resolution=2),
    }

    print()
    ok = True
    for name, img in images.items():
        grid = Grid.of(img)
        print("  " + describe(img, name))
        if grid.matches(canon):
            print("      -> ON canonical grid, no action needed")
            continue

        lo_a, hi_a = _world_box(img)
        lo_c, hi_c = _world_box(
            __import__("nibabel").Nifti1Image(np.zeros(canon.shape, np.int8), canon.affine)
        )
        same_box = np.allclose(lo_a, lo_c, atol=1e-3) and np.allclose(hi_a, hi_c, atol=1e-3)

        moved = to_canonical(img, "nearest" if "MNI152 template" not in name else "continuous")
        fixed = Grid.of(moved).matches(canon)

        if same_box:
            print(f"      -> same world box, different array order ({grid.orientation} vs "
                  f"{canon.orientation}); reorienting is exact")
        else:
            print(f"      -> different world box  x[{lo_a[0]:.0f},{hi_a[0]:.0f}] vs "
                  f"x[{lo_c[0]:.0f},{hi_c[0]:.0f}]; resampling interpolates")
        print(f"      -> to_canonical() lands on grid: {fixed}")
        if not fixed:
            ok = False

    print()
    print("-" * 72)
    print("ANATOMY PRESERVED THROUGH REORIENTATION?")
    print("-" * 72)
    ok = _check_anatomy_preserved() and ok

    print()
    if ok:
        print("OK: every atlas reaches the canonical grid with anatomy intact.")
    else:
        print("FAIL: an atlas could not be brought onto the canonical grid.")
    return ok


def _check_anatomy_preserved() -> bool:
    """Left must stay left. This is the check that catches an L/R flip.

    A mirrored subcortical atlas is invisible to every shape, count and volume
    check -- the only thing that reveals it is asking where a named structure
    actually sits in world coordinates.
    """
    import numpy as np
    from nibabel.affines import apply_affine

    from lesionconn.atlases import fetch_harvard_oxford_subcortical

    img, labels = fetch_harvard_oxford_subcortical()
    data = np.asarray(img.dataobj)
    ok = True
    n_checked = 0
    # NB: the image still carries ORIGINAL Harvard-Oxford values; `index` is the
    # post-merge label. Comparing against `index` here would match nothing and
    # pass vacuously.
    for _, row in labels.iterrows():
        if not any(k in row["name"].lower() for k in ("putamen", "hippocampus", "thalamus")):
            continue
        ijk = np.argwhere(data == row["original_value"])
        if not len(ijk):
            print(f"  {row['name']:<22} NO VOXELS -- cannot verify")
            ok = False
            continue
        n_checked += 1
        x = apply_affine(img.affine, ijk.mean(axis=0))[0]
        expect_negative = row["hemisphere"] == "LH"
        good = (x < 0) if expect_negative else (x > 0)
        side = "left" if x < 0 else "right"
        print(f"  {row['name']:<22} centre x = {x:+6.1f} mm ({side})  "
              f"{'OK' if good else 'FLIPPED'}")
        ok = ok and good

    if n_checked < 6:
        print(f"  FAIL: only {n_checked} structures verified; this check is not doing its job.")
        return False
    return ok


def check_guardrail_fires() -> bool:
    """A guardrail that never fires is not a guardrail."""
    import numpy as np
    import nibabel as nib

    print()
    print("=" * 72)
    print("GUARDRAIL SELF-TEST")
    print("=" * 72)
    canon = Grid.canonical()
    good = nib.Nifti1Image(np.zeros(canon.shape, np.int16), canon.affine)

    shifted_affine = canon.affine.copy()
    shifted_affine[0, 3] += 2.0  # one voxel to the left
    shifted = nib.Nifti1Image(np.zeros(canon.shape, np.int16), shifted_affine)

    from lesionconn.spaces import assert_same_grid

    try:
        assert_same_grid(good, shifted, names=("aligned", "shifted-by-1-voxel"))
    except GridMismatch:
        print("  OK: a 2mm shift is caught by assert_same_grid.")
        return True
    print("  FAIL: assert_same_grid did NOT catch a 2mm shift.")
    return False


if __name__ == "__main__":
    results = [check_packages(), check_grids(), check_guardrail_fires()]
    print()
    print("=" * 72)
    print("RESULT: " + ("ALL CHECKS PASSED" if all(results) else "CHECKS FAILED"))
    print("=" * 72)
    sys.exit(0 if all(results) else 1)
