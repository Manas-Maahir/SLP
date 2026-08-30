"""Phase 0 data step: metadata-first pull of the Aphasia Recovery Cohort.

ATLAS R2.0 ships no clinical outcome scores (withheld for privacy in R2.0), so
ARC (OpenNeuro ds004884, 230 chronic stroke survivors with WAB scores) is the
labelled cohort for the Phase 3 thesis test.

ARC has 902 scanning sessions. Do NOT blind-sync it. This pulls metadata only,
reports which subjects carry both a behavioural score and a lesion mask, and
leaves the bulk decision to a human.
"""

import _bootstrap  # noqa: F401
import sys

from lesionconn import config

DATASET = "ds004884"
TARGET = config.RAW_DIR / "arc"
# Names vary per dataset (README vs README.txt), so each is fetched
# independently and a miss is reported rather than aborting the whole pull.
REQUIRED_INCLUDES = ["participants.tsv", "dataset_description.json"]
OPTIONAL_INCLUDES = ["participants.json", "README.txt", "README", "CHANGES"]


def fetch_metadata() -> bool:
    try:
        import openneuro
    except ImportError:
        print("openneuro-py not installed:  pip install openneuro-py")
        return False

    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"pulling metadata for {DATASET} -> {TARGET}")

    def grab(name: str) -> bool:
        try:
            openneuro.download(dataset=DATASET, target_dir=TARGET, include=[name])
            return True
        except Exception as exc:
            print(f"  - {name}: {type(exc).__name__.replace('RuntimeError', 'not present')}")
            return False

    ok = True
    for name in REQUIRED_INCLUDES:
        if grab(name):
            print(f"  + {name}")
        else:
            ok = False
    for name in OPTIONAL_INCLUDES:
        if grab(name):
            print(f"  + {name}")
    return ok


def summarise() -> int:
    import pandas as pd

    tsv = TARGET / "participants.tsv"
    if not tsv.exists():
        print(f"no participants.tsv at {tsv}")
        return 1

    df = pd.read_csv(tsv, sep="\t")
    print()
    print("=" * 72)
    print(f"ARC participants.tsv: {len(df)} rows, {len(df.columns)} columns")
    print("=" * 72)
    print("columns:")
    for c in df.columns:
        n_present = int(df[c].notna().sum())
        print(f"  {c:<28} non-null {n_present:>4}/{len(df)}")

    outcome_cols = [c for c in df.columns
                    if any(k in c.lower() for k in ("wab", "aq", "aphasia", "fluency", "comprehension"))]
    print()
    print(f"candidate outcome columns: {outcome_cols or 'NONE FOUND -- inspect manually'}")
    for c in outcome_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            print(f"  {c}: n={int(s.notna().sum())} "
                  f"min={s.min():.1f} max={s.max():.1f} mean={s.mean():.1f}")

    return 0


def fetch_lesion_masks() -> bool:
    """Pull derivatives/lesion_masks -- ~25 kB per subject, so all 228 is ~6 MB."""
    import openneuro

    print()
    print("=" * 72)
    print("LESION MASKS (derivatives/lesion_masks)")
    print("=" * 72)
    try:
        openneuro.download(dataset=DATASET, target_dir=TARGET,
                           include=["derivatives/lesion_masks"])
        return True
    except Exception as exc:
        print(f"download failed: {type(exc).__name__}: {exc}")
        return False


def probe_mask_space() -> int:
    """Determine empirically what space the masks are in.

    The filenames carry no BIDS `space-` entity, which implies native T2w space
    -- but infer that from the headers, not the naming convention.
    """
    import nibabel as nib

    from lesionconn.spaces import Grid

    masks = sorted((TARGET / "derivatives" / "lesion_masks").rglob("*_mask.nii.gz"))
    if not masks:
        print("no lesion masks found on disk")
        return 1

    print(f"  found {len(masks)} lesion masks")
    canon = Grid.canonical()
    grids, on_canon = {}, 0
    for m in masks:
        g = Grid.of(nib.load(m))
        key = (g.shape, tuple(round(float(v), 2) for v in g.voxel_sizes), g.orientation)
        grids[key] = grids.get(key, 0) + 1
        on_canon += g.matches(canon)

    print(f"  distinct grids: {len(grids)}   on canonical grid: {on_canon}/{len(masks)}")
    print("  most common grids:")
    for (shape, vox, orient), n in sorted(grids.items(), key=lambda kv: -kv[1])[:5]:
        print(f"    n={n:>4}  shape={shape} voxel={vox} orient={orient}")

    print()
    if on_canon == len(masks):
        print("  => masks are already in canonical MNI space. No registration needed.")
    else:
        print("  => masks are in NATIVE space, on per-subject grids.")
        print("     ARC therefore REQUIRES registration (M1) before M3 can run on it.")
        print("     This pulls Phase 4 work forward for the labelled cohort. ATLAS R2.0")
        print("     (pre-normalized) remains the zero-registration path for building M3.")
    return 0


if __name__ == "__main__":
    if not fetch_metadata():
        sys.exit(1)
    rc = summarise()
    if fetch_lesion_masks():
        rc = probe_mask_space() or rc
    sys.exit(rc)
