"""Build the combined parcellation that defines the N x N matrix shape.

Schaefer-400 is cortex-only, so it is stitched to the Harvard-Oxford subcortical
atlas. Both ship on the FSL MNI152 2mm grid, so no resampling is needed -- but we
assert that rather than trust it.
"""

from __future__ import annotations

import re

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.image import new_img_like

from . import config
from .spaces import Grid, assert_canonical, to_canonical


def _as_img(maps):
    """nilearn fetchers return either a path or an image depending on atlas."""
    return nib.load(maps) if isinstance(maps, (str, bytes)) else maps


def _clean_labels(labels) -> list[str]:
    out = []
    for lab in labels:
        if isinstance(lab, bytes):
            lab = lab.decode("utf-8")
        out.append(str(lab).strip())
    return out


def fetch_schaefer() -> tuple[nib.Nifti1Image, pd.DataFrame]:
    """Schaefer-400 cortical parcels, labels 1..400."""
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=config.SCHAEFER_N_ROIS,
        yeo_networks=config.SCHAEFER_YEO_NETWORKS,
        resolution_mm=config.SCHAEFER_RESOLUTION_MM,
    )
    img = _as_img(atlas.maps)
    labels = _clean_labels(atlas.labels)

    # nilearn versions differ on whether a Background entry is prepended.
    if labels and labels[0].lower() == "background":
        labels = labels[1:]
    if len(labels) != config.SCHAEFER_N_ROIS:
        raise ValueError(
            f"expected {config.SCHAEFER_N_ROIS} Schaefer labels, got {len(labels)}"
        )

    rows = []
    for i, name in enumerate(labels, start=1):
        # e.g. '7Networks_LH_Vis_1' -> hemisphere LH, network Vis
        m = re.match(r"^\d+Networks_(LH|RH)_([^_]+)_", name + "_")
        rows.append(
            {
                "index": i,
                "name": name,
                "hemisphere": m.group(1) if m else "NA",
                "network": m.group(2) if m else "NA",
                "source_atlas": "Schaefer2018-400-7Networks",
            }
        )
    return img, pd.DataFrame(rows)


def fetch_harvard_oxford_subcortical() -> tuple[nib.Nifti1Image, pd.DataFrame]:
    """Harvard-Oxford subcortical structures, reoriented and tissue-classes dropped.

    Two corrections happen here, both load-bearing:

    1. ORIENTATION. nilearn ships this atlas RAS-ordered (affine x-row +2/-90)
       while Schaefer is LAS (-2/+90). Same world box, same voxel size, x stored
       in reverse array order. Merging the raw arrays would mirror every
       subcortical structure -- a left-putamen lesion would come back labelled
       right putamen, and nothing downstream would notice. Nearest-neighbour
       resampling onto the canonical grid is exact here (identical lattice), so
       this costs nothing but must not be skipped.
    2. TISSUE CLASSES. The atlas ships 'Cerebral White Matter', 'Cerebral Cortex'
       and 'Lateral Ventricle' entries. Those are not parcels -- left in, they
       become enormous pseudo-regions that swallow most of a lesion's overlap.
    """
    atlas = datasets.fetch_atlas_harvard_oxford(config.HARVARD_OXFORD_SUB_ATLAS)
    img = to_canonical(_as_img(atlas.maps), interpolation="nearest")
    assert_canonical(img, "Harvard-Oxford subcortical (reoriented)")
    labels = _clean_labels(atlas.labels)

    kept, dropped = [], []
    for original_value, name in enumerate(labels):
        if original_value == 0 or name.lower() == "background":
            continue
        if any(p in name.lower() for p in config.HO_DROP_PATTERNS):
            dropped.append((original_value, name))
            continue
        kept.append((original_value, name))

    rows = []
    for new_i, (original_value, name) in enumerate(kept, start=1):
        lower = name.lower()
        hemi = "LH" if lower.startswith("left") else "RH" if lower.startswith("right") else "MID"
        rows.append(
            {
                "index": config.SUBCORTICAL_LABEL_OFFSET + new_i,
                "original_value": original_value,
                "name": name,
                "hemisphere": hemi,
                "network": "Subcortical",
                "source_atlas": "HarvardOxford-" + config.HARVARD_OXFORD_SUB_ATLAS,
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["dropped"] = dropped
    return img, df


def build_parcellation(verbose: bool = True):
    """Merge cortical + subcortical into one label volume on the canonical grid.

    Subcortical labels are written ONLY where Schaefer is 0, so no cortical parcel
    is ever overwritten by the coarser subcortical atlas.
    """
    ctx_img, ctx_labels = fetch_schaefer()
    sub_img, sub_labels = fetch_harvard_oxford_subcortical()

    assert_canonical(ctx_img, "Schaefer-400")
    assert_canonical(sub_img, "Harvard-Oxford subcortical")  # reoriented by its fetcher

    ctx = np.asarray(ctx_img.dataobj).astype(np.int16)
    sub = np.asarray(sub_img.dataobj).astype(np.int16)

    merged = ctx.copy()
    n_written, n_blocked = 0, 0
    for row in sub_labels.itertuples():
        region = sub == row.original_value
        free = region & (ctx == 0)
        merged[free] = row.index
        n_written += int(free.sum())
        n_blocked += int((region & (ctx != 0)).sum())

    labels = pd.concat(
        [ctx_labels, sub_labels.drop(columns=["original_value"])], ignore_index=True
    )

    if verbose:
        dropped = sub_labels.attrs.get("dropped", [])
        print(f"  cortical parcels     : {len(ctx_labels)}")
        print(f"  subcortical kept     : {len(sub_labels)}")
        print(f"  subcortical dropped  : {len(dropped)} -> {[n for _, n in dropped]}")
        print(f"  subcortical voxels written : {n_written:,}")
        print(f"  ceded to cortex on overlap : {n_blocked:,}")
        print(f"  N = {len(labels)}")

    parc_img = new_img_like(ctx_img, merged, copy_header=True)
    parc_img.set_data_dtype(np.int16)
    return parc_img, labels


def save_parcellation(parc_img, labels: pd.DataFrame) -> None:
    config.ensure_dirs()
    parc_img.to_filename(config.PARCELLATION_PATH)
    labels.to_csv(config.PARCELLATION_LABELS_PATH, index=False)


def load_parcellation() -> tuple[nib.Nifti1Image, pd.DataFrame]:
    """Load the built parcellation from disk (the M3/M4 feature substrate)."""
    if not config.PARCELLATION_PATH.exists():
        raise FileNotFoundError(
            f"{config.PARCELLATION_PATH} not found -- run scripts/01_build_parcellation.py first"
        )
    img = nib.load(config.PARCELLATION_PATH)
    labels = pd.read_csv(config.PARCELLATION_LABELS_PATH)
    assert_canonical(img, "parcellation")
    return img, labels


# ---------------------------------------------------------------------------
# Location-baseline atlas (see docs/BUGS.md B1)
#
# The parcellation above is the disconnection node set: cortical ribbon plus deep
# nuclei, which is the right substrate for M3. It is the WRONG substrate for the
# location baseline, because 38.9% of the brain carries no label and a corona
# radiata or centrum semiovale lesion -- among the commonest stroke locations --
# produces an all-zero feature vector. Model A would then have no information
# about exactly the subjects where disconnection is meant to win, biasing the
# Phase 3 comparison in our own favour.
# ---------------------------------------------------------------------------


def _world_x(grid: Grid) -> np.ndarray:
    """World-space x coordinate of every voxel, for hemisphere splitting."""
    from nibabel.affines import apply_affine

    ijk = np.stack(np.indices(grid.shape), axis=-1).reshape(-1, 3).astype(float)
    return apply_affine(grid.affine, ijk)[:, 0].reshape(grid.shape)


def fetch_juelich_wm() -> tuple[nib.Nifti1Image, list[tuple[int, str]]]:
    """Juelich fibre-tract labels only, on the canonical grid.

    Returns the raw atlas image plus the (value, name) pairs of its WM entries.
    Juelich's grey-matter areas are dropped: they duplicate Schaefer.
    """
    atlas = datasets.fetch_atlas_juelich(config.JUELICH_ATLAS)
    img = to_canonical(_as_img(atlas.maps), interpolation="nearest")
    assert_canonical(img, "Juelich (reoriented)")
    labels = _clean_labels(atlas.labels)
    wm = [
        (value, name)
        for value, name in enumerate(labels)
        if value > 0 and name.lower().startswith(config.JUELICH_WM_PREFIX)
    ]
    return img, wm


def build_location_atlas(parc_img=None, parc_labels=None, verbose: bool = True):
    """Parcellation + lateralised Juelich tracts + a hemisphere catch-all.

    Assignment is strictly first-come: grey-matter parcels are never overwritten,
    then fibre tracts fill white matter, then anything still unlabelled inside the
    brain becomes 'unassigned white matter' for that hemisphere. That last block
    is a coverage guarantee, not anatomy -- it exists so no lesion can ever be
    invisible to the baseline.
    """
    if parc_img is None:
        parc_img, parc_labels = build_parcellation(verbose=False)
    assert_canonical(parc_img, "parcellation")

    grid = Grid.of(parc_img)
    merged = np.asarray(parc_img.dataobj).astype(np.int16).copy()
    rows = parc_labels.to_dict("records")
    next_index = int(parc_labels["index"].max()) + 1

    xw = _world_x(grid)
    hemis = (("LH", xw < 0), ("RH", xw >= 0))

    # --- lateralised fibre tracts ---
    jl_img, wm_entries = fetch_juelich_wm()
    jl = np.rint(np.asarray(jl_img.dataobj)).astype(np.int64)
    n_wm = 0
    for value, name in wm_entries:
        region = jl == value
        for hemi, side in hemis:
            free = region & side & (merged == 0)
            if not free.any():
                continue
            merged[free] = next_index
            rows.append({
                "index": next_index, "name": f"{name} ({hemi})", "hemisphere": hemi,
                "network": "WhiteMatter", "source_atlas": "Juelich-" + config.JUELICH_ATLAS,
            })
            next_index += 1
            n_wm += int(free.sum())

    # --- cerebellum, from AAL3 (already lateralised, already on this grid) ---
    aal = datasets.fetch_atlas_aal()
    aal_img = to_canonical(_as_img(aal.maps), interpolation="nearest")
    assert_canonical(aal_img, "AAL3")
    aal_data = np.rint(np.asarray(aal_img.dataobj)).astype(np.int64)
    aal_labels = _clean_labels(aal.labels)
    n_cb = 0
    for name, value in zip(aal_labels, aal.indices, strict=True):
        if not any(k in name.lower() for k in config.AAL_CEREBELLAR_KEYS):
            continue
        free = (aal_data == int(value)) & (merged == 0)
        if not free.any():
            continue
        lower = name.lower()
        hemi = "LH" if lower.endswith("_l") else "RH" if lower.endswith("_r") else "MID"
        merged[free] = next_index
        rows.append({
            "index": next_index, "name": name, "hemisphere": hemi,
            "network": "Cerebellum", "source_atlas": "AAL3",
        })
        next_index += 1
        n_cb += int(free.sum())

    # --- catch-all over everything either atlas calls brain ---
    ho_all = datasets.fetch_atlas_harvard_oxford(config.HARVARD_OXFORD_SUB_ATLAS)
    brain = np.rint(
        np.asarray(to_canonical(_as_img(ho_all.maps), "nearest").dataobj)
    ).astype(np.int64) > 0
    brain |= np.asarray(
        to_canonical(datasets.load_mni152_brain_mask(resolution=2), "nearest").dataobj
    ) > 0

    n_catch = 0
    for hemi, side in hemis:
        free = brain & side & (merged == 0)
        if not free.any():
            continue
        merged[free] = next_index
        rows.append({
            "index": next_index, "name": f"Unassigned white matter ({hemi})",
            "hemisphere": hemi, "network": "Unassigned", "source_atlas": "catch-all",
        })
        next_index += 1
        n_catch += int(free.sum())

    labels = pd.DataFrame(rows)
    out = new_img_like(parc_img, merged, copy_header=True)
    out.set_data_dtype(np.int16)

    if verbose:
        # fraction of brain that carries a label -- not labelled/brain, which
        # exceeds 100% because a few cortical parcels sit outside HO's extent
        cov = 100 * np.count_nonzero((merged > 0) & brain) / np.count_nonzero(brain)
        print(f"  grey-matter parcels    : {len(parc_labels)}")
        n_tracts = sum(1 for r in rows if r["network"] == "WhiteMatter")
        print(f"  fibre-tract labels     : {len(wm_entries)} tracts -> "
              f"{n_tracts} lateralised ({n_wm:,} voxels)")
        print(f"  cerebellar parcels     : {n_cb and sum(1 for r in rows if r['network'] == 'Cerebellum')} "
              f"({n_cb:,} voxels)")
        print(f"  catch-all labels       : 2 ({n_catch:,} voxels)")
        print(f"  N_location = {len(labels)}   brain coverage {cov:.1f}%")

    return out, labels


def save_location_atlas(img, labels: pd.DataFrame) -> None:
    config.ensure_dirs()
    img.to_filename(config.LOCATION_ATLAS_PATH)
    labels.to_csv(config.LOCATION_ATLAS_LABELS_PATH, index=False)


def load_location_atlas() -> tuple[nib.Nifti1Image, pd.DataFrame]:
    """Load the location-baseline atlas (Model A's feature space)."""
    if not config.LOCATION_ATLAS_PATH.exists():
        raise FileNotFoundError(
            f"{config.LOCATION_ATLAS_PATH} not found -- run scripts/01_build_parcellation.py"
        )
    img = nib.load(config.LOCATION_ATLAS_PATH)
    labels = pd.read_csv(config.LOCATION_ATLAS_LABELS_PATH)
    assert_canonical(img, "location atlas")
    return img, labels
