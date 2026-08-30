"""M1 -- bring a native-space lesion mask onto the canonical grid.

The 108-lesion cohort arrived pre-normalised, so M3 was built registration-free.
ARC does not: its 228 masks sit in native T2w space across five distinct grids,
so the labelled cohort cannot reach the disconnectome without this step.

Three things here are load-bearing.

**The fixed image is the FSL MNI152 1mm grid**, which ANTsPy ships as
`get_ants_data('mni')` -- (182,218,182) LAS, origin (90,-126,-72). That is the
same grid the 108 lesions and `S35_1mm.trk` already live on, so ARC lands in
exactly the space the rest of the project asserts, and `to_canonical` does the
2mm step afterwards through the same code path every other lesion takes.
`mni_template` asserts that rather than trusting this docstring.

**Cost-function masking is not optional.** A lesion is a hole in the anatomy the
registration is trying to match, so an unmasked warp pulls healthy tissue into
it and distorts the very region being measured. The moving mask excludes the
lesion (dilated) from the similarity metric, and `mask_all_stages=True` applies
it to the rigid and affine stages too -- ANTsPy's default applies it to SyN
only, which leaves the affine free to make the same mistake.

**T2w against a T1 template is cross-modality**, so the metric must be mutual
information. ANTsPy already defaults `aff_metric` and `syn_metric` to `mattes`;
this module states it explicitly so a future default change cannot silently
switch to a correlation metric that is meaningless across contrasts.

Registration quality IS the experiment. A bad warp produces a null result that
looks like a failed thesis, so every subject carries the numbers needed to throw
it out -- see `RegistrationQC` and `scripts/17_register_arc.py`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from . import spaces
from .spaces import Grid

# The moving mask excludes the lesion plus a rim, because a hand-drawn boundary
# sits inside abnormal tissue rather than exactly on the edge of it.
LESION_DILATION_MM = 3.0

# Anything above this fraction of the robust maximum is brain in the ANTs MNI
# template (float32, 0..8364).
TEMPLATE_BRAIN_THRESHOLD = 0.10

DEFAULT_TRANSFORM = "SyNRA"

_TEMPLATE_CACHE: dict[str, object] = {}


def _ants():
    import ants

    return ants


def template_path() -> Path:
    if "path" not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE["path"] = Path(_ants().get_ants_data("mni"))
    return _TEMPLATE_CACHE["path"]


def mni_template():
    """The FSL MNI152 1mm template, asserted onto the grid this project expects.

    ANTsPy ships this file; there is nothing to download. The assertion is the
    same class of check as `assert_canonical` -- two different images are
    legitimately called "MNI" (Phase 0 caught nilearn returning ICBM 2009c), and
    picking the wrong one displaces every warped lesion by millimetres in a way
    no shape or volume check would reveal.
    """
    if "img" not in _TEMPLATE_CACHE:
        img = nib.load(str(template_path()))
        canonical = Grid.canonical()
        grid = Grid.of(img)
        assert grid.shape == (182, 218, 182), f"template shape {grid.shape}, expected 1mm MNI152"
        assert grid.orientation == canonical.orientation, (
            f"template is {grid.orientation}, canonical is {canonical.orientation}"
        )
        origin = np.asarray(grid.affine)[:3, 3]
        want = np.asarray(canonical.affine)[:3, 3]
        assert np.allclose(origin, want, atol=1e-3), (
            f"template origin {origin} != canonical origin {want} -- a different MNI"
        )
        _TEMPLATE_CACHE["img"] = img
    return _TEMPLATE_CACHE["img"]


def template_brain_mask() -> np.ndarray:
    """Boolean brain mask on the 1mm template grid, for the outside-brain check."""
    if "brain" not in _TEMPLATE_CACHE:
        data = np.asanyarray(mni_template().dataobj, dtype=np.float32)
        hi = float(np.percentile(data[data > 0], 99.5))
        _TEMPLATE_CACHE["brain"] = data > (TEMPLATE_BRAIN_THRESHOLD * hi)
    return _TEMPLATE_CACHE["brain"]


@dataclass(frozen=True)
class RegistrationQC:
    """Everything needed to decide whether to keep a warp, per subject.

    `frac_left` is the strongest check available for this cohort and it costs
    nothing: ARC is an aphasia cohort, so lesions are overwhelmingly left. A run
    that comes back right-dominant is mirrored, which is the bug class that has
    already hit this project twice (Harvard-Oxford vs Schaefer in Phase 0, then
    HCP-842's LPS vs LAS in Phase 1). Two ARC masks are PSR-oriented and the
    rest RAS, so both paths have to survive it.
    """

    subject_id: str
    transform: str
    seconds: float
    native_grid: str
    native_volume_mm3: float
    mni_volume_mm3: float
    volume_delta_pct: float
    frac_outside_brain: float
    frac_left: float
    mutual_information: float
    n_voxels_mni: int

    def as_row(self) -> dict:
        return asdict(self)


def _dilate_mm(mask: np.ndarray, zooms, radius_mm: float) -> np.ndarray:
    """Isotropic-in-millimetres dilation, so anisotropic grids behave."""
    from scipy.ndimage import distance_transform_edt

    if radius_mm <= 0 or not mask.any():
        return mask
    dist = distance_transform_edt(~mask, sampling=tuple(float(z) for z in zooms))
    return dist <= radius_mm


def frac_left_hemisphere(img) -> float:
    """Fraction of mask volume with world x < 0. Orientation-proof by construction.

    Computed from world coordinates rather than array indices precisely because
    array order is what goes wrong -- an LAS/RAS mix-up leaves the indices intact
    and flips the anatomy.
    """
    data = np.asanyarray(img.dataobj) > 0
    if not data.any():
        return float("nan")
    idx = np.argwhere(data)
    world_x = nib.affines.apply_affine(img.affine, idx)[:, 0]
    return float((world_x < 0).mean())


def _mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """Normalised MI between warped moving and fixed, inside the brain only.

    Cross-modality, so correlation would be meaningless; MI is the same family
    of metric the registration itself optimised, which makes this a consistency
    check rather than an independent one -- it catches gross failure, not
    subtle misalignment.
    """
    brain = template_brain_mask()
    x, y = a[brain].ravel(), b[brain].ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 1000:
        return float("nan")
    hist, _, _ = np.histogram2d(x, y, bins=bins)
    p = hist / hist.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    mi = float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())
    hx = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    return mi / max(float(np.sqrt(hx * hy)), 1e-12)


def save_qc_image(data: np.ndarray, affine, path: Path) -> Path:
    """Write a warped anatomical for eyeballing -- as **int16**, never float32.

    ARC ships int16 and this is a visual QC image, so rounding costs nothing.
    The dtype is not cosmetic: a float32 payload byte-greps as printable ASCII,
    and the repo-wide de-identification scan found 26 fragments of cohort first
    names inside the mantissas of a single warped volume. The same volume as
    int16 matches nothing (docs/BUGS.md B8).

    The fix is here, in the artifact, rather than in `tests/test_deid.py` --
    NIfTI payloads are still scanned in full, so a real name in a float32 NIfTI
    would still be caught.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.rint(np.clip(np.asarray(data), -32768, 32767)).astype(np.int16)
    nib.save(nib.Nifti1Image(payload, affine, dtype=np.int16), path)
    return path


def register_lesion(
    t2w_path: Path,
    lesion_path: Path,
    subject_id: str,
    transform: str = DEFAULT_TRANSFORM,
    out_dir: Path | None = None,
    save_warped_t2w: bool = False,
) -> tuple[nib.Nifti1Image, RegistrationQC]:
    """Warp one subject's T2w to MNI and carry its lesion mask along.

    Returns the lesion on the **canonical 2mm grid**, ready for the existing
    cohort pipeline, plus the QC row. The T2w and its mask must already share a
    grid -- BIDS says they do via the sidecar's SpatialReference, and this
    asserts it rather than resampling behind your back, because a silent
    resample would misplace the lesion before registration even starts.
    """
    ants = _ants()
    template = mni_template()  # assert the grid before anything is warped onto it

    t2w_img = nib.load(str(t2w_path))
    lesion_img = nib.load(str(lesion_path))
    spaces.assert_same_grid(t2w_img, lesion_img, names=("t2w", "lesion mask"))

    native_grid = Grid.of(lesion_img)
    lesion_native = np.asanyarray(lesion_img.dataobj) > 0
    if not lesion_native.any():
        raise ValueError(f"{subject_id}: empty lesion mask")
    native_volume = float(lesion_native.sum() * native_grid.voxel_volume_mm3)

    fixed = ants.image_read(str(template_path()))
    moving = ants.image_read(str(t2w_path))

    # Cost-function masking: everything EXCEPT the lesion drives the similarity.
    healthy = ~_dilate_mm(lesion_native, native_grid.voxel_sizes, LESION_DILATION_MM)
    moving_mask = ants.from_numpy(
        healthy.astype("float32"),
        origin=moving.origin,
        spacing=moving.spacing,
        direction=moving.direction,
    )

    started = time.perf_counter()
    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=transform,
        moving_mask=moving_mask,
        mask_all_stages=True,
        aff_metric="mattes",
        syn_metric="mattes",
    )
    seconds = time.perf_counter() - started

    warped = ants.apply_transforms(
        fixed=fixed,
        moving=ants.image_read(str(lesion_path)),
        transformlist=reg["fwdtransforms"],
        interpolator="nearestNeighbor",
    )

    warped_np = (warped.numpy() > 0).astype(np.uint8)
    lesion_mni = nib.Nifti1Image(warped_np, template.affine, dtype=np.uint8)
    spaces.assert_same_grid(lesion_mni, template, names=("warped lesion", "MNI 1mm template"))

    mni_grid = Grid.of(lesion_mni)
    n_vox = int(warped_np.sum())
    mni_volume = float(n_vox * mni_grid.voxel_volume_mm3)
    brain = template_brain_mask()
    outside = float(((warped_np > 0) & ~brain).sum() / n_vox) if n_vox else float("nan")

    mi = _mutual_information(
        reg["warpedmovout"].numpy().astype(np.float32),
        np.asanyarray(template.dataobj, dtype=np.float32),
    )

    if out_dir is not None and save_warped_t2w:
        save_qc_image(
            reg["warpedmovout"].numpy(),
            template.affine,
            out_dir / f"{subject_id}_T2w_in_MNI.nii.gz",
        )

    qc = RegistrationQC(
        subject_id=subject_id,
        transform=transform,
        seconds=round(seconds, 1),
        native_grid=f"{native_grid.shape} {native_grid.orientation}",
        native_volume_mm3=round(native_volume, 1),
        mni_volume_mm3=round(mni_volume, 1),
        volume_delta_pct=round(100.0 * (mni_volume - native_volume) / native_volume, 2),
        frac_outside_brain=round(outside, 4),
        frac_left=round(frac_left_hemisphere(lesion_mni), 4),
        mutual_information=round(mi, 4),
        n_voxels_mni=n_vox,
    )

    canonical = spaces.to_canonical(lesion_mni)
    spaces.assert_canonical(canonical, name=f"{subject_id} warped lesion")
    return canonical, qc
