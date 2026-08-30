"""Grid conformance guardrail.

The silent failure mode of this whole project is two volumes that are both
"in MNI" but sit on different grids. `nilearn.datasets.load_mni152_template()`
returns the ICBM 2009c asymmetric grid; Schaefer and Harvard-Oxford ship on the
FSL MNI152 grid. Intersecting a lesion with a misaligned parcellation produces a
disconnection matrix that looks entirely plausible and is quietly wrong, and
nothing downstream would ever flag it.

Every function here exists to turn that class of mistake into an exception.
"""

from __future__ import annotations

from dataclasses import dataclass

import nibabel as nib
import numpy as np
from nilearn.image import resample_img

from . import config

LABEL_INTERPOLATION = "nearest"


class GridMismatch(AssertionError):
    """Two images that must share a grid do not."""


@dataclass(frozen=True, eq=False)
class Grid:
    """A spatial grid: array shape plus voxel-to-world affine."""

    shape: tuple[int, int, int]
    affine: np.ndarray

    @classmethod
    def of(cls, img) -> "Grid":
        # Copy and freeze: a Grid must never be a live view onto someone else's
        # affine, or mutating that image silently redefines this Grid.
        affine = np.array(img.affine, dtype=float)
        affine.flags.writeable = False
        return cls(tuple(int(v) for v in img.shape[:3]), affine)

    @classmethod
    def canonical(cls) -> "Grid":
        return cls(config.CANONICAL_SHAPE, config.CANONICAL_AFFINE)

    @property
    def voxel_sizes(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in np.sqrt((self.affine[:3, :3] ** 2).sum(axis=0)))

    @property
    def voxel_volume_mm3(self) -> float:
        return float(abs(np.linalg.det(self.affine[:3, :3])))

    @property
    def orientation(self) -> str:
        return "".join(nib.aff2axcodes(self.affine))

    def matches(self, other: "Grid", tol: float = 1e-4) -> bool:
        return self.shape == other.shape and np.allclose(self.affine, other.affine, atol=tol)

    def __str__(self) -> str:
        vx = "x".join(f"{v:g}" for v in self.voxel_sizes)
        return f"shape={self.shape} voxel={vx}mm orient={self.orientation}"


def describe(img, name: str = "image") -> str:
    """One-line human summary, for env checks and error messages."""
    grid = Grid.of(img)
    data = np.asarray(img.dataobj)
    nonzero = int(np.count_nonzero(data))
    return (
        f"{name}: {grid} dtype={data.dtype} "
        f"nonzero={nonzero:,} ({100 * nonzero / data.size:.1f}% of {data.size:,})"
    )


def assert_same_grid(a, b, tol: float = 1e-4, names: tuple[str, str] = ("a", "b")) -> None:
    """Raise GridMismatch, with a readable diff, unless a and b share a grid."""
    ga, gb = Grid.of(a), Grid.of(b)
    if ga.matches(gb, tol=tol):
        return

    lines = [f"{names[0]} and {names[1]} are not on the same grid."]
    if ga.shape != gb.shape:
        lines.append(f"  shape:  {names[0]}={ga.shape}  {names[1]}={gb.shape}")
    if not np.allclose(ga.affine, gb.affine, atol=tol):
        delta = np.abs(ga.affine - gb.affine).max()
        lines.append(f"  affine: max abs difference {delta:g} (tol {tol:g})")
        lines.append(f"    {names[0]} affine:\n{np.array2string(ga.affine, precision=3)}")
        lines.append(f"    {names[1]} affine:\n{np.array2string(gb.affine, precision=3)}")
    lines.append("  Fix by resampling one onto the other with to_grid(), not by ignoring this.")
    raise GridMismatch("\n".join(lines))


def assert_canonical(img, name: str = "image") -> None:
    """Raise unless img is on the project's canonical grid."""
    canon = Grid.canonical()
    grid = Grid.of(img)
    if not grid.matches(canon):
        raise GridMismatch(
            f"{name} is not on the canonical grid.\n"
            f"  expected: {canon}\n  got:      {grid}\n"
            "  Use to_canonical() first."
        )


MAX_PLAUSIBLE_LABELS = 2048


def is_integer_like(img, sample_limit: int = 2_000_000) -> bool:
    """True if every voxel value is a whole number."""
    data = np.asarray(img.dataobj)
    if np.issubdtype(data.dtype, np.integer):
        return True
    flat = data.reshape(-1)
    if flat.size > sample_limit:
        flat = flat[:: max(1, flat.size // sample_limit)]
    return bool(np.all(flat == np.round(flat)))


def looks_like_labels(img, max_labels: int = MAX_PLAUSIBLE_LABELS) -> bool:
    """True if the image holds labels or a mask, rather than continuous anatomy.

    Integer dtype alone is NOT enough: MNI template T1s ship as int16 with
    thousands of intensity levels, and refusing to interpolate those would be
    wrong. A label volume is integer-valued *and* has few distinct values.
    """
    if not is_integer_like(img):
        return False
    return len(np.unique(np.asarray(img.dataobj))) <= max_labels


def to_grid(img, grid: Grid, interpolation: str = LABEL_INTERPOLATION):
    """Resample img onto `grid`. Refuses to interpolate label data continuously."""
    if interpolation != LABEL_INTERPOLATION and looks_like_labels(img):
        raise ValueError(
            f"refusing interpolation={interpolation!r} on integer-valued (label or mask) "
            f"data -- it would invent labels that do not exist. Use {LABEL_INTERPOLATION!r}."
        )
    if Grid.of(img).matches(grid):
        # Already aligned -- but return a distinct image, never the caller's own
        # object. Handing back the input makes "output" and "input" the same
        # array, so mutating one silently mutates the other.
        return img.__class__(
            np.array(img.dataobj), np.array(img.affine, dtype=float), img.header
        )
    return resample_img(
        img,
        target_affine=grid.affine,
        target_shape=grid.shape,
        interpolation=interpolation,
        copy_header=True,
        force_resample=True,
    )


def to_canonical(img, interpolation: str = LABEL_INTERPOLATION):
    """Resample img onto the project's canonical grid."""
    return to_grid(img, Grid.canonical(), interpolation=interpolation)
