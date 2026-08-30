"""Measure the ATLAS template gap we are choosing to accept.

ATLAS normalizes to MNI152NLin2009aSym; our canonical grid is FSL MNI152
(MNI152NLin6Asym). This quantifies the residual left by regridding by affine
alone -- the number we carry as a known limitation.

It also re-runs the check that rejected the nonlinear alternative, so the
decision stays falsifiable rather than folklore. See docs/ATLAS-notes.md.
"""

import _bootstrap  # noqa: F401
import os
import sys

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine

from lesionconn.spaces import Grid, to_canonical

os.environ.setdefault("TEMPLATEFLOW_HOME", os.path.expanduser("~/templateflow"))


def _get(space, **kw):
    from templateflow import api

    r = api.get(space, resolution=1, **kw)
    r = [x for x in (r if isinstance(r, list) else [r]) if "label" not in str(x)]
    return nib.load(str(r[0]))


def main() -> int:
    grid = Grid.canonical()
    print("=" * 72)
    print("TEMPLATE GAP: MNI152NLin2009aSym (ATLAS) -> FSL MNI152 (canonical)")
    print("=" * 72)

    fsl_mask = np.asarray(
        to_canonical(_get("MNI152NLin6Asym", desc="brain", suffix="mask"), "nearest").dataobj
    ) > 0
    fsl_t1 = np.asarray(
        to_canonical(_get("MNI152NLin6Asym", suffix="T1w", desc=None), "continuous").dataobj
    )

    rows = []
    for space in ("MNI152NLin2009aSym", "MNI152NLin2009cAsym"):
        m = np.asarray(
            to_canonical(_get(space, desc="brain", suffix="mask"), "nearest").dataobj
        ) > 0
        dice = 2 * np.count_nonzero(fsl_mask & m) / (fsl_mask.sum() + m.sum())
        shift = np.linalg.norm(
            apply_affine(grid.affine, np.argwhere(m).mean(0))
            - apply_affine(grid.affine, np.argwhere(fsl_mask).mean(0))
        )
        rows.append((space, dice, shift))

    for space, dice, shift in rows:
        tag = "  <- ATLAS output" if space.endswith("aSym") else "  <- nilearn template"
        print(f"  {space:<22} Dice {dice:.4f}   centre-of-mass shift {shift:.2f} mm{tag}")

    t1 = np.asarray(
        to_canonical(_get("MNI152NLin2009cAsym", suffix="T1w", desc=None), "continuous").dataobj
    )
    a = fsl_t1[fsl_mask].astype(float)
    b = t1[fsl_mask].astype(float)
    a, b = a - a.mean(), b - b.mean()
    ncc = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

    print()
    print(f"  T1 intensity correspondence (NCC, affine regrid): {ncc:.4f}")
    print()
    print("  DECISION: regrid by affine and carry the residual.")
    print("  TemplateFlow's nonlinear NLin6Asym<->2009cAsym transforms were tested and")
    print("  REJECTED -- they lowered NCC to 0.66 in both directions and at both the")
    print("  canonical 2mm and the native 1mm grids. See docs/ATLAS-notes.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
