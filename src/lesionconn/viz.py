"""Rendering helpers -- thin wrappers over nilearn.plotting.

M5 proper comes much later (see ARCHITECTURE.md). These exist only so Phase 0
has an eyeball check to go with its assertions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: these are written to disk, never shown
import matplotlib.pyplot as plt
from nilearn import datasets, plotting



def plot_lesion_overlay(lesion_img, out_path: str | Path, title: str = "lesion on MNI152"):
    """Lesion on the MNI template, plus a glass-brain projection."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    template = datasets.load_mni152_template(resolution=2)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    plotting.plot_roi(
        lesion_img, bg_img=template, axes=axes[0], title=title,
        display_mode="ortho", cmap="autumn", alpha=0.8,
    )
    plotting.plot_glass_brain(
        lesion_img, axes=axes[1], title="glass brain projection",
        display_mode="lyrz", colorbar=False,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_parcellation(parc_img, out_path: str | Path, title: str = "parcellation"):
    """The merged parcellation over the template, to confirm coverage."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    template = datasets.load_mni152_template(resolution=2)

    display = plotting.plot_roi(
        parc_img, bg_img=template, title=title,
        display_mode="z", cut_coords=6, cmap="prism", alpha=0.7,
    )
    display.savefig(str(out_path), dpi=130)
    display.close()
    return out_path


def plot_lesion_on_parcellation(lesion_img, parc_img, out_path: str | Path):
    """Lesion and parcellation together -- the alignment eyeball check.

    The background is the anatomical template, never the parcellation: passing a
    label volume as `bg_img` renders label *indices* as grey levels, giving an
    arbitrary patchwork you cannot read anatomy from (docs/BUGS.md B6).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    template = datasets.load_mni152_template(resolution=2)
    cut_coords = plotting.find_xyz_cut_coords(lesion_img)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    display = plotting.plot_anat(
        template, axes=axes[0], display_mode="ortho", cut_coords=cut_coords,
        title="parcellation over MNI152 (cut at the lesion)",
    )
    display.add_overlay(parc_img, cmap="prism", alpha=0.65)

    display = plotting.plot_anat(
        template, axes=axes[1], display_mode="ortho", cut_coords=cut_coords,
        title="lesion (red) over parcellation -- check it lands in labelled tissue",
    )
    display.add_overlay(parc_img, cmap="prism", alpha=0.35)
    display.add_contours(lesion_img, levels=[0.5], colors="r", linewidths=1.5)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
