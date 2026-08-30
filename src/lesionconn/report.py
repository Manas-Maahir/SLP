"""M5 -- the atlas-referenced report, and the figures that go with it.

The presentation layer, and deliberately last (ARCHITECTURE.md M5). Everything
here reads M3's output; nothing here computes disconnection.

Two rules this module exists to enforce:

1. **The framing travels with the output.** `config.DISCLAIMER` is written into
   the report text, into `severed_tracts.json`, and onto every figure. A report
   that is read away from this repo must still say what it is.
2. **Severity is shown two ways.** Ranking tracts by `frac_severed` alone is
   misleading: it is normalised by tract size, so small bundles dominate. Over
   the 108-subject cohort, none of the six largest tracts was ever the top hit,
   though the corpus callosum is involved in nearly every large lesion
   (docs/DISCONNECTION.md limit 7). So every tract row carries the absolute
   severed volume beside the percentage.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import datasets, plotting

from . import config, deid, disconnectome as dc
from .spaces import Grid

WIDTH = 78
RULE = "=" * WIDTH
THIN = "-" * WIDTH

TOP_TRACTS = 12
TOP_PARCELS = 12
TOP_PAIRS = 8


@dataclass
class Report:
    subject_id: str
    text: str
    figures: list[Path] = field(default_factory=list)
    out_dir: Path | None = None


def _wrap(body: str, indent: str = "  ") -> str:
    return "\n".join(
        textwrap.fill(line, WIDTH, initial_indent=indent, subsequent_indent=indent)
        for line in body.strip().split("\n")
    )


def _short_name(name: str, limit: int = 34) -> str:
    """A label that stays distinct when truncated.

    Every Schaefer parcel starts '7Networks_', which carries no information and
    ate ten of the characters that distinguish one parcel from another -- four
    axis labels on the first matrix figure truncated to the identical string.
    The tail is what identifies a parcel, so the tail is what is kept.
    """
    name = str(name)
    for prefix in ("7Networks_",):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name if len(name) <= limit else "..." + name[-(limit - 3):]


def _hemisphere_split(lesion_img) -> tuple[float, float]:
    """Fraction of the lesion in each hemisphere, from world coordinates."""
    grid = Grid.of(lesion_img)
    mask = np.asarray(lesion_img.dataobj) > 0
    if not mask.any():
        return 0.0, 0.0
    xw = dc._world_x_field(grid)[mask]
    return float((xw < 0).mean()), float((xw > 0).mean())


def render_text(
    subject_id: str,
    lesion_img,
    severance: dc.TractSeverance,
    parcel_overlap: pd.DataFrame,
    location_overlap: pd.DataFrame | None = None,
    matrix: np.ndarray | None = None,
    parc_labels: pd.DataFrame | None = None,
    native_volume_mm3: float | None = None,
    native_grid: str | None = None,
    qc_status: str | None = None,
    figures: list[Path] | None = None,
) -> str:
    """The atlas-referenced text report."""
    grid = Grid.canonical()
    mask = np.asarray(lesion_img.dataobj) > 0
    canonical_volume = float(mask.sum() * grid.voxel_volume_mm3)
    com = dc.centre_of_mass_world(mask, grid)
    left, right = _hemisphere_split(lesion_img)

    out: list[str] = [RULE, f" LESION DISCONNECTION REPORT  --  {subject_id}", RULE, ""]
    out += [_wrap(config.DISCLAIMER, indent=" "), ""]

    # --- the lesion itself ---
    out += [THIN, " LESION", THIN]
    if native_volume_mm3 is not None:
        delta = (
            100.0 * (canonical_volume - native_volume_mm3) / native_volume_mm3
            if native_volume_mm3
            else 0.0
        )
        out.append(
            f"  volume (as acquired)   : {native_volume_mm3:>12,.0f} mm3   <- use this one"
        )
        if native_grid:
            out.append(f"    measured on          : {native_grid}")
        out.append(f"  volume (2mm canonical) : {canonical_volume:>12,.0f} mm3   ({delta:+.1f}%)")
        out.append(
            "    the canonical grid is the intersection substrate only; resampling a 1mm"
        )
        out.append("    mask to 2mm moves small lesions by up to 26% (docs/BUGS.md B2)")
    else:
        out.append(f"  volume (2mm canonical) : {canonical_volume:>12,.0f} mm3")
    out.append(f"  centre of mass         : MNI ({com[0]:+.1f}, {com[1]:+.1f}, {com[2]:+.1f})")
    side = "left" if left > right else "right" if right > left else "midline"
    out.append(f"  hemisphere             : {side}  ({100 * max(left, right):.0f}% of voxels)")
    if qc_status:
        out.append(f"  QC                     : {qc_status}")
    out.append("")

    # --- where the lesion is, from the coverage-complete atlas ---
    #
    # This section MUST come from the N=463 location atlas, not the N=415 node
    # set. The first draft used the node set and produced a report that led with
    # left-hemisphere occipital parcels for a lesion that is 70% RIGHT CEREBELLUM:
    # the node set has no cerebellar labels, so it saw only the 17.8% of that
    # lesion lying outside the cerebellum, and the reader saw a confident table
    # with no hint that most of the lesion was missing from it.
    if location_overlap is not None and not location_overlap.empty:
        covered = float(location_overlap["frac_of_lesion"].sum())
        out += [THIN, " ANATOMY   (location atlas, N=463 -- full brain coverage)", THIN]
        out.append(f"  {'region':<46}{'% of region':>12}{'% of lesion':>12}{'mm3':>9}")
        for row in location_overlap.head(TOP_PARCELS).itertuples():
            out.append(
                f"  {str(row.name)[:45]:<46}{100 * row.frac_of_parcel:>11.1f}%"
                f"{100 * row.frac_of_lesion:>11.1f}%{row.volume_mm3:>9,.0f}"
            )
        if len(location_overlap) > TOP_PARCELS:
            out.append(f"  ... and {len(location_overlap) - TOP_PARCELS} more regions")
        out.append(f"  {'':<46}{'':>12}{100 * covered:>11.1f}% of lesion described")
        out.append("")

    # --- how much of the lesion the disconnection node set can even see ---
    out += [THIN, " DISCONNECTION NODE SET   (Schaefer-400 + Harvard-Oxford, N=415)", THIN]
    node_covered = (
        0.0 if parcel_overlap.empty else float(parcel_overlap["frac_of_lesion"].sum())
    )
    out.append(f"  node-set coverage of this lesion : {100 * node_covered:.1f}% of its volume")
    if node_covered < 0.5:
        out.append("")
        out.append(_wrap(
            "MOST OF THIS LESION IS INVISIBLE TO THE NODE SET. It is grey matter and "
            "deep nuclei only -- no cerebellum, no deep white matter (docs/BUGS.md B1, "
            "docs/DISCONNECTION.md limit 6). The parcel-pair matrix below therefore "
            "describes only the labelled fraction, and the regions listed here are a "
            "minority of the damage. Read the ANATOMY section above for where the "
            "lesion actually is."
        ))
    out.append("")
    if parcel_overlap.empty:
        out.append("  no node-set parcel is involved at all.")
    else:
        out.append(f"  {'parcel':<46}{'% of parcel':>12}{'% of lesion':>12}{'mm3':>9}")
        for row in parcel_overlap.head(TOP_PARCELS).itertuples():
            out.append(
                f"  {str(row.name)[:45]:<46}{100 * row.frac_of_parcel:>11.1f}%"
                f"{100 * row.frac_of_lesion:>11.1f}%{row.volume_mm3:>9,.0f}"
            )
        if len(parcel_overlap) > TOP_PARCELS:
            out.append(f"  ... and {len(parcel_overlap) - TOP_PARCELS} more parcels")
    out.append("")

    # --- tracts ---
    hit = severance.hit
    out += [THIN, f" WHITE-MATTER TRACTS SEVERED   (HCP-842, {len(severance.table)} tracts)", THIN]
    if hit.empty:
        out.append("  none.")
    else:
        out.append(f"  {'tract':<46}{'% severed':>11}{'severed mm3':>13}")
        for row in hit.head(TOP_TRACTS).itertuples():
            flag = "  *" if row.thin else ""
            out.append(
                f"  {str(row.name)[:45]:<46}{100 * row.frac_severed:>10.1f}%"
                f"{row.volume_hit_mm3:>13,.0f}{flag}"
            )
        if len(hit) > TOP_TRACTS:
            out.append(f"  ... and {len(hit) - TOP_TRACTS} more tracts")
        if hit.head(TOP_TRACTS)["thin"].any():
            out.append("  * thin at 2mm -- under-sampled, treat its percentage as coarse")
        out.append("")
        out.append(_wrap(
            "Read the two columns together. '% severed' is normalised by tract size, so "
            "a small bundle scores high on a modest lesion while the corpus callosum "
            "rarely exceeds a few percent however much of it is destroyed. Neither "
            "column alone ranks clinical importance."
        ))
    out.append("")

    # --- parcel pairs ---
    if matrix is not None and parc_labels is not None:
        out += [THIN, " PARCEL-PAIR DISCONNECTION   (derived -- see docs/DISCONNECTION.md)", THIN]
        pairs = dc.matrix_summary(matrix, parc_labels, top_n=TOP_PAIRS)
        if not pairs:
            out.append("  none.")
        else:
            for entry in pairs:
                out.append(
                    f"  {entry['value']:>7.4f}   {_short_name(entry['parcel_a'], 31):<32} <-> "
                    f"{_short_name(entry['parcel_b'], 31)}"
                )
            out.append("")
            out.append(_wrap(
                "These values are WEIGHTED EVIDENCE of disruption, not severed-streamline "
                "fractions. HCP-842 ships tract volumes, not streamlines, so which parcels "
                "a tract joins is inferred from which parcels it abuts. Rank by value; "
                "never count how many pairs are non-zero."
            ))
        out.append("")

    # --- figures ---
    if figures:
        out += [THIN, " FIGURES", THIN]
        for path in figures:
            out.append(f"  {path.name}")
        out.append("")

    # --- method ---
    out += [THIN, " METHOD AND LIMITS", THIN]
    out.append(_wrap(
        "Disconnection is inferred by intersecting this lesion with the HCP-842 "
        "population-averaged tractography atlas (Yeh et al. 2018), resampled onto the "
        "FSL MNI152 2mm grid. No diffusion imaging of this patient was used, and nothing "
        "here reflects this individual's own wiring, pre-existing anomalies, or "
        "post-stroke reorganisation.\n"
        "'% severed' counts atlas tract VOXELS overlapped, not streamlines cut. A small "
        "lesion at a bundle's narrow waist can disconnect it almost entirely while "
        "scoring low here, so every figure is a lower bound on functional disconnection.\n"
        "The parcel-pair matrix is a derived approximation whose construction and eight "
        "named limits are set out in docs/DISCONNECTION.md. It has not been validated "
        "against a streamline-derived connectome."
    ))
    out += ["", RULE, _wrap(config.DISCLAIMER, indent=" "), RULE, ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _footer(fig, y: float = 0.005) -> None:
    fig.text(
        0.5, y,
        "Research tool, not a diagnostic device -- normative atlas, not this patient's wiring",
        ha="center", va="bottom", fontsize=7, color="0.35",
    )


def figure_glass_brain(lesion_img, out_path: Path, subject_id: str) -> Path:
    fig = plt.figure(figsize=(11, 3.4))
    display = plotting.plot_glass_brain(
        lesion_img, figure=fig, display_mode="lyrz", colorbar=False,
        title=f"{subject_id} -- lesion, glass brain",
    )
    _footer(fig, y=-0.06)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    display.close()
    plt.close(fig)
    return out_path


def figure_lesion_on_template(lesion_img, out_path: Path, subject_id: str) -> Path:
    template = datasets.load_mni152_template(resolution=2)
    cut = plotting.find_xyz_cut_coords(lesion_img)
    fig = plt.figure(figsize=(11, 3.8))
    display = plotting.plot_roi(
        lesion_img, bg_img=template, figure=fig, display_mode="ortho",
        cut_coords=cut, cmap="autumn", alpha=0.75,
        title=f"{subject_id} -- lesion on MNI152 (cut at the lesion)",
    )
    _footer(fig, y=-0.06)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    display.close()
    plt.close(fig)
    return out_path


def figure_severed_tracts(severance: dc.TractSeverance, out_path: Path, subject_id: str,
                          top_n: int = 15) -> Path:
    """Percentage severed and absolute volume, side by side and deliberately so."""
    hit = severance.hit.head(top_n).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.2, 0.34 * len(hit) + 1.6)), sharey=True)

    if hit.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "no tract involved", ha="center", va="center")
            ax.axis("off")
    else:
        colours = {"L": "#c0392b", "R": "#2471a3", "M": "#7d3c98"}
        bar_colours = [colours.get(h, "0.4") for h in hit["hemisphere"]]
        labels = [_short_name(n, 40).replace("_", " ") for n in hit["name"]]

        axes[0].barh(labels, 100 * hit["frac_severed"], color=bar_colours)
        axes[0].set_xlabel("% of tract severed  (normalised -- favours small tracts)")
        axes[0].grid(axis="x", alpha=0.25)

        axes[1].barh(labels, hit["volume_hit_mm3"], color=bar_colours)
        axes[1].set_xlabel("severed volume (mm3)  (absolute)")
        axes[1].grid(axis="x", alpha=0.25)

        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()]
        axes[1].legend(handles, ["left", "right", "midline"], fontsize=8, loc="lower right")

    fig.suptitle(
        f"{subject_id} -- severed tracts: the same tracts ranked two ways", fontsize=11
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    _footer(fig)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def figure_matrix(matrix: np.ndarray, parc_labels: pd.DataFrame, out_path: Path,
                  subject_id: str, top_n: int = 25) -> Path:
    """Whole matrix beside a labelled submatrix of the parcels that matter.

    The full 415x415 view alone is unreadable and, because the corpus callosum
    fills most of it at very low values, actively misleading about where the
    signal is. The submatrix is the honest read.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))

    vmax = float(matrix.max()) or 1.0
    im = axes[0].imshow(matrix, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
    axes[0].set_title(f"all {matrix.shape[0]} parcels", fontsize=10)
    axes[0].set_xlabel("parcel index")
    axes[0].set_ylabel("parcel index")
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="disconnection (weighted evidence)")

    strength = matrix.max(axis=1)
    keep = np.argsort(strength)[::-1][:top_n]
    keep = keep[strength[keep] > 0]
    if len(keep) >= 2:
        keep = np.sort(keep)
        sub = matrix[np.ix_(keep, keep)]
        name_of = dict(zip(parc_labels["index"], parc_labels["name"], strict=True))
        names = [_short_name(name_of.get(int(i) + 1, i + 1)) for i in keep]
        im2 = axes[1].imshow(sub, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        axes[1].set_xticks(range(len(keep)), names, rotation=90, fontsize=6)
        axes[1].set_yticks(range(len(keep)), names, fontsize=6)
        axes[1].set_title(f"the {len(keep)} most disconnected parcels", fontsize=10)
        fig.colorbar(im2, ax=axes[1], fraction=0.046)
    else:
        axes[1].text(0.5, 0.5, "no parcel pair disconnected", ha="center", va="center")
        axes[1].axis("off")

    fig.suptitle(
        f"{subject_id} -- parcel-pair disconnection (derived; see docs/DISCONNECTION.md)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _footer(fig)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_figures(
    subject_id: str,
    lesion_img,
    severance: dc.TractSeverance,
    matrix: np.ndarray | None,
    parc_labels: pd.DataFrame | None,
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        figure_glass_brain(lesion_img, out_dir / f"{subject_id}_glass_brain.png", subject_id),
        figure_lesion_on_template(
            lesion_img, out_dir / f"{subject_id}_lesion_on_template.png", subject_id
        ),
        figure_severed_tracts(
            severance, out_dir / f"{subject_id}_severed_tracts.png", subject_id
        ),
    ]
    if matrix is not None and parc_labels is not None:
        figures.append(
            figure_matrix(
                matrix, parc_labels, out_dir / f"{subject_id}_disconnection_matrix.png",
                subject_id,
            )
        )
    return figures


def write_report(report: Report, path: Path) -> Path:
    """Write report.txt, refusing to emit anything carrying a patient name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    deid.guard_output(report.text, f"report for {report.subject_id}")
    path.write_text(report.text, encoding="utf-8")
    return path
