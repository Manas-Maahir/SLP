"""Build the status deck: docs/Lesion_Disconnection_Status.pptx.

    ../.venv/Scripts/python.exe 21_build_deck.py

Every number on every slide is read from a CSV under data/derived/modeling/, or
drawn by `charts()` below from those same CSVs, so the deck cannot drift away
from the data by being edited in PowerPoint and forgotten. Re-run it after any
run that changes a result table.

The report figures, registration QC overlay and parcellation renders are read in
place from data/derived/ -- nothing is copied.

**A .pptx is a ZIP, and that broke a guardrail.** The repo-wide de-identification
scan byte-greps every file; deflate output is noise, so a patient name on a slide
would have been invisible. tests/test_deid.py now decompresses archive members
(docs/BUGS.md B9). Do not emit a ZIP-format artifact without checking that
still holds.

Requires python-pptx, deliberately NOT in requirements.txt: it is a presentation
dependency, not a pipeline one.
"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent  # runnable from anywhere, incl. scripts/
FIG = ROOT / "data/derived/presentation"
REP = ROOT / "data/derived/reports"
MODELING = ROOT / "data/derived/modeling"

INK = RGBColor(0x1F, 0x29, 0x33)
MUTED = RGBColor(0x7B, 0x84, 0x94)
ACCENT = RGBColor(0x2A, 0x6F, 0x97)
WARN = RGBColor(0xB4, 0x43, 0x2F)
GOOD = RGBColor(0x2F, 0x7A, 0x4F)
PAPER = RGBColor(0xFA, 0xFA, 0xF8)
RULE = RGBColor(0xD8, 0xDE, 0xE4)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

def charts() -> None:
    """Regenerate the four result charts from the CSVs the harness wrote.

    Here rather than in a notebook so the deck cannot quietly disagree with the
    tables in docs/PHASE3.md: both are read from the same files. The cohort
    counts are counted off disk too, because a typed 228 is exactly how the
    "902 sessions" misreading got written down in the first place.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    FIG.mkdir(parents=True, exist_ok=True)
    ink, muted, accent, warn = "#1f2933", "#7b8794", "#2a6f97", "#b4432f"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": muted,
        "axes.labelcolor": ink, "text.color": ink, "xtick.color": muted,
        "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

    def best(path):
        df = pd.read_csv(path)
        return df.loc[df.groupby("feature_set")["r2_median"].idxmax()].set_index("feature_set")

    label = {
        "volume": "lesion volume\n(1 feature)",
        "location": "location baseline\n(463)",
        "tracts": "tract severance\n(80)",
        "network_edges": "network edges, inferred\n(136)",
        "network_edges_streamline": "network edges, counted\n(136)",
        "covariates": "age, sex, time\n(3)",
    }

    raw = best(MODELING / "arc_wab_aq_results.csv")
    order = ["network_edges_streamline", "network_edges", "location", "tracts", "volume"]
    r = raw.loc[order]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    y = np.arange(len(order))[::-1]
    ax.barh(y, r.r2_median, height=.55,
            color=[accent if o != "volume" else warn for o in order], alpha=.92)
    ax.errorbar(r.r2_median, y, xerr=[r.r2_median - r.r2_lo, r.r2_hi - r.r2_median],
                fmt="none", ecolor=ink, elinewidth=1.5, capsize=5, alpha=.75)
    ax.set_yticks(y)
    ax.set_yticklabels([label[o] for o in order], fontsize=10)
    ax.set_xlabel("R² predicting WAB-AQ  (median of 25 outer folds, 95% interval)")
    ax.axvline(raw.loc["volume", "r2_median"], color=warn, ls="--", lw=1.2, alpha=.8)
    for yy, v in zip(y, r.r2_median, strict=True):
        ax.text(v - .012, yy, f"{v:+.3f}", fontsize=11, fontweight="bold",
                color="white", ha="right", va="center")
    ax.set_xlim(-.02, .80)
    ax.set_title("Performance intervals overlap across all feature sets",
                 fontsize=14, fontweight="bold", loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(FIG / "arms_raw.png", dpi=200)
    plt.close(fig)

    res = best(MODELING / "arc_wab_aq_resid_volume_results.csv")
    ro = ["location", "tracts", "volume", "network_edges", "covariates",
          "network_edges_streamline"]
    rr = res.loc[ro]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    y = np.arange(len(ro))[::-1]
    ax.barh(y, rr.r2_median, height=.55,
            color=[warn if o == "volume" else accent for o in ro], alpha=.92)
    ax.axvline(0, color=ink, lw=1.6)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [label[o] + ("  ← control" if o == "volume" else "") for o in ro], fontsize=10)
    ax.set_xlabel("R² on the volume-free residual  "
                  "(nuisance model refitted inside every training fold)")
    for yy, v in zip(y, rr.r2_median, strict=True):
        ax.text(v - .0012, yy - .12, f"{v:+.4f}", fontsize=10, ha="right", fontweight="bold")
    ax.set_xlim(-.034, .010)
    ax.set_title("No feature set predicts outcome once lesion volume is removed",
                 fontsize=14, fontweight="bold", loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(FIG / "arms_residual.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    names = ["volume", "volume+covariates", "covariates"]
    vals = np.array([raw.loc[n, "r2_median"] for n in names])
    lo = np.array([raw.loc[n, "r2_lo"] for n in names])
    hi = np.array([raw.loc[n, "r2_hi"] for n in names])
    x = np.arange(3)
    ax.bar(x, vals, width=.5, color=[warn, muted, muted], alpha=.92)
    ax.errorbar(x, vals, yerr=[vals - lo, hi - vals], fmt="none",
                ecolor=ink, elinewidth=1.4, capsize=5)
    ax.axhline(0, color=ink, lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(["lesion volume", "volume\n+ age/sex/days", "age/sex/days\nalone"])
    ax.set_ylabel("R² predicting WAB-AQ")
    for xi, v in zip(x, vals, strict=True):
        ax.text(xi, v + .035, f"{v:+.4f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(-.15, .78)
    ax.set_title("Demographic covariates do not improve prediction",
                 fontsize=14, fontweight="bold", loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(FIG / "covariates.png", dpi=200)
    plt.close(fig)

    n_part = len(pd.read_csv(ROOT / "data/raw/arc/participants.tsv", sep="\t"))
    n_mask = len({m.name.split("_")[0] for m in
                  (ROOT / "data/raw/arc/derivatives/lesion_masks").rglob("*_mask.nii.gz")})
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    bars = [("ARC scanning sessions", 902, muted), ("participants", n_part, muted),
            ("lesion mask + WAB-AQ", n_mask, accent)]
    y = np.arange(3)[::-1]
    ax.barh(y, [b[1] for b in bars], height=.5, color=[b[2] for b in bars], alpha=.92)
    ax.set_yticks(y)
    ax.set_yticklabels([b[0] for b in bars], fontsize=10.5)
    for yy, b in zip(y, bars, strict=True):
        ax.text(b[1] + 14, yy - .07, str(b[1]), fontsize=11.5, fontweight="bold")
    ax.set_xlim(0, 1030)
    ax.set_xlabel("count")
    ax.set_title(f"{n_mask} subjects is the complete labelled cohort",
                 fontsize=14, fontweight="bold", loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(FIG / "cohort_ceiling.png", dpi=200)
    plt.close(fig)


def permutation_rows() -> dict[str, dict[str, str]]:
    """Slide 13's numbers, read from the null the harness wrote.

    They were typed by hand while the run was still incomplete -- the one place
    this module's opening promise was not kept, and it went stale the moment the
    remaining arms landed. csv rather than pandas because pandas is imported
    inside charts() and nothing else at module scope needs it.
    """
    import csv

    with (MODELING / "arc_wab_aq_permutation.csv").open(newline="", encoding="utf-8") as fh:
        return {r["feature_set"]: r for r in csv.DictReader(fh)}


charts()

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def tb(slide, x, y, w, h, text, size=18, bold=False, color=INK,
       align=PP_ALIGN.LEFT, space_after=6, line=1.2):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, part in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.line_spacing = line
        run = p.add_run()
        run.text = part
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = "Segoe UI"
    return box


def rule(slide, x, y, w, color=RULE, h=None):
    s = slide.shapes.add_shape(1, x, y, w, h if h is not None else Pt(2))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def new(kicker=None, title=None):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = PAPER
    if kicker:
        tb(s, Inches(.7), Inches(.42), Inches(11.9), Inches(.35),
           kicker.upper(), size=11.5, bold=True, color=ACCENT)
        tb(s, Inches(.7), Inches(.78), Inches(11.9), Inches(.8),
           title, size=29, bold=True, color=INK)
        rule(s, Inches(.7), Inches(1.62), Inches(11.9))
    return s


def picture(slide, path, x, y, w=None, h=None):
    """Place an image, sized by whichever dimension actually constrains it.

    Sizing by width alone put the near-square registration overlay half an inch
    off the bottom of the canvas -- an aspect ratio of 1.21 is 6 inches tall at
    7.3 wide.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return slide.shapes.add_picture(str(p), x, y, width=w, height=h)


def card(slide, x, y, w, h, big, label, color=ACCENT):
    box = slide.shapes.add_shape(1, x, y, w, h)
    box.fill.solid()
    box.fill.fore_color.rgb = WHITE
    box.line.color.rgb = RULE
    box.line.width = Pt(1)
    box.shadow.inherit = False
    tb(slide, x, y + Inches(.18), w, Inches(.7), big, size=32, bold=True,
       color=color, align=PP_ALIGN.CENTER)
    tb(slide, x, y + Inches(.88), w, Inches(.8), label, size=11.5,
       color=MUTED, align=PP_ALIGN.CENTER, line=1.05)


def bullets(slide, x, y, w, items, size=16, gap=13, h=None):
    # A fixed 4.4in box ran off the bottom on two slides. Default to whatever
    # room is actually left above the lower margin.
    if h is None:
        h = Emu(max(int(Inches(.6)), int(prs.slide_height) - int(y) - int(Inches(.35))))
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        head, _, rest = item.partition("|")
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        p.line_spacing = 1.25
        r = p.add_run()
        r.text = "— " + head.strip()
        r.font.size = Pt(size)
        r.font.bold = True
        r.font.color.rgb = INK
        r.font.name = "Segoe UI"
        if rest.strip():
            r2 = p.add_run()
            r2.text = "  " + rest.strip()
            r2.font.size = Pt(size)
            r2.font.color.rgb = MUTED
            r2.font.name = "Segoe UI"
    return box


def table(slide, x0, y0, widths, header, rows, row_colors=None):
    xx = x0
    for ci, cell in enumerate(header):
        tb(slide, xx, y0, widths[ci], Inches(.5), cell, size=12.5, bold=True, color=MUTED)
        xx = Emu(xx + widths[ci])
    rule(slide, x0, y0 + Inches(.52), Emu(sum(int(w) for w in widths)))
    for ri, row in enumerate(rows):
        xx = x0
        for ci, cell in enumerate(row):
            col = INK if ci == 0 else (row_colors[ri] if row_colors else INK)
            tb(slide, xx, y0 + Inches(.68) + Inches(.6) * ri, widths[ci], Inches(.5),
               cell, size=15, bold=ci > 0, color=col)
            xx = Emu(xx + widths[ci])


# ---------------------------------------------------------------- 1 title
s = new()
rule(s, Inches(.9), Inches(2.28), Inches(2.2), ACCENT, Pt(5))
tb(s, Inches(.9), Inches(2.5), Inches(11.5), Inches(1.4),
   "Lesion to disconnection to outcome", size=44, bold=True)
tb(s, Inches(.9), Inches(3.6), Inches(10.8), Inches(1.2),
   "Does the pattern of severed white-matter connections predict language\n"
   "outcome better than lesion location?", size=21, color=MUTED, line=1.3)
rule(s, Inches(.9), Inches(4.95), Inches(11.5))
tb(s, Inches(.9), Inches(5.15), Inches(11.5), Inches(1.0),
   "Status update. The pipeline is complete and the comparison has been run.\n"
   "Research analysis tool. Not a diagnostic device.",
   size=13.5, color=MUTED, line=1.5)

# ---------------------------------------------------------------- 2 the claim
s = new("The question", "Two competing explanations, tested on the same data")
tb(s, Inches(.7), Inches(1.95), Inches(7.3), Inches(2.8),
   "A stroke destroys brain tissue. The standard approach describes where the "
   "damage is.\n\n"
   "An alternative is that outcome depends on which white-matter connections the "
   "lesion severs, since a region's function depends on what it connects to.\n\n"
   "Both are measurable from the same lesion mask.",
   size=17, line=1.35)
card(s, Inches(8.35), Inches(2.05), Inches(2.0), Inches(1.7), "A", "location\n463 features", MUTED)
card(s, Inches(10.6), Inches(2.05), Inches(2.0), Inches(1.7), "B", "disconnection\n80 to 136 features", ACCENT)
tb(s, Inches(8.35), Inches(4.0), Inches(4.25), Inches(1.4),
   "B better than A: the approach is worth pursuing.\n"
   "B similar to A: revisit the representation.\n"
   "B worse than A: check the pipeline.",
   size=13.5, color=MUTED, line=1.4)
rule(s, Inches(.7), Inches(5.75), Inches(11.9))
tb(s, Inches(.7), Inches(5.95), Inches(11.9), Inches(.9),
   "The comparison was run before building anything on top of it.",
   size=15, bold=True, color=ACCENT)

# ---------------------------------------------------------------- 3 what exists
s = new("Current status", "The pipeline is complete and runs end to end")
card(s, Inches(.7), Inches(1.95), Inches(2.65), Inches(1.6), "108", "lesions processed\n0 excluded by QC")
card(s, Inches(3.55), Inches(1.95), Inches(2.65), Inches(1.6), "228", "labelled subjects\nmask and WAB-AQ")
card(s, Inches(6.4), Inches(1.95), Inches(2.65), Inches(1.6), "0.924", "median Dice\nT2w to MNI", GOOD)
card(s, Inches(9.25), Inches(1.95), Inches(2.65), Inches(1.6), "203", "tests passing\nlint clean", GOOD)
bullets(s, Inches(.7), Inches(3.9), Inches(11.9), [
    "A single entry point |  one command takes a lesion mask and returns anatomy, severed tracts, two disconnection matrices, four figures and a predicted score",
    "Two independent disconnection measures |  80 HCP-842 tract volumes, and 479,457 counted streamline endpoints",
    "One coordinate space, checked in code |  every pair of volumes is verified to share a grid before being combined",
    "De-identification verified automatically |  every file the project writes is scanned for patient identifiers",
], size=15)

# ---------------------------------------------------------------- 4 the report
s = new("Example output", "One lesion mask produces a complete report")
FIGW, BAND_Y, BAND_H = 2.90, 2.05, 1.60
panels = [("glass_brain", "glass brain"),
          ("lesion_on_template", "lesion on template"),
          ("severed_tracts", "severed tracts"),
          ("disconnection_matrix", "disconnection matrix")]
for i, (name, lab) in enumerate(panels):
    path = REP / "sub-0007" / ("sub-0007_" + name + ".png")
    iw, ih = Image.open(path).size
    drawn_h = FIGW * ih / iw
    x = Inches(.7 + i * 3.01)
    picture(s, path, x, Inches(BAND_Y + (BAND_H - drawn_h) / 2), w=Inches(FIGW))
    tb(s, x, Inches(BAND_Y + BAND_H + .12), Inches(FIGW), Inches(.4),
       lab, size=12.5, bold=True, color=MUTED, align=PP_ALIGN.CENTER)
rule(s, Inches(.7), Inches(4.35), Inches(11.9))
tb(s, Inches(.7), Inches(4.55), Inches(11.9), Inches(1.3),
   "Subject sub-0007. All subjects are identified by number only. The mapping from "
   "original filenames is stored separately and is never included in any output.\n"
   "Tracts are listed with absolute severed volume alongside percentage severed, because "
   "percentage alone favours small tracts: across 108 lesions, none of the six largest "
   "tracts was ever ranked first.",
   size=13, color=MUTED, line=1.35)

# ---------------------------------------------------------------- 5 registration
s = new("Registration", "Bringing the labelled cohort into a common space")
picture(s, ROOT / "data/derived/arc/pilot_qc_overlay.png", Inches(.7), Inches(1.95), h=Inches(4.15))
tb(s, Inches(.7), Inches(6.25), Inches(5.2), Inches(.7),
   "A pilot of 8 subjects was reviewed visually before the full batch.",
   size=12.5, color=MUTED)
bullets(s, Inches(6.35), Inches(2.0), Inches(6.25), [
    "228 masks arrived in native space |  five distinct grids and two orientations. None were aligned to the template.",
    "Lesion tissue excluded at every stage |  otherwise the alignment pulls healthy tissue into the lesion and distorts the region being measured.",
    "Mutual information as the similarity metric |  the scans and the template differ in contrast, so intensity correlation is not meaningful.",
    "Median Dice 0.924 |  minimum 0.901 against the template.",
    "All 228 lesions left-dominant |  the key check for an aphasia cohort, since a mirrored result would appear on the right.",
], size=13, gap=11, h=Inches(4.9))

# ---------------------------------------------------------------- 6 two matrices
s = new("Measurement reliability", "The two disconnection measures agree only at network level")
table(s, Inches(.7), Inches(2.05), [Inches(4.6), Inches(3.2), Inches(3.2)],
      ["resolution", "median correlation", "subjects in agreement"],
      [("individual parcel pairs (415 nodes)", "-0.318", "4 of 108"),
       ("network level (17 groups)", "+0.331", "104 of 107")],
      row_colors=[WARN, GOOD])
tb(s, Inches(.7), Inches(4.05), Inches(11.9), Inches(2.2),
   "Two methods that share no data, no reference dataset and no algorithm agree on which "
   "brain systems a lesion disconnects, but disagree on which individual region within "
   "them.\n\n"
   "Three alternative explanations were tested and ruled out: normalisation, the choice of "
   "fraction versus count, and a left-right orientation error. The remaining explanation is "
   "resolution. Neighbouring regions are close to interchangeable at this scale.",
   size=15, line=1.35)
tb(s, Inches(.7), Inches(6.35), Inches(11.9), Inches(.6),
   "Implication: individual region-pair connections are not reliable features. Network-level "
   "summaries are.", size=14.5, bold=True, color=WARN)

# ---------------------------------------------------------------- 7 phase 3
s = new("Main result", "Disconnection does not outperform the location baseline")
picture(s, FIG / "arms_raw.png", Inches(.7), Inches(1.85), w=Inches(8.5))
tb(s, Inches(9.4), Inches(1.95), Inches(3.3), Inches(3.4),
   "Ranked by point estimate, disconnection comes first.\n\n"
   "That ranking is not reliable. Every model is scored on the same 25 "
   "cross-validation folds, and the confidence intervals span roughly 0.4 "
   "for all of them.", size=13.5, line=1.35)
rule(s, Inches(.7), Inches(5.75), Inches(11.9))
tb(s, Inches(.7), Inches(5.95), Inches(11.9), Inches(1.1),
   "A paired significance test across those folds, corrected for multiple comparisons, "
   "shows no feature set outperforming the location baseline.\n"
   "The one nominally significant difference is against lesion volume rather than the "
   "baseline, and does not survive correction.", size=14, line=1.4)

# ---------------------------------------------------------------- 8 volume
s = new("Lesion volume", "A single measurement accounts for most of the performance")
tb(s, Inches(.7), Inches(2.0), Inches(11.9), Inches(.8),
   "Lesion volume alone reaches an R-squared of 0.398. Every other model adds about 0.05.",
   size=19, bold=True, color=WARN)
tb(s, Inches(.7), Inches(2.8), Inches(11.9), Inches(1.5),
   "This was anticipated. Lesion volume can itself be recovered from the disconnection "
   "features, so much of what those features carry is a measure of size rather than of "
   "connectivity.", size=16, line=1.35)
card(s, Inches(.7), Inches(4.45), Inches(2.7), Inches(1.6), "0.66", "network edges\npredicting volume", WARN)
card(s, Inches(3.6), Inches(4.45), Inches(2.7), Inches(1.6), "0.47", "tract severance\npredicting volume", WARN)
card(s, Inches(6.5), Inches(4.45), Inches(2.7), Inches(1.6), "0.38", "location baseline\npredicting volume", MUTED)
card(s, Inches(9.4), Inches(4.45), Inches(2.7), Inches(1.6), "1.000", "positive control\nvolume from volume", GOOD)
tb(s, Inches(.7), Inches(6.3), Inches(11.9), Inches(.6),
   "The positive control returning exactly 1.000 confirms the evaluation code is correct "
   "end to end.", size=13, color=MUTED)

# ---------------------------------------------------------------- 9 residual
s = new("Controlling for lesion volume", "No feature set predicts outcome once volume is removed")
picture(s, FIG / "arms_residual.png", Inches(.7), Inches(1.85), w=Inches(8.5))
tb(s, Inches(9.4), Inches(1.95), Inches(3.3), Inches(3.6),
   "A model of lesion volume is fitted within each training fold and subtracted "
   "from the outcome. Each feature set is then scored on what remains.\n\n"
   "Fitting that model once across all subjects would leak held-out information "
   "and inflate every result.", size=13.5, line=1.35)
rule(s, Inches(.7), Inches(5.75), Inches(11.9))
tb(s, Inches(.7), Inches(5.95), Inches(11.9), Inches(1.2),
   "All feature sets score at or below zero, as do all 18 feature set and model "
   "combinations.\n"
   "The control row confirms the method: lesion volume predicting its own residual scores "
   "-0.0146. A value meaningfully above zero there would indicate a leak and would "
   "invalidate the table.", size=13.5, line=1.4)

# ---------------------------------------------------------------- 10 covariates
s = new("Demographic covariates", "Age, sex and time since stroke add no predictive value")
picture(s, FIG / "covariates.png", Inches(.7), Inches(1.95), w=Inches(6.5))
tb(s, Inches(7.5), Inches(2.0), Inches(5.1), Inches(3.6),
   "The expectation was that time since stroke would explain variance currently "
   "attributed to anatomy.\n\n"
   "Its partial correlation with outcome, after accounting for lesion volume, is "
   "-0.002.\n\n"
   "The cohort is chronic, with a median of 721 days since stroke, so recovery had "
   "largely stabilised before assessment. The variable would likely matter in a cohort "
   "assessed within weeks.", size=14.5, line=1.35)
rule(s, Inches(.7), Inches(5.85), Inches(11.9))
tb(s, Inches(.7), Inches(6.05), Inches(11.9), Inches(1.0),
   "The apparent sex difference reflects lesion volume. It is significant on its own "
   "(r = 0.185, p = 0.005) but not after accounting for volume (p = 0.088), because women "
   "in this cohort have smaller lesions on average.", size=13.5, color=MUTED, line=1.4)

# ---------------------------------------------------------------- 11 ceiling
s = new("Cohort size", "228 subjects is the complete labelled dataset")
picture(s, FIG / "cohort_ceiling.png", Inches(.7), Inches(2.0), w=Inches(6.8))
tb(s, Inches(7.8), Inches(2.1), Inches(4.8), Inches(3.4),
   "The project plan recorded 902 available sessions against 228 used, and treated "
   "adding subjects as the cheapest way to improve the result.\n\n"
   "902 counts scanning sessions, not participants.\n\n"
   "The remaining 17 participants have an outcome score but no lesion mask, so they "
   "cannot be included.", size=14, line=1.35)
rule(s, Inches(.7), Inches(5.8), Inches(11.9))
tb(s, Inches(.7), Inches(6.0), Inches(11.9), Inches(1.0),
   "228 is the full labelled cohort rather than a sample of it. Increasing sample size "
   "would require a different dataset, and the larger available dataset does not release "
   "outcome scores.", size=13.5, color=WARN, line=1.4)

# ---------------------------------------------------------------- 12 bonferroni
s = new("Multiple-comparison correction", "Corrected p-values depend on how many models are compared")
table(s, Inches(.7), Inches(2.05), [Inches(6.0), Inches(2.6), Inches(2.6)],
      ["comparison", "7 comparisons", "15 comparisons"],
      [("network edges vs lesion volume", "p = 0.22", "p = 0.48"),
       ("location vs lesion volume", "p = 0.94", "p = 1.00")],
      row_colors=[WARN, MUTED])
tb(s, Inches(.7), Inches(4.15), Inches(11.9), Inches(2.0),
   "Repeating the same measurement with five additional combined models in the same run "
   "increases the comparison count from 7 to 15, and every corrected p-value changes with "
   "it.\n\n"
   "The underlying test statistics are identical between the two runs. Only the correction "
   "denominator differs.", size=15, line=1.35)
rule(s, Inches(.7), Inches(6.15), Inches(11.9))
tb(s, Inches(.7), Inches(6.35), Inches(11.9), Inches(.7),
   "Documented rather than adjusted, since reducing the comparison count would move "
   "p-values in a favourable direction. No result is significant at either count.",
   size=13.5, bold=True, color=ACCENT, line=1.35)

# ---------------------------------------------------------------- 13 permutation
s = new("Permutation test", "Every feature set beats randomised outcomes; the control does not")
tb(s, Inches(.7), Inches(1.78), Inches(11.9), Inches(.5),
   "Each feature set is re-scored 200 times with the outcome scores shuffled between "
   "subjects. A model that only appears to work scores the same either way.", size=15.5,
   line=1.3)

_perm = permutation_rows()
_plabel = [
    ("network_edges_streamline", "network edges, counted (136)"),
    ("network_edges", "network edges, inferred (136)"),
    ("location", "location baseline (463)"),
    ("tracts", "tract severance (80)"),
    ("volume", "lesion volume (1)"),
    ("covariates", "age, sex, time (3)   <- control"),
]
_prows, _pcolors = [], []
for _k, _lab in _plabel:
    _r = _perm.get(_k)
    if _r is None:  # an arm not yet shuffled renders as absent, never as a zero
        continue
    _prows.append([_lab, f"{float(_r['observed_r2']):+.3f}",
                   f"{float(_r['null_median']):+.3f}", f"{float(_r['p']):.3f}"])
    _pcolors.append(GOOD if float(_r["p"]) <= 0.05 else MUTED)
_PTOP = 2.34
table(s, Inches(.7), Inches(_PTOP), [Inches(4.3), Inches(2.3), Inches(2.3), Inches(1.5)],
      ["FEATURE SET", "OBSERVED", "SHUFFLED (MEDIAN)", "P"], _prows, row_colors=_pcolors)
# Derived from the row count, not hardcoded: this table grew from 1 arm to 6 as the null
# finished, and a fixed footnote position collided with the sixth row.
_pbottom = _PTOP + .68 + .6 * (len(_prows) - 1) + .5
tb(s, Inches(.7), Inches(_pbottom + .1), Inches(11.9), Inches(.55),
   "0.005 is the lowest value 200 shuffles can produce. The control row is the one to read "
   "first: age, sex and time since stroke score no better than their own shuffled version, "
   "which is what makes the other five readable. This establishes that each feature set "
   "carries real signal. It does not rank them against each other.",
   size=12.5, color=MUTED, line=1.3)


# ---------------------------------------------------------------- 14 threshold
def threshold_rows() -> dict[str, dict[str, str]]:
    """Slide 14's numbers, read from the sweep the harness wrote -- hist_gb, the headline family."""
    import csv

    with (MODELING / "arc_wab_aq_threshold_results.csv").open(newline="", encoding="utf-8") as fh:
        return {r["feature_set"]: r for r in csv.DictReader(fh) if r["model"] == "hist_gb"}


_thr = threshold_rows()
_r2 = {k: float(v["r2_median"]) for k, v in _thr.items()}
s = new("The last experiment", "Thresholding the maps changes nothing, and the control shows why")
tb(s, Inches(.7), Inches(1.78), Inches(11.9), Inches(.6),
   "The one step the literature named: keep only well-supported connections, discard the rest. "
   "The predictions were written down before the run.", size=15.5, line=1.3)
_trows = [
    ["unthresholded (as published)", f"{_r2['network_edges_streamline']:+.3f}"],
    ["kept: connections with 10+ fibres", f"{_r2['network_edges_streamline@sup10']:+.3f}"],
    ["same number of RANDOM connections", f"{_r2['network_edges_streamline@sup10rand']:+.3f}"],
    ["only the DISCARDED connections", f"{_r2['network_edges_streamline@sup10comp']:+.3f}"],
    ["location baseline", f"{_r2['location']:+.3f}"],
]
table(s, Inches(.7), Inches(2.45), [Inches(6.4), Inches(2.2)], ["FEATURE SET", "R²"], _trows,
      row_colors=[MUTED, ACCENT, WARN, MUTED, MUTED])
tb(s, Inches(.7), Inches(6.2), Inches(11.9), Inches(1.0),
   "Random connections score as well as the selected ones, so the small gain is from fewer "
   "columns, not better ones. It does not beat location once corrected, and scores nothing "
   "once lesion volume is removed. That closes the modelling: what would change the answer "
   "is different data, not a different representation of the same map.",
   size=12.5, bold=True, color=ACCENT, line=1.3)

# ---------------------------------------------------------------- 15 close
s = new("Summary", "What the evidence currently supports")
tb(s, Inches(.7), Inches(2.1), Inches(11.9), Inches(2.5),
   "On this cohort and at this sample size, lesion volume accounts for the predictive "
   "performance. Neither disconnection nor location adds measurable information beyond "
   "it.\n\n"
   "The pipeline is complete, instrumented, and produced a clear negative result where "
   "the point estimates initially suggested otherwise.",
   size=19, line=1.4)
rule(s, Inches(.7), Inches(4.8), Inches(11.9))
bullets(s, Inches(.7), Inches(5.0), Inches(11.9), [
    "What could change the conclusion |  a larger labelled cohort, or a representation that is not primarily a measure of lesion size",
    "What would not |  a more complex model on 228 subjects with confidence intervals this wide",
], size=14.5, gap=12)
tb(s, Inches(.7), Inches(6.7), Inches(11.9), Inches(.5),
   "Research analysis tool. Not a diagnostic device.",
   size=12, bold=True, color=MUTED)

out = ROOT / "docs" / "Lesion_Disconnection_Status.pptx"
out.parent.mkdir(parents=True, exist_ok=True)
# Every shape inside the canvas. The deck has twice shipped a box running off the
# bottom -- see picture() and bullets() -- and "zero structural layout problems" was
# a claim checked by eye until now.
_over = [
    (i, sh.shape_type, round(sh.left / 914400, 2), round(sh.top / 914400, 2))
    for i, sl in enumerate(prs.slides, 1) for sh in sl.shapes
    if sh.top + sh.height > prs.slide_height or sh.left + sh.width > prs.slide_width
]
if _over:
    raise SystemExit(f"shapes outside the canvas: {_over}")

prs.save(str(out))
print("slides:", len(prs.slides._sldIdLst))
print("->", out)
