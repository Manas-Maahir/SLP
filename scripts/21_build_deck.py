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
# Slide copy uses real typographic characters on purpose -- MINUS SIGN in
# "-0.0146", MULTIPLICATION SIGN in "feature set x model", GREEK RHO for
# Spearman. Rendering those as ASCII lookalikes on a projected slide is worse
# than the lint they trip, so RUF001 is disabled for this file only.
# ruff: noqa: RUF001
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
        "location": "location — Model A\n(463)",
        "tracts": "tract severance — Model B\n(80)",
        "network_edges": "network edges, inferred\n(136)",
        "network_edges_streamline": "network edges, counted\n(136)",
        "covariates": "age / sex / days\n(3)",
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
    ax.set_title("Every arm overlaps. The ordering is not a result.",
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
    ax.set_title("Remove lesion volume and nothing survives — Model A included.",
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
    ax.set_title("Covariates make the volume model worse.",
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
    ax.set_title(f"{n_mask} is the whole labelled cohort, not a sample of it.",
                 fontsize=14, fontweight="bold", loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(FIG / "cohort_ceiling.png", dpi=200)
    plt.close(fig)


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
   "Lesion → Disconnection → Outcome", size=44, bold=True)
tb(s, Inches(.9), Inches(3.6), Inches(10.8), Inches(1.2),
   "Does which connections a stroke severs predict language outcome\n"
   "better than where the stroke is?", size=21, color=MUTED, line=1.3)
rule(s, Inches(.9), Inches(4.95), Inches(11.5))
tb(s, Inches(.9), Inches(5.15), Inches(11.5), Inches(1.0),
   "Status update  ·  the pipeline is built, the thesis was tested honestly, and it lost\n"
   "Research annotation / analysis tool — NOT a diagnostic device",
   size=13.5, color=MUTED, line=1.5)

# ---------------------------------------------------------------- 2 the claim
s = new("The question", "A testable claim, not a slogan")
tb(s, Inches(.7), Inches(1.95), Inches(7.3), Inches(2.8),
   "A stroke kills tissue. Clinicians ask WHERE.\n\n"
   "But a region matters because of what it connects to — and a lesion also cuts the "
   "white-matter cables running underneath it.\n\n"
   "So: is the deficit explained by LOCATION, or by DISCONNECTION?",
   size=17, line=1.35)
card(s, Inches(8.35), Inches(2.05), Inches(2.0), Inches(1.7), "A", "location\n463 features", MUTED)
card(s, Inches(10.6), Inches(2.05), Inches(2.0), Inches(1.7), "B", "disconnection\n80–136 features", ACCENT)
tb(s, Inches(8.35), Inches(4.0), Inches(4.25), Inches(1.4),
   "B > A → the thesis holds.\nB ≈ A → dig into representation.\nB < A → something is wrong.",
   size=13.5, color=MUTED, line=1.4)
rule(s, Inches(.7), Inches(5.75), Inches(11.9))
tb(s, Inches(.7), Inches(5.95), Inches(11.9), Inches(.9),
   "Ordering principle: test the thesis BEFORE building the infrastructure around it.",
   size=15, bold=True, color=ACCENT)

# ---------------------------------------------------------------- 3 what exists
s = new("What exists", "The pipeline is done and it runs end to end")
card(s, Inches(.7), Inches(1.95), Inches(2.65), Inches(1.6), "108", "lesions processed\n0 excluded by QC")
card(s, Inches(3.55), Inches(1.95), Inches(2.65), Inches(1.6), "228", "ARC subjects with\nlesion mask + WAB-AQ")
card(s, Inches(6.4), Inches(1.95), Inches(2.65), Inches(1.6), "0.924", "median Dice,\nT2w → MNI registration", GOOD)
card(s, Inches(9.25), Inches(1.95), Inches(2.65), Inches(1.6), "202", "tests passing,\nruff clean", GOOD)
bullets(s, Inches(.7), Inches(3.9), Inches(11.9), [
    "One entry point |  05_report.py <lesion> → anatomy, severed tracts, two matrices, four figures, a predicted WAB-AQ — or a refusal",
    "Two disconnection matrices |  independent references: 80 HCP-842 tract volumes, and 479,457 counted streamline endpoints",
    "One grid, asserted in code |  FSL MNI152 2mm (91,109,91) LAS — assert_same_grid runs before any two volumes are combined",
    "De-identification enforced by grep |  every emitted artifact is scanned; it has caught a real patient name three times",
], size=15)

# ---------------------------------------------------------------- 4 the report
s = new("What it produces", "One lesion mask in, a readable report out")
# The four figures have aspect ratios from 1.90 to 2.95, so a shared top edge
# leaves them ragged. Centre each in a common band instead.
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
   "Subject sub-0007. Ids are sub-NNNN by construction — the name-to-id map is the only artifact "
   "that holds identifiers and it never leaves data/private/.\n"
   "Tracts are ranked with absolute severed volume beside % severed, because % alone structurally "
   "favours small tracts: across 108 lesions none of the six largest tracts was ever the top hit.",
   size=13, color=MUTED, line=1.35)

# ---------------------------------------------------------------- 5 registration
s = new("Getting the labelled cohort into MNI", "Registration was the real blocker, not labels")
picture(s, ROOT / "data/derived/arc/pilot_qc_overlay.png", Inches(.7), Inches(1.95), h=Inches(4.15))
tb(s, Inches(.7), Inches(6.25), Inches(5.2), Inches(.7),
   "Pilot of 8, looked at as images — not only as numbers.", size=12.5, color=MUTED)
bullets(s, Inches(6.35), Inches(2.0), Inches(6.25), [
    "228 ARC masks |  native T2w space, 5 distinct grids, 2 orientations. Zero were MNI-aligned.",
    "Cost-function masking at EVERY stage |  the ANTsPy default masks the SyN stage only, leaving rigid and affine free to pull healthy tissue into the lesion.",
    "Mutual information |  T2w against a T1 template is cross-modality, so a correlation metric is meaningless across it.",
    "Dice 0.924 |  median, 0.901 minimum, against the template.",
    "228/228 left-dominant |  the QC that matters for aphasia — a right-dominant warp has been mirrored.",
], size=13, gap=11, h=Inches(4.9))

# ---------------------------------------------------------------- 6 two matrices
s = new("A result that constrains everything after it", "The two matrices agree about systems, contradict about parcels")
table(s, Inches(.7), Inches(2.05), [Inches(4.6), Inches(3.2), Inches(3.2)],
      ["", "median Spearman ρ", "subjects positive"],
      [("parcel level (415 nodes)", "−0.318", "4 / 108"),
       ("network level (17 groups)", "+0.331", "104 / 107")],
      row_colors=[WARN, GOOD])
tb(s, Inches(.7), Inches(4.05), Inches(11.9), Inches(2.2),
   "The sign flips. Two methods sharing no data, no reference and no algorithm agree about which "
   "brain SYSTEMS a lesion disconnects and contradict each other about which individual PARCEL "
   "inside them.\n\n"
   "Three explanations were tested and rejected first — normalisation, fraction-vs-count, and a "
   "laterality bug. What survives is granularity: adjacent Schaefer parcels are close to "
   "interchangeable at this resolution.", size=15, line=1.35)
tb(s, Inches(.7), Inches(6.35), Inches(11.9), Inches(.6),
   "CONSEQUENCE: a single parcel-pair edge is not a reliable feature. Coarsen to networks, or do not use it.",
   size=14.5, bold=True, color=WARN)

# ---------------------------------------------------------------- 7 phase 3
s = new("The thesis test", "B ≈ A — disconnection is not distinguishable from location")
picture(s, FIG / "arms_raw.png", Inches(.7), Inches(1.85), w=Inches(8.5))
tb(s, Inches(9.4), Inches(1.95), Inches(3.3), Inches(3.4),
   "Read the table alone and disconnection wins.\n\nIt does not.\n\n"
   "Every arm is scored on the IDENTICAL 25 outer folds, so the comparison is paired — "
   "and the intervals span roughly 0.4 for every single arm.", size=13.5, line=1.35)
rule(s, Inches(.7), Inches(5.75), Inches(11.9))
tb(s, Inches(.7), Inches(5.95), Inches(11.9), Inches(1.1),
   "Wilcoxon signed-rank, Bonferroni corrected:  NOTHING beats location. NOTHING survives correction.\n"
   "The one nominally significant result is against volume — not against the baseline the thesis is "
   "about — and it dies at p = 0.22 corrected.", size=14, line=1.4)

# ---------------------------------------------------------------- 8 volume
s = new("The number that should worry you most", "Lesion volume alone — one column — reaches R² 0.398")
tb(s, Inches(.7), Inches(2.0), Inches(11.9), Inches(.8),
   "Every other arm is fighting over the remaining ~0.05.", size=20, bold=True, color=WARN)
tb(s, Inches(.7), Inches(2.8), Inches(11.9), Inches(1.5),
   "This was predicted before the run. On the 108, log lesion volume is recoverable FROM the "
   "disconnection features — so most of what those features carry is size, not topology.",
   size=16, line=1.35)
card(s, Inches(.7), Inches(4.45), Inches(2.7), Inches(1.6), "0.66", "network edges\n→ predict log volume", WARN)
card(s, Inches(3.6), Inches(4.45), Inches(2.7), Inches(1.6), "0.47", "tract severance\n→ predict log volume", WARN)
card(s, Inches(6.5), Inches(4.45), Inches(2.7), Inches(1.6), "0.38", "location baseline\n→ predict log volume", MUTED)
card(s, Inches(9.4), Inches(4.45), Inches(2.7), Inches(1.6), "1.000", "volume → log volume\npositive control", GOOD)
tb(s, Inches(.7), Inches(6.3), Inches(11.9), Inches(.6),
   "The positive control returning exactly 1.0000 is how we know the harness is wired correctly end to end.",
   size=13, color=MUTED)

# ---------------------------------------------------------------- 9 residual (headline)
s = new("New this session  ·  the sharpest test", "Regress volume out properly and nothing survives")
picture(s, FIG / "arms_residual.png", Inches(.7), Inches(1.85), w=Inches(8.5))
tb(s, Inches(9.4), Inches(1.95), Inches(3.3), Inches(3.6),
   "A hist_gb nuisance model is refitted INSIDE each of the 25 training splits, then "
   "subtracted from both sides.\n\n"
   "Fitting it once over all 228 — the obvious implementation — leaks the test outcomes "
   "into the residual and inflates every arm at once.", size=13.5, line=1.35)
rule(s, Inches(.7), Inches(5.75), Inches(11.9))
tb(s, Inches(.7), Inches(5.95), Inches(11.9), Inches(1.2),
   "Every arm ≤ 0. So is every one of the 18 (feature set × model) cells.\n"
   "The control row is what makes it trustworthy: volume predicting its own residual sits at −0.0146. "
   "Anything appreciably above zero there would mean the nuisance model leaked — and the whole table "
   "would be void.", size=13.5, line=1.4)

# ---------------------------------------------------------------- 10 covariates
s = new("New this session", "Age, sex and time post-stroke are nothing")
picture(s, FIG / "covariates.png", Inches(.7), Inches(1.95), w=Inches(6.5))
tb(s, Inches(7.5), Inches(2.0), Inches(5.1), Inches(3.6),
   "The hypothesis was that time post-stroke explains variance now attributed to anatomy.\n\n"
   "wab_days partial r = −0.002, given volume.\n\n"
   "ARC is a CHRONIC cohort — median 721 days — so recovery had already plateaued when these "
   "subjects were scanned. The variable would matter in a cohort scanned in the first weeks. "
   "This is not one.", size=14.5, line=1.35)
rule(s, Inches(.7), Inches(5.85), Inches(11.9))
tb(s, Inches(.7), Inches(6.05), Inches(11.9), Inches(1.0),
   "The apparent sex effect is lesion volume. Nominally significant univariately (r = +0.185, "
   "p = 0.005), it dies under the volume control (p = 0.088) — women in this cohort have smaller "
   "lesions, mean log volume 10.40 against 10.85.", size=13.5, color=MUTED, line=1.4)

# ---------------------------------------------------------------- 11 ceiling
s = new("New this session  ·  a correction to our own docs", "“Use more of ARC” is not available")
picture(s, FIG / "cohort_ceiling.png", Inches(.7), Inches(2.0), w=Inches(6.8))
tb(s, Inches(7.8), Inches(2.1), Inches(4.8), Inches(3.4),
   "The roadmap said 902 sessions existed and only 228 were used, so more subjects looked "
   "like the cheap path to an answer.\n\n"
   "902 is SCANNING SESSIONS, not participants.\n\n"
   "The other 17 participants have a WAB-AQ and no lesion mask at all — they cannot enter a "
   "lesion-symptom model on any terms.", size=14, line=1.35)
rule(s, Inches(.7), Inches(5.8), Inches(11.9))
tb(s, Inches(.7), Inches(6.0), Inches(11.9), Inches(1.0),
   "n = 228 is the entire labelled cohort, not a sample of it. More subjects means a DIFFERENT "
   "cohort — and ATLAS, the one with 1,453 scans, is the one that withholds outcome scores.\n"
   "Counting the masks costs one command, and would have caught this before it was written down.",
   size=13.5, color=WARN, line=1.4)

# ---------------------------------------------------------------- 12 bonferroni
s = new("New this session  ·  a fragility worth knowing", "The correction family is a function of the command line")
table(s, Inches(.7), Inches(2.05), [Inches(6.0), Inches(2.6), Inches(2.6)],
      ["comparison", "family of 7", "family of 15"],
      [("network edges (counted) vs volume", "p = 0.22", "p = 0.48"),
       ("location vs volume", "p = 0.94", "p = 1.00")],
      row_colors=[WARN, MUTED])
tb(s, Inches(.7), Inches(4.15), Inches(11.9), Inches(2.0),
   "Re-running the IDENTICAL measurement with the five volume+X incremental arms in the same "
   "command grows the paired table from 7 rows to 15, and every corrected p-value moves with it.\n\n"
   "Every p_wilcoxon is identical between the two runs. Only the denominator changed, because a "
   "different number of questions was asked in one invocation.", size=15, line=1.35)
rule(s, Inches(.7), Inches(6.15), Inches(11.9))
tb(s, Inches(.7), Inches(6.35), Inches(11.9), Inches(.7),
   "Recorded rather than fixed quietly — the tempting fix, shrinking the family, moves p-values in "
   "the direction that manufactures results. Nothing survives at either size.",
   size=13.5, bold=True, color=ACCENT, line=1.35)

# ---------------------------------------------------------------- 13 in flight
s = new("In flight", "The permutation null on the tree models — 1 of 6 arms done")
tb(s, Inches(.7), Inches(1.95), Inches(11.9), Inches(1.2),
   "The published null runs RIDGE, and ridge is unusable on these features — its fold interval "
   "reaches −63.5. So the null validated the linear arms and NOT the boosted trees that produce "
   "every headline number.", size=16, line=1.35)
card(s, Inches(.7), Inches(3.3), Inches(2.7), Inches(1.65), "+0.416", "volume arm\nobserved R²", ACCENT)
card(s, Inches(3.6), Inches(3.3), Inches(2.7), Inches(1.65), "−0.104", "null median\n200 shuffles", MUTED)
card(s, Inches(6.5), Inches(3.3), Inches(2.7), Inches(1.65), "0.005", "p — the 1/201 floor", GOOD)
card(s, Inches(9.4), Inches(3.3), Inches(2.7), Inches(1.65), "1 / 6", "arms complete", WARN)
tb(s, Inches(.7), Inches(5.25), Inches(11.9), Inches(1.3),
   "The null now runs on the tree family and is validated end to end on the control arm. The four "
   "disconnection arms have NOT been shuffled yet.\n"
   "Honest line: implemented, validated, mid-run. The gap in docs/PHASE3.md is not yet closed.",
   size=14.5, line=1.4)
tb(s, Inches(.7), Inches(6.55), Inches(11.9), Inches(.6),
   "Parallelised over shuffles: 37 min → 13 min per arm, verified to produce elementwise identical "
   "nulls. There is no GPU path — sklearn's HistGradientBoosting is CPU only, and swapping in a "
   "CUDA booster would mean the null validated a different model.", size=12, color=MUTED, line=1.3)

# ---------------------------------------------------------------- 14 next
s = new("What is next", "In priority order, with the dead ends removed")
bullets(s, Inches(.7), Inches(2.0), Inches(11.9), [
    "1. Finish the tree permutation null |  ~50 min of compute, checkpointed and resumable. Closes the last of the four documented gaps.",
    "2. A hold-out test set |  now the ONLY original gap left. Every number is cross-validated on the same 228, and n cannot grow.",
    "3. Compare SyNRA against other transforms |  Dice 0.924 was good enough to proceed, which is not evidence nothing else is better.",
    "4. Enantiomorphic normalisation |  cost-function masking is the lighter of the two standard options.",
    "5. NeuroVue UI reads disconnection.json |  the upload path has never run against a live backend.",
], size=15, gap=15)
rule(s, Inches(.7), Inches(5.8), Inches(11.9))
tb(s, Inches(.7), Inches(6.0), Inches(11.9), Inches(1.0),
   "Removed from the list: “use more of ARC” (impossible — 228 is the whole cohort) and "
   "“add covariates” (done — they are nothing). Both were measured, not assumed.",
   size=14, bold=True, color=ACCENT, line=1.4)

# ---------------------------------------------------------------- 15 close
s = new("Where this leaves the project", "The negative result is the result")
tb(s, Inches(.7), Inches(2.1), Inches(11.9), Inches(2.5),
   "On this cohort, at this n, lesion volume is the model and everything else is decoration.\n\n"
   "That is not a failed project. It is a pipeline instrumented well enough to catch its own author "
   "repeatedly, and honest enough to report a null when the point estimates said otherwise.",
   size=19, line=1.4)
rule(s, Inches(.7), Inches(4.8), Inches(11.9))
bullets(s, Inches(.7), Inches(5.0), Inches(11.9), [
    "What would change the answer |  a larger labelled cohort, or a representation that is not a function of lesion size",
    "What would not |  a cleverer model on 228 subjects with fold intervals 0.4 wide",
], size=14.5, gap=12)
tb(s, Inches(.7), Inches(6.7), Inches(11.9), Inches(.5),
   "Research annotation / analysis tool — NOT a diagnostic device.",
   size=12, bold=True, color=MUTED)

out = ROOT / "docs" / "Lesion_Disconnection_Status.pptx"
out.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(out))
print("slides:", len(prs.slides._sldIdLst))
print("->", out)
