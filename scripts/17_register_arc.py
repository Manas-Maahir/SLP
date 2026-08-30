"""M1 over ARC: 228 native-space lesion masks -> the canonical grid.

Run the pilot FIRST and read its output:

    ../.venv/Scripts/python.exe 17_register_arc.py --pilot 8

A batch of 228 is long enough that launching it on unvalidated settings costs a
whole session, and every wrong threshold in this project so far was wrong on the
first guess (docs/BUGS.md). The pilot registers a handful, saves the warped T2w
next to the template for visual comparison, and prints the QC distribution that
decides whether the batch is worth starting.

Then:

    ../.venv/Scripts/python.exe 17_register_arc.py --workers 4

Outputs land in data/derived/arc/: one canonical-grid mask per subject plus
registration_qc.csv. Nothing is written into the 108-lesion cohort's caches.
"""

import _bootstrap  # noqa: F401
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import pandas as pd

from lesionconn import config

ARC_DIR = config.RAW_DIR / "arc"
MASK_DIR = ARC_DIR / "derivatives" / "lesion_masks"
OUT_DIR = config.DERIVED_DIR / "arc"
LESION_DIR = OUT_DIR / "lesions"
QC_PATH = OUT_DIR / "registration_qc.csv"
WARPED_T2W_DIR = OUT_DIR / "pilot_warped_t2w"


def subject_table() -> pd.DataFrame:
    """Every ARC subject with a mask, its T2w, and its WAB-AQ."""
    labels = pd.read_csv(ARC_DIR / "participants.tsv", sep="\t").set_index("participant_id")
    rows = []
    for sidecar in sorted(MASK_DIR.rglob("*_desc-lesion_mask.json")):
        mask = sidecar.with_name(sidecar.name.replace(".json", ".nii.gz"))
        ref = json.loads(sidecar.read_text()).get("SpatialReference", "")
        if ref.startswith("bids:raw:"):
            t2w = ARC_DIR / ref[len("bids:raw:") :]
        else:
            stem = sidecar.name.replace("_desc-lesion_mask.json", "")
            t2w = ARC_DIR / sidecar.relative_to(MASK_DIR).parent / f"{stem}.nii.gz"
        subject_id = sidecar.name.split("_")[0]
        if subject_id not in labels.index:
            continue
        rec = labels.loc[subject_id]
        rows.append(
            {
                "subject_id": subject_id,
                "t2w_path": str(t2w),
                "mask_path": str(mask),
                "wab_aq": float(rec["wab_aq"]),
                "wab_type": rec["wab_type"],
                "age_at_stroke": rec["age_at_stroke"],
                "sex": rec["sex"],
                "wab_days": rec["wab_days"],
            }
        )
    return pd.DataFrame(rows)


def _register_one(args: tuple) -> dict:
    """Worker. ITK threads are capped here because the parallelism is per-subject."""
    row, transform, threads, save_t2w = args
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    from lesionconn import registration as reg

    sid = row["subject_id"]
    try:
        img, qc = reg.register_lesion(
            Path(row["t2w_path"]),
            Path(row["mask_path"]),
            sid,
            transform=transform,
            out_dir=WARPED_T2W_DIR if save_t2w else None,
            save_warped_t2w=save_t2w,
        )
        LESION_DIR.mkdir(parents=True, exist_ok=True)
        nib.save(img, LESION_DIR / f"{sid}_lesion.nii.gz")
        out = qc.as_row()
        out["ok"] = True
        out["error"] = ""
        return out
    except Exception as exc:
        return {"subject_id": sid, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def summarise(qc: pd.DataFrame) -> str:
    ok = qc[qc["ok"]]
    lines = [f"registered {len(ok)}/{len(qc)}"]
    if qc["ok"].eq(False).any():
        for _, r in qc[~qc["ok"]].iterrows():
            lines.append(f"  FAILED {r['subject_id']}: {r['error']}")
    if ok.empty:
        return "\n".join(lines)

    def dist(col, fmt="{:.3f}"):
        s = ok[col].astype(float)
        return (
            f"  {col:20s} median {fmt.format(s.median())}   "
            f"p5 {fmt.format(s.quantile(0.05))}   p95 {fmt.format(s.quantile(0.95))}"
        )

    lines += [
        "",
        dist("seconds", "{:.1f}"),
        dist("volume_delta_pct", "{:+.1f}"),
        dist("frac_outside_brain"),
        dist("frac_left"),
        dist("mutual_information", "{:.4f}"),
        "",
        # The check that actually matters. ARC is an aphasia cohort, so a warp
        # that comes back right-dominant is mirrored -- the bug class that hit
        # Phase 0 and Phase 1. Stated as anatomy, not as a number to admire.
        f"  left-dominant (frac_left > 0.5): {int((ok['frac_left'] > 0.5).sum())}/{len(ok)}"
        f"  = {100 * (ok['frac_left'] > 0.5).mean():.1f}%",
        f"  outside brain > 10%            : {int((ok['frac_outside_brain'] > 0.10).sum())}",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0, help="register only the first N, verbosely")
    ap.add_argument("--transform", default="SyNRA")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4, help="ITK threads per worker")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    table = subject_table()
    print(f"{len(table)} ARC subjects with a mask and a WAB-AQ")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.pilot:
        table = table.head(args.pilot)
        print(f"PILOT: {len(table)} subjects, sequential, saving warped T2w for visual QC\n")
    elif not args.force:
        done = {p.name.split("_lesion")[0] for p in LESION_DIR.glob("*_lesion.nii.gz")}
        before = len(table)
        table = table[~table["subject_id"].isin(done)]
        print(f"{before - len(table)} already registered, {len(table)} to do")

    if table.empty:
        print("nothing to do")
        return 0

    jobs = [
        (row, args.transform, args.threads, bool(args.pilot))
        for row in table.to_dict("records")
    ]

    results = []
    if args.pilot or args.workers <= 1:
        for i, job in enumerate(jobs, 1):
            r = _register_one(job)
            results.append(r)
            flag = "" if r.get("ok") else "  <-- FAILED"
            print(
                f"  {i}/{len(jobs)}  {r['subject_id']}  "
                + (
                    f"{r['seconds']:5.1f}s  left={r['frac_left']:.2f}  "
                    f"dV={r['volume_delta_pct']:+.1f}%  out={r['frac_outside_brain']:.3f}"
                    if r.get("ok")
                    else r["error"]
                )
                + flag
            )
    else:
        print(f"{args.workers} workers x {args.threads} ITK threads\n")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, r in enumerate(pool.map(_register_one, jobs), 1):
                results.append(r)
                if not r.get("ok"):
                    print(f"  !! {r['subject_id']}: {r['error']}")
                elif i % 10 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)}")

    fresh = pd.DataFrame(results)
    if QC_PATH.exists() and not args.force:
        prior = pd.read_csv(QC_PATH)
        fresh = pd.concat([prior[~prior["subject_id"].isin(fresh["subject_id"])], fresh])
    fresh = fresh.sort_values("subject_id")
    fresh.to_csv(QC_PATH, index=False)

    print("\n" + summarise(fresh))
    print(f"\nQC  -> {QC_PATH}")
    print(f"masks -> {LESION_DIR}")
    if args.pilot:
        print(f"warped T2w -> {WARPED_T2W_DIR}  (compare against the MNI template by eye)")
    return 0 if fresh["ok"].all() else 1


if __name__ == "__main__":
    sys.exit(main())
