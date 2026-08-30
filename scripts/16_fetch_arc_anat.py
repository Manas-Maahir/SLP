"""Fetch the ARC T2w anatomicals that the lesion masks are defined on.

Phase 0 pulled ARC metadata and the lesion-mask derivatives only. The masks are
in native T2w space and a binary mask carries no anatomy, so nothing can drive a
warp from it -- registration needs the image each mask was drawn on.

Each mask sidecar names that image:

    "SpatialReference": "bids:raw:sub-M2001/ses-1253/anat/..._T2w.nii.gz"

so this is a targeted 228-file pull (~1 GB), not the blind sync of all 902
sessions that scripts/03_fetch_arc.py warns about. Re-running skips whatever is
already on disk and verifies gzip magic on every file, so a truncated download
is re-fetched rather than silently registered later.
"""

import _bootstrap  # noqa: F401
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lesionconn import config

DATASET = "ds004884"
TAG = "1.0.2"
ARC_DIR = config.RAW_DIR / "arc"
MASK_DIR = ARC_DIR / "derivatives" / "lesion_masks"
BASE = f"https://openneuro.org/crn/datasets/{DATASET}/snapshots/{TAG}/files/"
WORKERS = 8
GZIP_MAGIC = b"\x1f\x8b"


def wanted_t2w() -> list[str]:
    """Dataset-relative paths of every T2w a lesion mask is defined on."""
    paths = []
    for sidecar in sorted(MASK_DIR.rglob("*_desc-lesion_mask.json")):
        ref = json.loads(sidecar.read_text()).get("SpatialReference", "")
        if ref.startswith("bids:raw:"):
            paths.append(ref[len("bids:raw:") :])
            continue
        # sub-M2272's sidecar says "Text" rather than a path. BIDS naming is
        # deterministic, so the mask's own filename yields the same answer;
        # the download 404s loudly if that ever stops being true.
        stem = sidecar.name.replace("_desc-lesion_mask.json", "")
        rel = sidecar.relative_to(MASK_DIR).parent
        paths.append(f"{rel.as_posix()}/{stem}.nii.gz")
    return sorted(set(paths))


def is_complete(dest: Path) -> bool:
    return dest.exists() and dest.stat().st_size > 0 and dest.open("rb").read(2) == GZIP_MAGIC


def fetch(rel: str) -> tuple[str, str, int]:
    dest = ARC_DIR / rel
    if is_complete(dest):
        return rel, "cached", dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = BASE + rel.replace("/", ":")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=180) as r:
            body = r.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return rel, f"FAILED {type(exc).__name__}", 0
    if body[:2] != GZIP_MAGIC:
        return rel, "FAILED not-gzip", len(body)
    tmp.write_bytes(body)
    tmp.replace(dest)
    return rel, "ok", len(body)


def main() -> int:
    paths = wanted_t2w()
    print(f"{len(paths)} T2w images referenced by the lesion masks")
    todo = [p for p in paths if not is_complete(ARC_DIR / p)]
    print(f"{len(paths) - len(todo)} already on disk, {len(todo)} to fetch\n")

    done = failed = 0
    total = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for rel, status, size in pool.map(fetch, todo):
            done += 1
            total += size
            if status.startswith("FAILED"):
                failed += 1
                print(f"  !! {status}  {rel}")
            elif done % 20 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}  {total / 1e9:.2f} GB")

    have = sum(1 for p in paths if is_complete(ARC_DIR / p))
    print(f"\ncomplete on disk: {have}/{len(paths)}   failures this run: {failed}")
    if have < len(paths):
        print("re-run to retry the stragglers; cached files are skipped")
    return 0 if have == len(paths) else 1


if __name__ == "__main__":
    sys.exit(main())
