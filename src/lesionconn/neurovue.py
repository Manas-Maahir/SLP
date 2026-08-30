"""Hand this pipeline's output to the NeuroVue viewer.

NeuroVue (React + Niivue + FastAPI, a separate project) already knows how to load
a lesion NIfTI: `POST /api/lesions` takes an uploaded `.nii.gz` plus a name and
writes it to `LESION_DIR/<slug>/lesion_<timestamp>.nii.gz`.

So **nothing in NeuroVue is modified**. All the integration lives here, in two
shapes, because NeuroVue's own route needs a MongoDB it may not have running:

* `export_bundle` writes the same on-disk layout that route produces, so pointing
  NeuroVue's `LESION_DIR` at the bundle makes the masks visible with no server
  call at all. This is the file drop.
* `post_lesion` uses the documented route when a backend *is* running.

**Verification status.** `export_bundle` is verified end to end -- the bundle is
written and its layout, contents and de-identification are asserted. `post_lesion`
is verified against a stub implementing the contract NeuroVue's route declares
(multipart `file` + `name` + `base`); it has NOT been run against a live NeuroVue
backend, which would need that project's FastAPI app and its MongoDB running.

Only de-identified artifacts leave this repo. The masks are the cached canonical
ones, already named `sub-NNNN`, and every filename and JSON body is checked
against `deid` before it is written or sent.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import config, deid

API_PATH = "/api/lesions"
# Matches NeuroVue's own naming in backend/server.py save_lesion().
LESION_FILENAME_FMT = "lesion_{ts}.nii.gz"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _severed_tracts_summary(subject_id: str, top_n: int = 15) -> dict:
    """The disconnection profile, trimmed to what a viewer sidebar can show."""
    from .cohort import subject_paths

    path = subject_paths(subject_id)["severed_tracts_path"]
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "subject_id": payload.get("subject_id", subject_id),
        "atlas": payload.get("atlas", {}),
        "lesion": payload.get("lesion", {}),
        "n_tracts_involved": payload.get("n_tracts_involved", 0),
        "tracts": payload.get("tracts", [])[:top_n],
        "measure": payload.get("measure", ""),
        "disclaimer": payload.get("disclaimer", config.DISCLAIMER),
    }


def export_bundle(
    subject_ids: list[str],
    dest: str | Path,
    include_disconnection: bool = True,
) -> Path:
    """Write a NeuroVue-shaped lesion store for the given subjects.

    Layout mirrors NeuroVue's `LESION_DIR` exactly, so:

        LESION_DIR=<dest> uvicorn server:app --port 8001

    picks the masks up with no code change and no database write.
    """
    from .cohort import subject_paths

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    ts = _timestamp()

    entries = []
    for subject_id in subject_ids:
        if not deid.SUBJECT_ID_RE.match(subject_id):
            raise ValueError(
                f"refusing to export {subject_id!r}: only de-identified sub-NNNN ids "
                "may leave this repo"
            )
        source = subject_paths(subject_id)["lesion_path"]
        if not source.exists():
            raise FileNotFoundError(
                f"{source} missing -- run scripts/13_build_cohort.py first"
            )

        subject_dir = dest / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        target = subject_dir / LESION_FILENAME_FMT.format(ts=ts)
        shutil.copyfile(source, target)

        entry = {
            "name": subject_id,
            "slug": subject_id,
            "file": target.name,
            "bytes": target.stat().st_size,
            "savedAt": datetime.now(timezone.utc).isoformat(),
        }

        if include_disconnection:
            summary = _severed_tracts_summary(subject_id)
            if summary:
                text = json.dumps(summary, indent=2)
                deid.guard_output(text, f"disconnection.json for {subject_id}")
                (subject_dir / "disconnection.json").write_text(text, encoding="utf-8")
                entry["disconnection"] = "disconnection.json"
                entry["n_tracts_involved"] = summary.get("n_tracts_involved", 0)

        entries.append(entry)

    index = {
        "source": "lesionconn",
        "space": {
            "shape": list(config.CANONICAL_SHAPE),
            "voxel_mm": [2.0, 2.0, 2.0],
            "orientation": "LAS",
            "template": "FSL MNI152",
        },
        "n_subjects": len(entries),
        "lesions": entries,
        "disclaimer": config.DISCLAIMER,
        "note": (
            "Point NeuroVue's LESION_DIR at this directory. Masks are on the FSL "
            "MNI152 2mm grid; load them over an MNI152 template, not a native scan."
        ),
    }
    text = json.dumps(index, indent=2)
    deid.guard_output(text, "neurovue index.json")
    (dest / "index.json").write_text(text, encoding="utf-8")
    return dest


def post_lesion(base_url: str, subject_id: str, lesion_path: Path, timeout: float = 30.0):
    """Upload one mask through NeuroVue's existing POST /api/lesions.

    Used only when a NeuroVue backend is actually running; the file drop above
    needs no server. The multipart fields are the ones that route declares:
    `file`, `name`, and an optional `base`.
    """
    import requests

    if not deid.SUBJECT_ID_RE.match(subject_id):
        raise ValueError(f"refusing to upload {subject_id!r}: not a de-identified id")

    lesion_path = Path(lesion_path)
    with lesion_path.open("rb") as handle:
        response = requests.post(
            base_url.rstrip("/") + API_PATH,
            files={"file": (f"{subject_id}.nii.gz", handle, "application/gzip")},
            data={"name": subject_id, "base": "MNI152_2mm_LAS"},
            timeout=timeout,
        )
    response.raise_for_status()
    return response.json()
