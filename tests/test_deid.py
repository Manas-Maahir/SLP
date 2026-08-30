"""The de-identification guarantee, enforced by grep.

Milestone 0's rule is absolute: after ids are assigned, no patient name or MRN
may appear in ANY artifact outside ``data/private/`` -- manifests, features,
filenames, figures, logs, reports, source, docs.

This is a whole-repo grep rather than a unit test on purpose. A unit test would
check the function that de-identifies; this checks the property that actually
matters, including artifacts written by code that never called that function.
Binary files (NIfTI, PNG) are scanned as bytes, because NIfTI headers carry a
``descrip`` field and matplotlib embeds titles in PNG metadata.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lesionconn import config, deid

ROOT = config.PROJECT_ROOT

# Everything under here is expected to hold identifiers, or is not ours.
EXCLUDED_DIRS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".ipynb_checkpoints", "node_modules",
}

# Compressed bytes are not searchable as plain text; a name inside a .nii.gz
# would be invisible to a byte grep. So gzipped NIfTIs are decompressed first
# (see _read_bytes) rather than skipped.
GZIP_SUFFIXES = (".gz",)

MAX_BYTES = 64 * 1024 * 1024


def _is_private(path: Path) -> bool:
    return config.PRIVATE_DIR in path.parents or path == config.SUBJECT_MAP_PATH


def _walk_artifacts() -> list[Path]:
    out: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if _is_private(path):
            continue
        out.append(path)
    return out


def _read_bytes(path: Path) -> bytes:
    if path.stat().st_size > MAX_BYTES:
        return b""
    raw = path.read_bytes()
    if path.suffix.lower() in GZIP_SUFFIXES:
        import gzip
        try:
            return gzip.decompress(raw)
        except OSError:
            return raw
    return raw


@pytest.fixture(scope="module")
def tokens() -> set[str]:
    if not config.SUBJECT_MAP_PATH.exists():
        pytest.skip(
            "no subject map on this machine -- data/ is gitignored, so a fresh clone "
            "has nothing to leak. Run scripts/10_deidentify.py to enable this test."
        )
    toks = deid.sensitive_tokens()
    assert toks, "subject map produced no identifying tokens -- the extractor is broken"
    return toks


def test_no_patient_token_appears_in_any_emitted_artifact(tokens):
    """The whole point of Milestone 0."""
    # One regex over all tokens: 260 separate scans of every file takes minutes;
    # this takes under a second. Word boundaries keep a short name from firing inside a
    # longer ordinary word. (This test first failed on a real cohort name quoted
    # in this very comment -- which is the argument for scanning our own source.)
    pattern = re.compile(
        rb"\b(?:" + b"|".join(re.escape(t.encode()) for t in sorted(tokens)) + rb")\b",
        re.IGNORECASE,
    )

    leaks: list[str] = []
    for path in _walk_artifacts():
        blob = _read_bytes(path)
        if blob and pattern.search(blob):
            leaks.append(str(path.relative_to(ROOT)))

    assert not leaks, (
        "patient-identifying tokens found outside data/private/ in "
        f"{len(leaks)} file(s):\n  " + "\n  ".join(leaks[:20])
    )


def test_no_emitted_filename_contains_a_patient_token(tokens):
    """Filenames leak as loudly as contents, and figures get named after subjects."""
    leaks = [
        str(p.relative_to(ROOT))
        for p in _walk_artifacts()
        if deid.filename_tokens(p.name) & tokens
    ]
    assert not leaks, "patient tokens in emitted filenames:\n  " + "\n  ".join(leaks[:20])


def test_subject_map_is_gitignored():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/private/" in text
    assert "data/" in text


def test_subject_ids_are_sequential_and_well_formed():
    if not config.SUBJECT_MAP_PATH.exists():
        pytest.skip("no subject map on this machine")
    df = deid.load_subject_map()
    assert df["subject_id"].is_unique
    assert df["source_path"].is_unique
    assert all(deid.SUBJECT_ID_RE.match(s) for s in df["subject_id"])
    expected = [deid.SUBJECT_ID_FMT.format(i) for i in range(1, len(df) + 1)]
    assert list(df["subject_id"]) == expected


def test_id_assignment_is_deterministic(tmp_path):
    """Re-running must not renumber a cohort."""
    for name in ("Zed_999999Z_lesion_MNI.nii", "Ann_111111A_lesion_MNI.nii"):
        (tmp_path / name).write_bytes(b"")
    first = deid.build_subject_map(tmp_path)
    second = deid.build_subject_map(tmp_path)
    assert list(first["subject_id"]) == list(second["subject_id"])
    assert first.loc[first.source_filename.str.startswith("Ann"), "subject_id"].iloc[0] == "sub-0001"


def test_filename_tokens_keeps_names_and_mrns_but_drops_pipeline_words():
    toks = deid.filename_tokens("Jane-Doe_123456X_lesion_MNI.nii")
    assert "jane" in toks and "doe" in toks and "123456x" in toks
    assert "lesion" not in toks and "mni" not in toks and "nii" not in toks


def test_save_subject_map_refuses_a_path_outside_private(tmp_path):
    import pandas as pd

    df = pd.DataFrame([{"subject_id": "sub-0001", "source_filename": "a.nii", "source_path": "a"}])
    with pytest.raises(ValueError, match="may only live under"):
        deid.save_subject_map(df, tmp_path / "subject_map.csv")


def test_assert_deidentified_catches_a_name_in_report_text():
    toks = {"jane", "123456x"}
    deid.assert_deidentified("sub-0001 lesion volume 4200 mm3", toks, "report")
    with pytest.raises(ValueError, match="patient-identifying token"):
        deid.assert_deidentified("Report for Jane, MRN 123456X", toks, "report")
