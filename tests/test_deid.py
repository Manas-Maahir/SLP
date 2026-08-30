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


def _npy_searchable(path: Path, raw: bytes) -> bytes | None:
    """The bytes of a .npy that could possibly carry text. None if not a .npy.

    A .npy is an ASCII header dict followed by the raw array buffer.

    * **Numeric payload** -- only the header can hold text, so only the header is
      searched. This narrowing is not a convenience: byte-grepping a float32
      buffer produces coincidences, and 3 of the 108 cohort matrices matched a
      three-letter cohort name inside their mantissa bytes. (0x3B, the exponent
      byte of small floats, is dense in this data, so such collisions are common
      rather than rare.) A check that fires on arithmetic teaches you to ignore
      it, which is worse than not having it.
    * **Text payload** -- searched in full, and DECODED first. numpy stores
      unicode arrays as UTF-32, so a name in one is invisible to a byte grep;
      decoding closes that gap rather than trusting the raw bytes.
    """
    if not raw.startswith(b"\x93NUMPY"):
        return None
    end = raw.find(b"\n", 0, 4096)
    if end == -1:
        return None
    header = raw[: end + 1]

    descr = header.split(b"'descr':")[1][:16] if b"'descr':" in header else b""
    if not any(code in descr for code in (b"U", b"S", b"O")):
        return header

    try:
        import numpy as np

        array = np.load(path, allow_pickle=False)
        decoded = " ".join(str(v) for v in array.ravel().tolist())
        return header + decoded.encode("utf-8", "replace")
    except Exception:
        return raw  # cannot decode it, so search everything rather than nothing


def _read_bytes(path: Path) -> bytes:
    if path.stat().st_size > MAX_BYTES:
        return b""
    raw = path.read_bytes()
    if path.suffix.lower() in GZIP_SUFFIXES:
        import contextlib
        import gzip

        with contextlib.suppress(OSError):
            raw = gzip.decompress(raw)
    searchable = _npy_searchable(path, raw)
    return raw if searchable is None else searchable


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
    # deid.token_pattern is the single definition of "this text names a patient",
    # shared with the write guard so the two cannot disagree. Encoded to bytes so
    # NIfTI headers and PNG metadata are searched the same way as source.
    pattern = re.compile(
        deid.token_pattern(tokens).pattern.encode(), re.IGNORECASE
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


def test_npy_narrowing_still_catches_a_name_in_the_header_or_a_text_payload(tmp_path):
    """The .npy narrowing must not become a hole to hide a name in.

    Only the raw numeric buffer is skipped. A name in the header, or a .npy whose
    dtype is textual, still has to be found.
    """
    import numpy as np

    tokens = {"jane", "123456x"}
    pattern = re.compile(deid.token_pattern(tokens).pattern.encode(), re.IGNORECASE)

    # a numeric payload: skipped, so an accidental byte collision cannot fire
    numeric = tmp_path / "numeric.npy"
    np.save(numeric, np.zeros((4, 4), dtype=np.float32))
    assert not pattern.search(_read_bytes(numeric))

    # a TEXT payload: decoded, then fully searched. Decoding matters -- numpy
    # stores unicode as UTF-32, so "Jane" is not contiguous in the raw bytes.
    textual = tmp_path / "textual.npy"
    np.save(textual, np.array(["Jane", "ok"]))
    assert b"Jane" not in textual.read_bytes(), "precondition: UTF-32, not raw ASCII"
    assert pattern.search(_read_bytes(textual)), "a string-dtype .npy must still be scanned"

    # and the same for a bytes-dtype array, where the name IS contiguous
    raw_bytes = tmp_path / "bytes.npy"
    np.save(raw_bytes, np.array([b"Jane"]))
    assert pattern.search(_read_bytes(raw_bytes))

    # a name smuggled into the header region is still visible
    doctored = tmp_path / "doctored.npy"
    doctored.write_bytes(b"\x93NUMPY\x01\x00v\x00{'descr': 'Jane', 'shape': (1,)}\n" + b"\x00" * 64)
    assert pattern.search(_read_bytes(doctored))


def test_nifti_payloads_are_still_scanned_in_full(tmp_path):
    """NIfTI headers carry a free-text descrip field, so nothing is narrowed there."""
    import nibabel as nib
    import numpy as np

    tokens = {"jane"}
    pattern = re.compile(deid.token_pattern(tokens).pattern.encode(), re.IGNORECASE)
    img = nib.Nifti1Image(np.zeros((4, 4, 4), np.int16), np.eye(4))
    img.header["descrip"] = b"Jane"
    path = tmp_path / "leaky.nii.gz"
    nib.save(img, path)
    assert pattern.search(_read_bytes(path)), "a name in a NIfTI descrip must be found"
