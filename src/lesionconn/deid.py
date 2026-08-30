"""De-identification: the cohort arrives with names and MRNs in its filenames.

The 108-lesion cohort ships as ``<Patient Name>_<MRN>_lesion_MNI.nii``. Those
strings must not survive into anything this project emits -- not a manifest, not
a cached NIfTI, not a figure title, not a log line, not a report.

The contract:

  * ``data/private/subject_map.csv`` is the ONLY artifact that may hold a source
    filename. It is gitignored twice (``data/`` and ``data/private/``).
  * Everything else addresses a subject by ``sub-NNNN`` and nothing else.
  * ``tests/test_deid.py`` greps every emitted artifact for the tokens
    ``sensitive_tokens()`` derives from the map, and fails on any hit. That test
    is the enforcement; this module only makes it cheap to write.

Ids are assigned by sorting source filenames, so the same directory always
yields the same ids -- re-running never silently renumbers a cohort.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import config

SUBJECT_ID_FMT = "sub-{:04d}"
SUBJECT_ID_RE = re.compile(r"^sub-\d{4}$")

# Tokens that appear in the source filenames but carry no identity: the pipeline
# vocabulary. Everything else in a filename is treated as identifying, which is
# the safe direction to be wrong in.
NON_IDENTIFYING_TOKENS = frozenset(
    {"nii", "gz", "mni", "lesion", "lesions", "mask", "masks", "resliced", "patients"}
)

# Below this length a token is an initial or a fragment ("A", "PS"), and grepping
# for it would match essentially every file in the repo. Such tokens are not
# identifying on their own; the full names and the MRNs are, and those are all
# well above this length.
MIN_TOKEN_LEN = 3

_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def _stem(filename: str) -> str:
    name = filename
    for suffix in (".nii.gz", ".nii"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def filename_tokens(filename: str) -> set[str]:
    """Lowercased identifying tokens in one source filename."""
    return {
        tok.lower()
        for tok in _TOKEN_SPLIT.split(_stem(filename))
        if len(tok) >= MIN_TOKEN_LEN and tok.lower() not in NON_IDENTIFYING_TOKENS
    }


def build_subject_map(source_dir: str | Path, pattern: str = "*.nii*") -> pd.DataFrame:
    """Assign ``sub-0001``.. to every mask in ``source_dir``, ordered by filename.

    Deterministic by construction: the id depends only on the sorted position of
    the filename, so a re-run of the same directory reproduces the same map.
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"{source_dir} is not a directory")

    paths = sorted(source_dir.glob(pattern), key=lambda p: p.name)
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} in {source_dir}")

    rows = [
        {
            "subject_id": SUBJECT_ID_FMT.format(i),
            "source_filename": p.name,
            "source_path": str(p.resolve()),
        }
        for i, p in enumerate(paths, start=1)
    ]
    return pd.DataFrame(rows)


def save_subject_map(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = Path(path or config.SUBJECT_MAP_PATH)
    if config.PRIVATE_DIR not in path.parents:
        raise ValueError(
            f"refusing to write the subject map to {path}: it holds patient names and "
            f"MRNs and may only live under {config.PRIVATE_DIR}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_subject_map(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or config.SUBJECT_MAP_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/10_deidentify.py before anything that "
            "touches the cohort"
        )
    return pd.read_csv(path)


def sensitive_tokens(df: pd.DataFrame | None = None) -> set[str]:
    """Every token the leak test must not find outside data/private/."""
    if df is None:
        df = load_subject_map()
    tokens: set[str] = set()
    for name in df["source_filename"]:
        tokens |= filename_tokens(str(name))
    return tokens


def source_path_for(subject_id: str, df: pd.DataFrame | None = None) -> Path:
    """Resolve a de-identified id back to its source file. Private side only."""
    if df is None:
        df = load_subject_map()
    hits = df.loc[df["subject_id"] == subject_id, "source_path"]
    if hits.empty:
        raise KeyError(f"{subject_id} is not in the subject map")
    return Path(str(hits.iloc[0]))


def token_pattern(tokens) -> re.Pattern[str]:
    """One compiled matcher for "does this text contain a patient identifier".

    Word-bounded, not substring. Two real cohort first names happen to sit inside
    ``corticothalamic_pathway_l`` and ``device``; a substring rule flags both and
    would block the pipeline from ever writing a tract name or the disclaimer.
    A name embedded in a longer word is not a disclosure.

    Both callers -- this module's write guard and the whole-repo grep in
    tests/test_deid.py -- go through here, so the two cannot drift apart. They
    did drift once, and the write guard was the looser of the two.
    """
    if not tokens:
        return re.compile(r"(?!x)x")  # matches nothing
    return re.compile(
        r"\b(?:" + "|".join(re.escape(t) for t in sorted(tokens)) + r")\b", re.IGNORECASE
    )


def assert_deidentified(text: str, tokens: set[str] | None = None, what: str = "text") -> None:
    """Raise if any patient token appears in ``text``. For anything about to be written."""
    tokens = sensitive_tokens() if tokens is None else tokens
    found = sorted(set(token_pattern(tokens).findall(text)))
    if found:
        raise ValueError(
            f"{what} contains {len(found)} patient-identifying token(s) and must not be "
            f"written outside {config.PRIVATE_DIR}. (Tokens withheld from this message.)"
        )


@lru_cache(maxsize=1)
def _cached_tokens() -> frozenset[str]:
    try:
        return frozenset(sensitive_tokens())
    except FileNotFoundError:
        return frozenset()


def guard_output(text: str, what: str = "output") -> None:
    """`assert_deidentified` for pipeline code, tolerant of no cohort being present.

    Synthetic-lesion runs and fresh clones have no subject map and therefore
    nothing to leak, so the check no-ops. When a cohort IS mapped, every string
    the pipeline is about to write goes through here.
    """
    tokens = _cached_tokens()
    if tokens:
        assert_deidentified(text, set(tokens), what)
