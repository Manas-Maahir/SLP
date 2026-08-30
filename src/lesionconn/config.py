"""Project paths and the canonical coordinate space.

Everything in this project lives on ONE grid. The choice is the FSL MNI152 2mm
grid, because Schaefer-400 and Harvard-Oxford both ship natively on it -- so the
parcellation needs no resampling and only incoming patient data is ever moved.

The constants below are a documented *expectation*. `scripts/00_env_check.py`
verifies them against the actually-fetched atlas images; if reality disagrees,
the constants follow the data, not this file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ATLAS_DIR = DATA_DIR / "atlases"
DERIVED_DIR = DATA_DIR / "derived"

# Patient-identifying material lives here and NOWHERE else. The 108-lesion cohort
# arrives with names and MRNs in its filenames; the id<->filename map is the only
# artifact allowed to hold them. Gitignored twice over (data/ and data/private/).
PRIVATE_DIR = DATA_DIR / "private"
SUBJECT_MAP_PATH = PRIVATE_DIR / "subject_map.csv"

# De-identified per-subject caches, all named by subject_id only.
COHORT_DIR = DERIVED_DIR / "cohort"
LESION_CACHE_DIR = COHORT_DIR / "lesions"
DISCONNECTOME_DIR = COHORT_DIR / "disconnectome"
MANIFEST_PATH = COHORT_DIR / "manifest.csv"
REPORTS_DIR = DERIVED_DIR / "reports"

# --- tractography atlas (M3b) --------------------------------------------
# HCP842 ships 4D (157,189,136,80) at 1mm in LPS; canonical is 2mm LAS. The
# reorientation is the whole ballgame -- see docs/ATLAS-notes.md and the
# laterality assertions in tests/test_disconnectome.py.
TRACT_ATLAS_PATH = ATLAS_DIR / "tract_atlas.nii.gz"
TRACT_LABELS_PATH = ATLAS_DIR / "tract_labels.csv"
TRACT_PARCEL_MAP_PATH = ATLAS_DIR / "tract_parcel_map.csv"
EXPECTED_N_TRACTS = 80

PARCELLATION_PATH = ATLAS_DIR / "parcellation.nii.gz"
PARCELLATION_LABELS_PATH = ATLAS_DIR / "parcellation_labels.csv"

# The location baseline needs its own, coverage-complete label volume. The
# parcellation above is the disconnection NODE SET and stays grey-matter only;
# reusing it for the baseline left white-matter lesions with an all-zero feature
# vector (see docs/BUGS.md B1).
LOCATION_ATLAS_PATH = ATLAS_DIR / "location_atlas.nii.gz"
LOCATION_ATLAS_LABELS_PATH = ATLAS_DIR / "location_atlas_labels.csv"

# --- canonical grid: FSL MNI152 2mm --------------------------------------
CANONICAL_SHAPE: tuple[int, int, int] = (91, 109, 91)
CANONICAL_AFFINE: np.ndarray = np.array(
    [
        [-2.0, 0.0, 0.0, 90.0],
        [0.0, 2.0, 0.0, -126.0],
        [0.0, 0.0, 2.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
# Read-only on purpose. Grid.canonical() hands this array out, and `frozen=True`
# on the dataclass protects the binding, not the array it points at -- without
# this, one in-place write corrupts the canonical grid for the whole process.
CANONICAL_AFFINE.flags.writeable = False

# --- atlas specs ----------------------------------------------------------
SCHAEFER_N_ROIS = 400
SCHAEFER_YEO_NETWORKS = 7
SCHAEFER_RESOLUTION_MM = 2

HARVARD_OXFORD_SUB_ATLAS = "sub-maxprob-thr25-2mm"

# Juelich carries real fibre-tract labels, which is what Schaefer + Harvard-Oxford
# lack. Its tracts are bilateral in a single label, so they are split by
# hemisphere -- a left vs right corticospinal lesion must not look identical.
JUELICH_ATLAS = "maxprob-thr25-2mm"
JUELICH_WM_PREFIX = "wm"

# Neither Schaefer, Harvard-Oxford nor Juelich labels the cerebellum, which left
# a second blind spot of the same kind as B1. AAL3 ships cerebellar and vermis
# parcels already split by hemisphere, on this exact grid.
AAL_CEREBELLAR_KEYS = ("cerebel", "vermis")

# Harvard-Oxford's subcortical atlas ships tissue-class entries alongside real
# structures. These are not parcels and must be dropped, or they would appear as
# enormous "regions" that swallow most of a lesion's overlap.
HO_DROP_PATTERNS = ("cerebral white matter", "cerebral cortex", "lateral ventricle")

# Cortical parcels occupy 1..400; subcortical are appended above them.
SUBCORTICAL_LABEL_OFFSET = SCHAEFER_N_ROIS

# Predicted N. Verified at build time against the real label inventory; the
# emitted CSV is the source of truth, not this number.
EXPECTED_N_PARCELS = 415


def ensure_dirs() -> None:
    """Create the on-disk layout the modules write into."""
    for d in (DATA_DIR, RAW_DIR, ATLAS_DIR, DERIVED_DIR, PRIVATE_DIR,
              COHORT_DIR, LESION_CACHE_DIR, DISCONNECTOME_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
