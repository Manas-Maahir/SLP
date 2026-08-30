"""M3a, done properly: a disconnection matrix from real streamline endpoints.

`disconnectome.disconnection_matrix` has to approximate which parcels a tract
joins, because HCP-842 ships tract *volumes* and a volume has no endpoints. This
module does not approximate that step. It reads an actual tractogram --
479,457 streamlines, each a polyline with two ends -- assigns each end to a
parcel, and counts what fraction of every parcel-pair's fibres the lesion cuts.

That is the object ARCHITECTURE.md M3a specifies, and it removes three of the
limits in docs/DISCONNECTION.md (voxel-overlap standing in for streamline
severance, abutment standing in for termination, and `max` discarding convergent
damage).

**It is not strictly better, and it does not replace the other one.** The two
rest on different references:

    HCP-842   population-averaged over 842 subjects, properly normative,
              but only bundle shapes -> the parcel mapping is inferred
    S35       one subject's tractogram -> exact endpoints, but a single
              individual's anatomy and tractography artifacts standing in
              for "average wiring"

So both matrices are emitted and the comparison between them is itself evidence.
Neither is the truth.

The tractogram sits on `(182, 218, 182)` 1mm LAS -- the FSL MNI152 1mm grid,
which is the canonical 2mm grid at double resolution and the same world box, so
no registration is involved. `assert_canonical` still guards the parcellation and
the lesion, and the tractogram's own affine is checked against that world box.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from . import config
from .spaces import Grid, assert_canonical

DEFAULT_TRACTOGRAM = Path(
    r"C:/Users/manas/Desktop/Projects/Cophe-NeuroVue-1-bug-small-fixes"
    r"/Cophe-NeuroVue-1-bug-small-fixes/backend/tracts/S35_1mm.trk"
)
TRACTOGRAM_NAME = "S35_1mm"

# How far from a streamline's end we will look for a parcel.
#
# Tractography terminates at the grey/white boundary, not inside the cortical
# ribbon, so a strict lookup assigns only half the endpoints. Measured over all
# 479,457 streamlines:
#
#   radius   endpoints assigned   streamlines with both ends assigned
#     0mm          50.8%                    22.3%
#     2mm          61.5%                    35.2%
#     4mm          73.4%                    53.4%
#     6mm          85.2%                    74.3%
#    10mm          94.9%                    92.5%
#
# The curve is smooth -- there is no elbow to read a threshold off, so this is a
# judgement call and is recorded as one. 4mm is two canonical voxels, comparable
# to cortical thickness, and matches the radial-search default that MRtrix's
# tck2connectome uses for the same problem. Beyond ~6mm an endpoint can cross a
# sulcus into a neighbouring gyrus, which is a different parcel and a wrong edge.
ENDPOINT_SEARCH_MM = 4.0

# Streamline points are ~2mm apart, the same size as a canonical voxel, so
# consecutive samples can step diagonally past a voxel the fibre really passes
# through. Points are densified to this spacing before voxelisation so a lesion
# cannot fall between two samples.
DENSIFY_MM = 1.0

STREAMLINE_INDEX_PATH = config.ATLAS_DIR / "streamline_index.npz"

_CHUNK = 20_000


@dataclass
class StreamlineIndex:
    """Everything about the tractogram that does not depend on the lesion.

    Built once and cached, because it takes minutes and never changes.

    `edge_id` is `min(a,b) * n_parcels + max(a,b)` over 1-based parcel indices,
    so an edge is orientation-free. Streamlines without both ends assigned carry
    `edge_id = -1` and take no part in the matrix, though they are still counted
    in `n_streamlines_cut` so the report can say how many were dropped.
    """

    n_parcels: int
    edge_id: np.ndarray          # (n_streamlines,) int64, -1 if unassigned
    voxel_offsets: np.ndarray    # (n_voxels + 1,) int64, CSR into voxel_streams
    voxel_streams: np.ndarray    # (n_pairs,) int32 -- which fibres cross a voxel
    edge_totals: np.ndarray      # (n_edges,) int32 -- fibres per edge
    shape: tuple[int, int, int]
    search_mm: float
    source: str

    @property
    def n_streamlines(self) -> int:
        return len(self.edge_id)

    @property
    def n_assigned(self) -> int:
        return int((self.edge_id >= 0).sum())


def _assert_tractogram_space(header, grid: Grid) -> None:
    """The tractogram must describe the same world box as the canonical grid.

    A tractogram in a different template is the streamline version of the bug
    `spaces.py` exists to prevent: the fibres would still be plausible polylines,
    just in the wrong brain.
    """
    dims = np.asarray(header.get("dimensions"), dtype=float)
    sizes = np.asarray(header.get("voxel_sizes"), dtype=float)
    extent = dims * sizes
    canon_extent = np.asarray(grid.shape, dtype=float) * np.asarray(grid.voxel_sizes)
    if not np.allclose(extent, canon_extent, atol=2.0):
        raise AssertionError(
            "tractogram does not share the canonical world box.\n"
            f"  tractogram: {dims.astype(int).tolist()} x {sizes.tolist()}mm "
            f"= {extent.tolist()}mm\n"
            f"  canonical : {list(grid.shape)} x {list(grid.voxel_sizes)}mm "
            f"= {canon_extent.tolist()}mm\n"
            "  Resample or reject it; do not assume the coordinates line up."
        )


def _densify(points: np.ndarray, step_mm: float = DENSIFY_MM) -> np.ndarray:
    """Linearly interpolate a polyline so consecutive points are <= step apart."""
    if len(points) < 2:
        return points
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    n_sub = np.maximum(1, np.ceil(seg / step_mm).astype(int))
    out = [points[:1]]
    for i, n in enumerate(n_sub):
        t = np.linspace(0.0, 1.0, n + 1)[1:][:, None]
        out.append(points[i] + t * (points[i + 1] - points[i]))
    return np.concatenate(out, axis=0)


def build_streamline_index(
    tractogram: str | Path = DEFAULT_TRACTOGRAM,
    parc_img=None,
    parc_labels: pd.DataFrame | None = None,
    search_mm: float = ENDPOINT_SEARCH_MM,
    verbose: bool = True,
) -> StreamlineIndex:
    """Read the tractogram once and reduce it to what the matrix needs."""
    from scipy import ndimage

    from .atlases import load_parcellation

    tractogram = Path(tractogram)
    if not tractogram.exists():
        raise FileNotFoundError(
            f"{tractogram} not found. Pass --tractogram, or set it to the .trk "
            "you want to use as the structural reference."
        )
    if parc_img is None:
        parc_img, parc_labels = load_parcellation()
    assert_canonical(parc_img, "parcellation")

    grid = Grid.canonical()
    parc = np.asarray(parc_img.dataobj).astype(np.int32)
    n_parcels = int(parc_labels["index"].max())

    trk = nib.streamlines.load(str(tractogram))
    _assert_tractogram_space(trk.header, grid)
    lines = trk.streamlines
    n = len(lines)
    if verbose:
        print(f"  tractogram : {tractogram.name}  {n:,} streamlines")

    # Nearest parcel, and its distance, for every voxel -- one exact pass rather
    # than a per-endpoint neighbourhood search.
    distance, indices = ndimage.distance_transform_edt(
        parc == 0, sampling=grid.voxel_sizes, return_indices=True
    )
    nearest = parc[indices[0], indices[1], indices[2]]

    world_to_vox = np.linalg.inv(np.asarray(grid.affine, dtype=float))
    shape = np.asarray(grid.shape)

    def to_voxel(points: np.ndarray) -> np.ndarray:
        ijk = np.rint(points @ world_to_vox[:3, :3].T + world_to_vox[:3, 3])
        return np.clip(ijk, 0, shape - 1).astype(np.int32)

    # --- endpoints -> edge ---
    ends = np.empty((n, 2, 3), dtype=np.float32)
    for i in range(n):
        line = lines[i]
        ends[i, 0], ends[i, 1] = line[0], line[-1]
    eijk = to_voxel(ends.reshape(-1, 3))
    d = distance[eijk[:, 0], eijk[:, 1], eijk[:, 2]]
    lab = np.where(d <= search_mm, nearest[eijk[:, 0], eijk[:, 1], eijk[:, 2]], 0)
    lab = lab.reshape(n, 2).astype(np.int64)

    a, b = np.minimum(lab[:, 0], lab[:, 1]), np.maximum(lab[:, 0], lab[:, 1])
    assigned = (a > 0) & (b > 0) & (a != b)  # self-edges are not connections
    edge_id = np.where(assigned, a * (n_parcels + 1) + b, -1)

    n_edges = (n_parcels + 1) * (n_parcels + 1)
    edge_totals = np.bincount(
        edge_id[assigned], minlength=n_edges
    ).astype(np.int32)

    if verbose:
        print(f"  endpoints assigned within {search_mm:g}mm : {100 * (lab > 0).mean():.1f}%")
        print(f"  streamlines with both ends assigned  : "
              f"{100 * assigned.mean():.1f}%  ({int(assigned.sum()):,})")
        print(f"  distinct parcel pairs connected      : "
              f"{int((edge_totals > 0).sum()):,}")

    # --- which voxels each streamline crosses (CSR: voxel -> streamlines) ---
    vox_parts, stream_parts = [], []
    for start in range(0, n, _CHUNK):
        stop = min(start + _CHUNK, n)
        chunk_vox, chunk_sid = [], []
        for i in range(start, stop):
            ijk = to_voxel(_densify(np.asarray(lines[i], dtype=np.float32)))
            flat = np.unique(np.ravel_multi_index(ijk.T, grid.shape))
            chunk_vox.append(flat)
            chunk_sid.append(np.full(len(flat), i, dtype=np.int32))
        vox_parts.append(np.concatenate(chunk_vox).astype(np.int32))
        stream_parts.append(np.concatenate(chunk_sid))
        if verbose and (stop % 100_000 == 0 or stop == n):
            print(f"    voxelised {stop:,}/{n:,}")

    vox = np.concatenate(vox_parts)
    sid = np.concatenate(stream_parts)
    del vox_parts, stream_parts

    order = np.argsort(vox, kind="stable")
    vox, sid = vox[order], sid[order]
    n_voxels = int(np.prod(grid.shape))
    voxel_offsets = np.zeros(n_voxels + 1, dtype=np.int64)
    np.cumsum(np.bincount(vox, minlength=n_voxels), out=voxel_offsets[1:])

    if verbose:
        print(f"  streamline-voxel pairs : {len(sid):,} "
              f"({sid.nbytes / 1e6:.0f} MB)")

    return StreamlineIndex(
        n_parcels=n_parcels, edge_id=edge_id, voxel_offsets=voxel_offsets,
        voxel_streams=sid, edge_totals=edge_totals, shape=tuple(grid.shape),
        search_mm=float(search_mm), source=tractogram.name,
    )


def save_streamline_index(index: StreamlineIndex, path: Path | None = None) -> Path:
    path = Path(path or STREAMLINE_INDEX_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        n_parcels=index.n_parcels, edge_id=index.edge_id,
        voxel_offsets=index.voxel_offsets, voxel_streams=index.voxel_streams,
        edge_totals=index.edge_totals, shape=np.asarray(index.shape),
        search_mm=index.search_mm, source=index.source,
    )
    return path


def load_streamline_index(path: Path | None = None) -> StreamlineIndex:
    path = Path(path or STREAMLINE_INDEX_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/15_build_streamline_index.py first"
        )
    z = np.load(path, allow_pickle=False)
    return StreamlineIndex(
        n_parcels=int(z["n_parcels"]), edge_id=z["edge_id"],
        voxel_offsets=z["voxel_offsets"], voxel_streams=z["voxel_streams"],
        edge_totals=z["edge_totals"], shape=tuple(int(v) for v in z["shape"]),
        search_mm=float(z["search_mm"]), source=str(z["source"]),
    )


def severed_streamlines(lesion_img, index: StreamlineIndex) -> np.ndarray:
    """Ids of every streamline passing through the lesion."""
    assert_canonical(lesion_img, "lesion")
    mask = np.asarray(lesion_img.dataobj) > 0
    if not mask.any():
        return np.empty(0, dtype=np.int32)

    voxels = np.flatnonzero(mask.reshape(-1))
    starts, stops = index.voxel_offsets[voxels], index.voxel_offsets[voxels + 1]
    keep = stops > starts
    if not keep.any():
        return np.empty(0, dtype=np.int32)
    hits = [index.voxel_streams[s:e] for s, e in zip(starts[keep], stops[keep], strict=True)]
    return np.unique(np.concatenate(hits))


def streamline_disconnection_matrix(
    lesion_img, index: StreamlineIndex | None = None
) -> np.ndarray:
    """N x N: the fraction of each parcel pair's streamlines that the lesion cuts.

    Unlike `disconnectome.disconnection_matrix`, every entry here is a counted
    ratio rather than a weighted inference -- entry (i, j) is
    `fibres of edge i-j through the lesion / fibres of edge i-j`.
    """
    if index is None:
        index = load_streamline_index()

    n = index.n_parcels
    matrix = np.zeros((n, n), dtype=np.float32)
    cut = severed_streamlines(lesion_img, index)
    if cut.size == 0:
        return matrix

    edges = index.edge_id[cut]
    edges = edges[edges >= 0]
    if edges.size == 0:
        return matrix

    counts = np.bincount(edges, minlength=len(index.edge_totals)).astype(np.float32)
    live = np.flatnonzero(counts > 0)
    fraction = counts[live] / np.maximum(index.edge_totals[live], 1)

    rows, cols = np.divmod(live, n + 1)          # 1-based parcel indices
    ok = (rows >= 1) & (rows <= n) & (cols >= 1) & (cols <= n)
    rows, cols, fraction = rows[ok] - 1, cols[ok] - 1, fraction[ok]

    matrix[rows, cols] = fraction
    matrix[cols, rows] = fraction
    np.fill_diagonal(matrix, 0.0)
    return matrix


def summary(lesion_img, index: StreamlineIndex | None = None) -> dict:
    """Counts for the report: how many fibres were cut, and how many were usable."""
    if index is None:
        index = load_streamline_index()
    cut = severed_streamlines(lesion_img, index)
    edges = index.edge_id[cut] if cut.size else np.empty(0, dtype=np.int64)
    return {
        "tractogram": index.source,
        "endpoint_search_mm": index.search_mm,
        "n_streamlines_total": index.n_streamlines,
        "n_streamlines_cut": int(cut.size),
        "n_streamlines_cut_with_both_ends_assigned": int((edges >= 0).sum()),
        "n_edges_affected": len(np.unique(edges[edges >= 0])) if edges.size else 0,
    }


# A parcel pair needs this many fibres behind it before its severed fraction is
# quoted anywhere a human reads. Measured over the 6,575 connected edges:
#
#   >=  1 fibre  : 6,575 edges (100.0%)      >= 10 : 3,152 (47.9%)
#   >=  2 fibres : 5,434 edges ( 82.6%)      >= 20 : 2,252 (34.3%)
#   >=  5 fibres : 4,121 edges ( 62.7%)      >= 50 : 1,260 (19.2%)
#
# Median support is 9 fibres, so an unfiltered ranking is dominated by edges
# carried by one or two streamlines, which read "100% severed" the moment a
# lesion clips them. That is the same failure as ranking tracts by frac_severed
# (docs/DISCONNECTION.md limit 7): a normalised score with a tiny denominator.
#
# Filtering to >=10 turns the top of a left corona radiata ranking into
# LH_SomMot_37 <-> Brain-Stem at 100% on 963 fibres -- the corticospinal tract,
# which is the correct answer and the one the independent HCP-842 matrix also
# gives. The matrix ITSELF is never filtered; M4 gets everything and decides for
# itself. This threshold governs reporting only.
MIN_EDGE_STREAMLINES = 10


def edge_support_matrix(index: StreamlineIndex | None = None) -> np.ndarray:
    """N x N fibre counts per parcel pair -- the denominator behind every ratio.

    A severed fraction is meaningless without it: 1.0 on 963 fibres and 1.0 on a
    single fibre are the same number and completely different findings.
    """
    if index is None:
        index = load_streamline_index()
    n = index.n_parcels
    support = np.zeros((n, n), dtype=np.int32)
    live = np.flatnonzero(index.edge_totals > 0)
    rows, cols = np.divmod(live, n + 1)
    ok = (rows >= 1) & (rows <= n) & (cols >= 1) & (cols <= n)
    rows, cols = rows[ok] - 1, cols[ok] - 1
    totals = index.edge_totals[live][ok]
    support[rows, cols] = totals
    support[cols, rows] = totals
    return support


def top_disconnections(
    matrix: np.ndarray,
    parc_labels: pd.DataFrame,
    support: np.ndarray | None = None,
    min_streamlines: int = MIN_EDGE_STREAMLINES,
    top_n: int = 10,
) -> list[dict]:
    """The strongest well-supported disconnections, worst first.

    Edges below `min_streamlines` are excluded from the RANKING, not from the
    matrix. Every row carries the fibre count it rests on so a reader can see
    how much weight it bears.
    """
    if support is None:
        support = edge_support_matrix()
    ranked = np.where(support >= min_streamlines, matrix, 0.0)
    iu = np.triu_indices(matrix.shape[0], k=1)
    values = ranked[iu]
    order = np.argsort(values)[::-1][:top_n]
    name_of = dict(zip(parc_labels["index"], parc_labels["name"], strict=True))
    return [
        {
            "parcel_a": name_of.get(int(iu[0][o]) + 1, f"parcel {iu[0][o] + 1}"),
            "parcel_b": name_of.get(int(iu[1][o]) + 1, f"parcel {iu[1][o] + 1}"),
            "value": float(values[o]),
            "n_streamlines": int(support[iu[0][o], iu[1][o]]),
        }
        for o in order
        if values[o] > 0
    ]
