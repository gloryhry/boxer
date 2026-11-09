"""Room clustering on the 2D PF map (paper section 3.3)."""
import warnings

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import HDBSCAN


def histogram_thresholds(pf_vals, bins=32):
    """(thr_low, thr_high): PF values of the lowest and highest histogram density peaks."""
    counts, edges = np.histogram(pf_vals, bins)
    mids = (edges[:-1] + edges[1:]) / 2
    peaks, _ = find_peaks(counts)
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(counts))])
    return mids[peaks[0]], mids[peaks[-1]]


def distance_matrix(cells, pf_vals, vis, w=(0.3, 0.6, 0.1)):
    """Pairwise distance D per Eq. (2): weighted visibility + euclidean + PF terms."""
    v = vis.astype(np.float32)
    hamming = v @ (1 - v).T + (1 - v) @ v.T
    d_vis = hamming / np.maximum(v.sum(1)[:, None] + v.sum(1)[None, :], 1)  # Eq. (3)
    d_eucl = squareform(pdist(cells.astype(float)))
    d_pf = np.abs(pf_vals[:, None] - pf_vals[None, :])
    return (w[0] * d_vis
            + w[1] * d_eucl / max(d_eucl.max(), 1e-9)
            + w[2] * d_pf / max(d_pf.max(), 1e-9))


def cluster_rooms(pf_vals, D, thr_low, thr_high, min_cluster_size=8, new_cluster_dist=0.3):
    """HDBSCAN on high-PF seed cells, then nearest-cluster assignment of the rest.

    Returns per-cell labels; -1 = below thr_low (unassigned)."""
    labels = np.full(len(pf_vals), -1)
    seeds = pf_vals >= thr_high
    if seeds.sum() >= min_cluster_size:
        model = HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed")
        with warnings.catch_warnings():  # sklearn's `copy` deprecation; D subset is a copy already
            warnings.simplefilter("ignore", FutureWarning)
            labels[seeds] = model.fit_predict(D[np.ix_(seeds, seeds)].astype(np.float64))

    rest = np.where((pf_vals >= thr_low) & (labels < 0))[0]
    nxt = labels.max() + 1
    for i in rest[np.argsort(-pf_vals[rest])]:  # strongest cells first
        assigned = labels >= 0
        d = D[i, assigned]
        if d.size and d.min() <= new_cluster_dist:
            labels[i] = labels[assigned][d.argmin()]
        else:
            labels[i] = nxt
            nxt += 1
    return labels


def _xy_diameter(cells, block_size):
    """Exact Euclidean normalization without an all-cells distance matrix.

    A farthest pair lies on the convex hull. Degenerate or numerically nearly
    collinear inputs use the same bounded-distance blocks as a safe fallback.
    Large hulls are blocked too, so a circular input cannot allocate O(N²).
    """
    unique = np.unique(cells, axis=0)
    if len(unique) < 2:
        return 0.
    if len(unique) > 2:
        try:
            boundary = unique[ConvexHull(unique).vertices]
        except QhullError:
            boundary = unique
    else:
        boundary = unique
    if len(boundary) <= 2048:
        return float(pdist(boundary).max(initial=0.))
    maximum = 0.
    for start in range(0, len(boundary), block_size):
        maximum = max(maximum, float(cdist(boundary[start:start + block_size], boundary).max(initial=0.)))
    return maximum


class _RoomDistanceBlocks:
    """The original metric, evaluated for bounded rows and optional columns."""
    def __init__(self, cells, pf_vals, vis, block_size):
        self.cells = cells
        self.pf_vals = pf_vals
        self.vis = np.asarray(vis, dtype=np.float32)
        self.visibility_counts = self.vis.sum(axis=1)
        self.diameter = max(_xy_diameter(cells, block_size), 1e-9)
        self.pf_range = max(float(np.ptp(pf_vals)), 1e-9)
        self.max_block_bytes = 0

    def rows(self, row_indices, column_indices=None):
        other_cells = self.cells if column_indices is None else self.cells[column_indices]
        other_pf = self.pf_vals if column_indices is None else self.pf_vals[column_indices]
        other_vis = self.vis if column_indices is None else self.vis[column_indices]
        other_counts = self.visibility_counts if column_indices is None else self.visibility_counts[column_indices]
        distances = cdist(self.cells[row_indices], other_cells)
        distances *= .6
        distances /= self.diameter
        counts = self.visibility_counts[row_indices, None] + other_counts[None, :]
        visibility = self.vis[row_indices] @ other_vis.T
        visibility *= -2
        visibility += counts
        visibility /= np.maximum(counts, 1)
        visibility *= .3  # Keep float32 rounding of the original weighted term.
        distances += visibility
        del counts, visibility
        potential = np.abs(self.pf_vals[row_indices, None] - other_pf[None, :])
        potential *= .1
        potential /= self.pf_range
        distances += potential
        self.max_block_bytes = max(self.max_block_bytes, distances.nbytes)
        return distances


def cluster_rooms_streaming(cells, pf_vals, vis, thr_low, thr_high,
                             min_cluster_size=8, new_cluster_dist=.3, block_size=128,
                             max_seed_matrix_bytes=512 * 1024**2):
    """Run the original clustering without storing all stack-pair distances.

    Only high-PF seeds need a dense HDBSCAN matrix. Remaining rows are computed
    in blocks against all cells and consumed in the original descending-PF
    order. Labels are updated after every row, including within one block.
    Thus earlier assignments and nearest-label tie ordering stay unchanged.

    The seed budget is checked before allocation. Diagnostics report array
    sizes, not an estimate of the full process RSS or HDBSCAN's work buffers.
    """
    cells = np.asarray(cells, dtype=np.float64)
    pf_vals = np.asarray(pf_vals, dtype=np.float64)
    vis = np.asarray(vis)
    if pf_vals.ndim != 1:
        raise ValueError("PF values must be one-dimensional")
    n = len(pf_vals)
    if cells.shape != (n, 2) or vis.ndim != 2 or len(vis) != n or pf_vals.ndim != 1:
        raise ValueError("PF cells, values and visibility have incompatible shapes")
    if not np.isfinite(cells).all() or not np.isfinite(pf_vals).all() or not np.isfinite(vis).all():
        raise ValueError("PF clustering inputs must be finite")
    if not np.all((vis == 0) | (vis == 1)):
        raise ValueError("Visibility must contain binary observations")
    if (isinstance(block_size, (bool, np.bool_)) or not isinstance(block_size, (int, np.integer))
            or block_size < 1):
        raise ValueError("Distance block size must be a positive integer")
    if (isinstance(max_seed_matrix_bytes, (bool, np.bool_))
            or not isinstance(max_seed_matrix_bytes, (int, np.integer)) or max_seed_matrix_bytes < 0):
        raise ValueError("Seed distance budget must be nonnegative integer bytes")
    if (isinstance(min_cluster_size, (bool, np.bool_)) or not isinstance(min_cluster_size, (int, np.integer))
            or min_cluster_size < 2):
        raise ValueError("Minimum seed cluster size must be an integer >= 2")
    if not np.isfinite([thr_low, thr_high, new_cluster_dist]).all() or new_cluster_dist < 0:
        raise ValueError("PF thresholds and new-cluster distance must be finite")
    labels = np.full(n, -1, dtype=int)
    seeds = np.flatnonzero(pf_vals >= thr_high)
    seed_bytes = int(len(seeds) ** 2 * np.dtype(np.float64).itemsize) if len(seeds) >= min_cluster_size else 0
    diagnostics = dict(mode="seed_matrix_and_streamed_rows", cell_count=n, seed_count=int(len(seeds)),
                       block_size=int(block_size), max_block_bytes=0, seed_matrix_bytes=seed_bytes,
                       dense_equivalent_bytes=int(n ** 2 * np.dtype(np.float64).itemsize), rest_count=0)
    if not n:
        return labels, diagnostics
    if seed_bytes > max_seed_matrix_bytes:
        raise ValueError(f"High-PF seed matrix needs {seed_bytes:,} bytes, exceeding the "
                         f"{max_seed_matrix_bytes:,}-byte budget; increase voxel size or crop the scene")
    metric = _RoomDistanceBlocks(cells, pf_vals, vis, block_size)
    if seed_bytes:
        seed_distances = np.empty((len(seeds), len(seeds)), dtype=np.float64)
        for start in range(0, len(seeds), block_size):
            seed_distances[start:start + block_size] = metric.rows(seeds[start:start + block_size], seeds)
        model = HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            labels[seeds] = model.fit_predict(seed_distances)
        del seed_distances, model
    rest = np.flatnonzero((pf_vals >= thr_low) & (labels < 0))
    rest = rest[np.argsort(-pf_vals[rest])]
    diagnostics["rest_count"] = int(len(rest))
    nxt = labels.max() + 1
    for start in range(0, len(rest), block_size):
        indices = rest[start:start + block_size]
        distances = metric.rows(indices)
        for index, row in zip(indices, distances):
            assigned = labels >= 0
            d = row[assigned]
            if d.size and d.min() <= new_cluster_dist:
                labels[index] = labels[assigned][d.argmin()]
            else:
                labels[index] = nxt
                nxt += 1
    diagnostics["max_block_bytes"] = int(metric.max_block_bytes)
    diagnostics["diameter_cells"] = float(metric.diameter)
    diagnostics["pf_range"] = float(metric.pf_range)
    return labels, diagnostics
