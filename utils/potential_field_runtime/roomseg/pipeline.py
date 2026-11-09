"""End-to-end room segmentation (paper Fig. 2) with per-stage benchmarks."""
import resource
import sys
import time
from contextlib import contextmanager

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mode

from .cluster import cluster_rooms_streaming, histogram_thresholds
from .field import height_consistent_map, relative_height_map, pf_map, potential_field, visibility
from .grid import voxelize
from .interior import dominant_directions, evidence, remove_clutter, segment_interior
from .refine import refine_rooms

_RSS_DIV = 2**20 if sys.platform == "darwin" else 2**10  # ru_maxrss unit -> MB


@contextmanager
def _timed(name, bench):
    t0 = time.perf_counter()
    yield
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RSS_DIV
    bench.append((name, time.perf_counter() - t0, rss))


def segment(xyz, normals, voxel=0.18, smoothness=0.6, vis_targets=None,
            min_cluster_size=8, new_cluster_dist=0.3, knn=10, rng_seed=0,
            min_room_area=2.0, merge_opening=1.6, max_projection_distance=0.8,
            ceiling_aware=False, structure_aware=False, boundary_review=False,
            furniture_height=1.8, wall_min_length=1.2, merge_confidence=.85):
    """Segment a Z-up point cloud into rooms.

    Returns (labels, art): per-point room labels (-1 = unassigned) and a dict of
    intermediate artifacts including per-stage (name, seconds, peak-RSS-MB)."""
    bench = []
    if not isinstance(boundary_review, (bool, np.bool_)):
        raise ValueError("Boundary review must be Boolean")
    for value, low, high in ((furniture_height, .8, 2.2), (wall_min_length, .6, 3.), (merge_confidence, .6, .99)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or not low <= value <= high):
            raise ValueError("Invalid structural-boundary parameters")
    review_enabled = bool(boundary_review and structure_aware)
    with _timed("voxelize", bench):
        grid = voxelize(xyz, voxel)
        observed_busy = grid.busy
        if not structure_aware:
            grid.busy = remove_clutter(grid.busy)
    heights, structure, boundary_evidence = None, None, None
    if structure_aware:
        from .heights import estimate_local_heights
        from .structure import build_structure
        with _timed("local floor and ceiling", bench):
            heights = estimate_local_heights(grid)
        if review_enabled:
            from .boundary_evidence import analyze_structure
            with _timed("structural surface evidence", bench):
                boundary_evidence = analyze_structure(grid, heights, furniture_height=furniture_height,
                                                      wall_min_length=wall_min_length)
        with _timed("conservative furniture weakening", bench):
            structure = (build_structure(grid, heights, evidence=boundary_evidence, furniture_height=furniture_height)
                         if review_enabled else build_structure(grid, heights))
            grid = structure["grid"]
    with _timed("interior MRF", bench):
        E = evidence(grid.busy, dominant_directions(normals))
        interior = segment_interior(E, grid.busy, smoothness)
        del E
    with _timed("potential field", bench):
        pf = potential_field(grid, interior)
        if structure_aware:
            if ceiling_aware:
                fallback_values, fallback_z, fallback_info = height_consistent_map(grid, interior, pf)
            else:
                fallback_values, fallback_z = pf_map(pf, interior)
                fallback_info = dict(mode="original_highest_stack")
            pf2d, top_z, vertical_sampling = relative_height_map(
                grid, interior, pf, heights, fallback=(fallback_values, fallback_z))
            vertical_sampling["fallback_sampling"] = fallback_info
        elif ceiling_aware:
            pf2d, top_z, vertical_sampling = height_consistent_map(grid, interior, pf)
            vertical_sampling["enabled"] = True
        else:
            pf2d, top_z = pf_map(pf, interior)
            vertical_sampling = dict(enabled=False, mode="original_highest_stack")
    with _timed("visibility", bench):
        cells = np.argwhere(top_z >= 0)
        targets = None
        if vis_targets and vis_targets < len(cells):
            targets = np.random.default_rng(rng_seed).choice(len(cells), vis_targets, False)
        visibility_audit = {}
        if structure_aware:
            vis = visibility(grid, top_z, cells, targets, heights=heights, interior=interior,
                             protected_walls=structure["protected_walls"], audit=visibility_audit, fallback_z=fallback_z)
        else:
            vis = visibility(grid, top_z, cells, targets)
    with _timed("clustering", bench):
        pf_vals = pf2d[tuple(cells.T)]
        thr_low, thr_high = histogram_thresholds(pf_vals)
        cell_labels, distance_computation = cluster_rooms_streaming(cells, pf_vals, vis, thr_low, thr_high,
                                                                   min_cluster_size, new_cluster_dist)
    raw_cell_labels = cell_labels.copy()
    partition_owners = np.zeros(grid.shape[:2], np.int32) if review_enabled else None
    with _timed("room refinement", bench):
        rejected = np.zeros(len(cells), bool)
        cell_labels, refinement = refine_rooms(grid, interior, cells, cell_labels,
                                               min_room_area, merge_opening, rejected,
                                               heights=heights,
                                               protected_walls=structure["protected_walls"] if structure is not None else None,
                                               partition_owners_out=partition_owners)
    before_review_labels = cell_labels.copy()
    boundary_info = dict(enabled=False, reason="disabled" if not boundary_review else "requires_structure_aware")
    if review_enabled:
        from .boundary_review import review_boundaries
        boundary_evidence["accepted_partitions"] = partition_owners
        with _timed("review local segmentation boundaries", bench):
            cell_labels, boundary_info = review_boundaries(grid, interior, cells, cell_labels,
                heights=heights, structure=structure, evidence=boundary_evidence,
                merge_confidence=merge_confidence, merge_opening=merge_opening)
    with _timed("back-projection", bench):
        labels = _backproject(grid, interior, cells, cell_labels, xyz, knn,
                              max_projection_distance, cells[rejected])

    art = dict(grid=grid, interior=interior, pf=pf, pf2d=pf2d, cells=cells,
               cell_labels=cell_labels, thresholds=(thr_low, thr_high),
               raw_cell_labels=raw_cell_labels, refinement=refinement,
               vertical_sampling=vertical_sampling, distance_computation=distance_computation,
               sample_z=top_z,
               projection_rejected=rejected,
               observed_busy=observed_busy, local_heights=heights, structure=structure,
               multilevel_visibility=visibility_audit.get("diagnostics", dict(enabled=False)),
               visibility_layers=visibility_audit.get("sample_z"),
               visibility_fallback_z=fallback_z if structure_aware else None,
               boundary_evidence=boundary_evidence, boundary_review=boundary_info,
               pre_review_cell_labels=before_review_labels, accepted_partitions=partition_owners,
               n_rooms=len(np.unique(cell_labels[cell_labels >= 0])), bench=bench)
    return labels, art


def _backproject(grid, interior, cells, cell_labels, xyz, knn,
                 max_distance=0.8, rejected_cells=None):
    """Stack propagation, then distance-bounded kNN vote (0 = legacy unlimited).

    Distances are between voxel centers in metres. Only neighbors inside the
    radius vote; a single nearby neighbor cannot import distant room votes.
    Rejected room columns stay unassigned instead of undoing area filtering.
    """
    lab3 = np.full(grid.shape, -1)
    for (x, y), lab in zip(cells, cell_labels):
        if lab >= 0:
            lab3[x, y, interior[x, y]] = lab
    labeled = np.argwhere(lab3 >= 0)

    pv = grid.index_of(xyz)
    uniq, inv = np.unique(pv, axis=0, return_inverse=True)
    need = lab3[tuple(uniq.T)] < 0
    rejected = np.zeros(grid.shape[:2], bool)
    if rejected_cells is not None and len(rejected_cells):
        rejected[tuple(rejected_cells.T)] = True
        need &= ~rejected[tuple(uniq[:, :2].T)]
    if max_distance > 0 and need.any():
        # Keep a point with its own room stack when nearby vertical support
        # exists. Euclidean neighbors across a thin wall must not outvote it.
        columns = np.full(grid.shape[:2], -1, dtype=int)
        columns[tuple(cells.T)] = cell_labels
        same_column = np.flatnonzero(need & (columns[tuple(uniq[:, :2].T)] >= 0))
        levels = np.arange(grid.shape[2])
        for start in range(0, len(same_column), 16384):
            idx = same_column[start:start + 16384]
            q = uniq[idx]
            supported = interior[q[:, 0], q[:, 1]]
            vertical = np.where(supported, np.abs(q[:, 2, None] - levels), grid.shape[2]).min(1)
            idx = idx[vertical * grid.voxel <= max_distance]
            lab3[tuple(uniq[idx].T)] = columns[tuple(uniq[idx, :2].T)]
            need[idx] = False
    if need.any() and len(labeled):
        k = min(knn, len(labeled))
        distance, nn = cKDTree(labeled).query(uniq[need], k=k)
        distance, nn = distance.reshape(-1, k), nn.reshape(-1, k)
        votes = lab3[tuple(labeled[nn].transpose(2, 0, 1))]
        if max_distance <= 0:
            winners = mode(votes, axis=1).mode
        else:
            valid = distance * grid.voxel <= max_distance
            winners = np.full(len(votes), -1, dtype=int)
            for start in range(0, len(votes), 16384):
                v, ok = votes[start:start + 16384], valid[start:start + 16384]
                counts = ((v[:, :, None] == v[:, None, :]) & ok[:, None, :]).sum(2)
                counts[~ok] = -1
                best = counts.argmax(1)
                winner = v[np.arange(len(v)), best]
                winners[start:start + len(v)] = np.where(ok.any(1), winner, -1)
        lab3[tuple(uniq[need].T)] = winners
    return lab3[tuple(uniq.T)][inv]
