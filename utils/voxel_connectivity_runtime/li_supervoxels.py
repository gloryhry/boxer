"""Geometry-based initial partitioning after Li, Shi and Sun (2025), §III-A.

https://doi.org/10.1109/JSTARS.2025.3586490

This implementation uses curvature-seeded local assignment and connected
components in place of the paper's fusion–exchange optimizer. All distances
and morphological thresholds are in metres.
"""

import math

import numpy as np
from scipy.ndimage import label, maximum_filter, minimum_filter
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


_BATCH = 32_768
_MAX_POINTS = 8_000_000
_MAX_PIXELS = 16_000_000
_NEIGHBORS = 21
_ANCHOR_CANDIDATES = 12
_MAX_DOOR_NEIGHBORS = 24
_CONNECTED_3D = np.ones((3, 3, 3), dtype=bool)


def _point_geometry(points):
    """Estimate unoriented normals, curvature and the Eq. (2) inlier mask."""
    tree = cKDTree(points)
    normal = np.empty(points.shape, np.float32)
    curvature = np.empty(len(points), np.float32)
    neighbor_distance = np.empty(len(points), np.float32)
    neighbors = min(_NEIGHBORS + 1, len(points))
    for start in range(0, len(points), _BATCH):
        block = slice(start, start + _BATCH)
        distances, indices = tree.query(points[block], k=neighbors, workers=4)
        neighbor_distance[block] = distances[:, 1:].mean(axis=1)
        near = points[indices[:, 1:]]
        near -= near.mean(axis=1, keepdims=True)
        values, axes = np.linalg.eigh(np.swapaxes(near, 1, 2) @ near)
        normal[block] = axes[:, :, 0]
        curvature[block] = np.maximum(values[:, 0], 0.) / np.maximum(values.sum(axis=1), 1e-20)
    inlier = neighbor_distance <= neighbor_distance.mean() + 3 * neighbor_distance.std() + 1e-9
    return normal, curvature, inlier


def _assign_anchors(points, normal, curvature, seed_size):
    """Select a low-curvature seed per occupied bin and normal family."""
    origin = points.min(axis=0)
    bins = np.empty(points.shape, np.int64)
    centrality = np.empty(len(points))
    for start in range(0, len(points), _BATCH):
        block = slice(start, start + _BATCH)
        normalized = (points[block] - origin) / seed_size
        bins[block] = normalized
        normalized -= bins[block]
        normalized -= .5
        centrality[block] = np.einsum('ni,ni->n', normalized, normalized)
    family = np.argmax(np.abs(normal), axis=1).astype(np.int8)
    order = np.lexsort((centrality, curvature.round(6), family, bins[:, 2], bins[:, 1], bins[:, 0]))
    ordered_family = family[order]
    starts = np.r_[True, ordered_family[1:] != ordered_family[:-1]]
    for axis in range(3):
        ordered_axis = bins[order, axis]
        starts[1:] |= ordered_axis[1:] != ordered_axis[:-1]
    anchors = order[starts]
    del bins, ordered_axis, centrality, order
    return _match_anchors(points, normal, curvature, anchors, seed_size), len(anchors)


def _match_anchors(points, normal, curvature, anchors, seed_size):
    tree = cKDTree(points[anchors])
    assignment = np.empty(len(points), np.int32)
    count = min(_ANCHOR_CANDIDATES, len(anchors))
    # Each point's bin contains a seed within sqrt(3) times the bin side.
    # Bound assignment radius so later connectivity uses only small local boxes.
    maximum_distance = math.sqrt(3.) * seed_size + 1e-9
    for start in range(0, len(points), _BATCH):
        block = slice(start, start + _BATCH)
        distance, candidate = tree.query(points[block], k=list(range(1, count + 1)), workers=4)
        index = anchors[candidate]
        dot = np.einsum('nki,ni->nk', normal[index], normal[block])
        cost = (.6 * distance / seed_size + 1. - np.minimum(1., np.abs(dot))
                + .4 * np.maximum(curvature[index], curvature[block, None]))
        cost[distance > maximum_distance] = np.inf
        assignment[block] = candidate[np.arange(len(candidate)), cost.argmin(axis=1)]
    return assignment


def _connected_supervoxels(points, assignment, seed_size, resolution):
    """Split disconnected anchor assignments and discard sparse components."""
    step = max(resolution, seed_size / 16.)
    origin = points.min(axis=0)
    cells = np.empty(points.shape, np.int64)
    for start in range(0, len(points), _BATCH):
        block = slice(start, start + _BATCH)
        cells[block] = (points[block] - origin) / step
    order = np.argsort(assignment, kind='stable')
    ordered = assignment[order]
    bounds = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1, len(order)]
    enough = np.diff(bounds) >= 4
    labels = np.zeros(len(points), np.int32)
    count = 0
    for start, stop in zip(bounds[:-1][enough], bounds[1:][enough]):
        group = order[start:stop]
        block = cells[group]
        block -= block.min(axis=0)
        occupied = np.zeros(tuple(block.max(axis=0) + 1), dtype=bool)
        index = tuple(block.T)
        occupied[index] = True
        components = label(occupied, structure=_CONNECTED_3D)[0][index]
        size = np.bincount(components)
        retained = size >= 4
        retained[0] = False
        new_ids = np.zeros(len(size), np.int32)
        number = int(retained.sum())
        new_ids[retained] = np.arange(count + 1, count + number + 1)
        labels[group] = new_ids[components]
        count += number
    return labels, count


def _top_projection(points, labels, origin_xy, resolution, shape):
    top = np.full(shape, -np.inf)
    projected = np.zeros(shape, np.int32)
    for start in range(0, len(points), _BATCH):
        block = slice(start, start + _BATCH)
        xy = np.floor((points[block, :2] - origin_xy) / resolution)
        valid = (labels[block] > 0) & (xy >= 0).all(axis=1) & (xy < shape).all(axis=1)
        index = xy[valid].astype(np.intp)
        at = tuple(index.T)
        heights = points[block, 2][valid]
        previous = top[at]
        np.maximum.at(top, at, heights)
        # Only a strictly higher return supersedes a previous batch's label.
        projected[tuple(index[heights > previous].T)] = 0
        highest = heights == top[at]
        np.maximum.at(projected, tuple(index[highest].T), labels[block][valid][highest])
    return projected, top


def _door_columns(projected, top, surviving, floor_z, resolution):
    """Find small lower projections between independent higher regions."""
    x, y = np.nonzero(projected)
    ids = projected[x, y]
    size = np.bincount(ids, minlength=len(surviving))
    height = np.bincount(ids, weights=top[x, y], minlength=len(size)) / np.maximum(size, 1)
    center = np.column_stack([np.bincount(ids, weights=axis, minlength=len(size)) / np.maximum(size, 1)
                              for axis in (x, y)])
    del x, y, ids
    possible = (size >= 2) & (size * resolution ** 2 <= 2.) & (height >= floor_z + 1.2)
    possible[0] = False
    if not possible.any():
        return np.zeros(projected.shape, bool)
    touching = []
    for a, b in ((projected[:-1], projected[1:]), (projected[:, :-1], projected[:, 1:])):
        boundary = (a != b) & (a > 0) & (b > 0)
        left, right = a[boundary], b[boundary]
        code = np.minimum(left, right).astype(np.int64) * len(size) + np.maximum(left, right)
        touching.append(code)
    adjacency, contacts = np.unique(np.concatenate(touching), return_counts=True)
    low, high = np.divmod(adjacency, len(size))
    source, target = np.r_[low, high], np.r_[high, low]
    contacts = np.tile(contacts, 2)
    acceptable = (possible[source] & surviving[target] & (size[target] >= 1.5 * size[source])
                  & (height[target] >= height[source] + .2))
    source, target = source[acceptable], target[acceptable]
    order = np.lexsort((-contacts[acceptable], source))
    source, target = source[order], target[order]
    bounds = np.r_[0, np.flatnonzero(source[1:] != source[:-1]) + 1, len(source)]
    enough = np.diff(bounds) >= 2
    openings = np.zeros(len(size), bool)
    for start, stop in zip(bounds[:-1][enough], bounds[1:][enough]):
        current = source[start]
        neighbors = target[start:min(stop, start + _MAX_DOOR_NEIGHBORS)]
        # Adjacent higher patches can represent a single room. Group them
        # locally before checking for independent patches on opposite sides.
        a, b = neighbors[:, None], neighbors[None, :]
        codes = np.minimum(a, b) * len(size) + np.maximum(a, b)
        positions = np.minimum(np.searchsorted(adjacency, codes), len(adjacency) - 1)
        _, groups = connected_components(adjacency[positions] == codes, directed=False)
        directions = center[neighbors] - center[current]
        openings[current] = bool(np.any((directions @ directions.T < 0.) & (groups[:, None] != groups)))
    return openings[projected]


def _project_assigned(points, assignment, origin_xy, resolution, shape, seed_size, geometry):
    labels, region_count = _connected_supervoxels(points, assignment, seed_size, resolution)
    projected, top = _top_projection(points, labels, origin_xy, resolution, shape)
    # A region survives if it contains a solid 0.2 m square at the
    # working resolution. A one-pixel line disappears at any resolution.
    kernel = max(2, min(16, int(math.ceil(.20 / resolution - 1e-9))))
    observed = projected > 0
    inside = (observed & (maximum_filter(projected, size=kernel, mode='constant') == projected)
              & (minimum_filter(projected, size=kernel, mode='constant') == projected))
    surviving = np.bincount(projected[inside], minlength=region_count + 1) > 0
    eroded = surviving[projected]
    doors = _door_columns(projected, top, surviving, geometry['floor_z'], resolution)
    footprint = eroded & ~doors
    diagnostics = dict(
        input_points=geometry['input_points'], outlier_points=geometry['outlier_points'],
        anchor_count=geometry['anchor_count'], supervoxels=region_count, projected_pixels=int(observed.sum()),
        erosion_kernel_pixels=kernel, eroded_pixels=int((observed & ~eroded).sum()),
        door_pixels=int(doors.sum()), footprint_pixels=int(footprint.sum()),
        initial_components=int(label(footprint)[1]), seed_size_m=float(seed_size),
        assignment='local_curvature_seeded_euclidean_normal_curvature',
    )
    return dict(footprint=footprint, observed_footprint=observed, door_columns=doors,
                door_seed_indices=np.flatnonzero(eroded & doors),
                projected_regions=projected, diagnostics=diagnostics)


def _valid_shape(shape):
    return (len(shape) == 2 and all(isinstance(side, (int, np.integer)) and side >= 1 for side in shape)
            and math.prod(map(int, shape)) <= _MAX_PIXELS)


def initial_partition(xyz, origin_xy, resolution, shape, seed_size=1., stage_log=None, *,
                      fallback_resolution=None):
    """Return XY footprint, projected IDs and virtual door columns.

    Points are a finite, Z-up, metrically downsampled Nx3 array. Grid indices
    are X then Y, obtained by flooring offsets from ``origin_xy``. Region ID
    zero denotes absence of a projected supervoxel. ``observed_footprint``
    precedes morphology and door removal. The unoriented normals stay aligned
    with ``xyz``, including filtered outliers. ``stage_log`` takes one string.
    A coarser resolution enables lazy ``fallback(resolution)`` projection on
    the same origin, reusing point geometry and anchor assignments.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    origin_xy = np.asarray(origin_xy, dtype=np.float64)
    shape = tuple(shape)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 4 or len(xyz) > _MAX_POINTS
            or origin_xy.shape != (2,) or not np.isfinite(origin_xy).all()
            or not np.isfinite(xyz).all() or not _valid_shape(shape) or not np.isfinite(resolution) or resolution <= 0
            or not np.isfinite(seed_size) or seed_size <= 0):
        raise ValueError('Li initial partition requires at most 16,000,000 XY cells and 4–8,000,000 finite XYZ points')
    if fallback_resolution is not None and (not np.isfinite(fallback_resolution)
                                             or fallback_resolution <= resolution):
        raise ValueError('Li fallback resolution must be coarser than the working XY grid')
    if np.any(np.ptp(xyz, axis=0) / min(seed_size, resolution) >= 2. ** 52):
        raise ValueError('Li initial partition dimensions exceed safe voxel indexing')
    if stage_log is not None:
        stage_log('Li 初分割：估计局部法向与曲率')
    normals, curvature, inlier = _point_geometry(xyz)
    if not inlier.all():
        points, normal, curvature = xyz[inlier], normals[inlier], curvature[inlier]
    else:
        points, normal = xyz, normals
    if stage_log is not None:
        stage_log('Li 初分割：生成并投影超体素')
    assignment, anchor_count = _assign_anchors(points, normal, curvature, seed_size)
    del normal, curvature
    geometry = dict(input_points=len(xyz), outlier_points=int((~inlier).sum()), anchor_count=anchor_count,
                    floor_z=float(np.quantile(points[:, 2], .05)))

    def project(at_resolution, at_shape):
        result = _project_assigned(points, assignment, origin_xy, at_resolution, at_shape, seed_size, geometry)
        result.update(normals=normals, inlier_mask=inlier)
        return result

    initial = project(resolution, shape)
    if fallback_resolution is not None:
        def fallback(at_resolution=fallback_resolution):
            if not np.isfinite(at_resolution) or at_resolution <= resolution:
                raise ValueError('Li fallback resolution must be coarser than the working XY grid')
            at_shape = tuple(math.ceil(side * resolution / at_resolution) for side in shape)
            if stage_log is not None:
                stage_log(f'Li 初分割：生成 {at_resolution:.3f} m/px 粗尺度种子')
            return project(at_resolution, at_shape)

        initial['fallback'] = fallback
    return initial
