"""Wall evidence and directional pixel linking from Li et al. (2025), §III-C.

This implementation operates on the sampled cloud and the initial XY
footprint. It uses a bounded height grid and preserves observed wall runs
independently of any gap repair. See https://doi.org/10.1109/JSTARS.2025.3586490.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


_DIRECTIONS = ((1, 0), (1, 1), (0, 1), (-1, 1))
_FREE_HEIGHT = .8  # Li et al. §IV-A use .8 m on S3DIS.
_MAX_HEIGHT_BINS = 64


def _column_evidence(points, normals, ids, scope, resolution):
    """Combine a top-down free run with upper-weighted nonhorizontal hits."""
    z = points[:, 2]
    low, high = np.quantile(z[::max(1, len(z) // 100_000)], [.001, .999])
    z_origin = low - resolution
    z_step = max(resolution, (high - z_origin + resolution) / (_MAX_HEIGHT_BINS - 2))
    levels = np.floor((z - z_origin) / z_step).clip(-1, _MAX_HEIGHT_BINS).astype(np.int8)
    valid = (levels >= 0) & (levels < _MAX_HEIGHT_BINS)

    if normals is None:
        horizontal = np.ones(len(points), bool)
        vertical = horizontal
    else:
        length = np.sqrt(np.einsum("ij,ij->i", normals, normals))
        abs_z = np.abs(normals[:, 2])
        horizontal = (length > 0) & (abs_z >= .85 * length)
        vertical = (length > 0) & (abs_z <= .65 * length)
    votes = np.bincount(levels[valid & horizontal], minlength=_MAX_HEIGHT_BINS)
    if not votes.any():
        votes = np.bincount(levels[valid], minlength=_MAX_HEIGHT_BINS)
    floor_limit = max(1, min(_MAX_HEIGHT_BINS - 1, int((min(.35 * (high - low), 1.2) + resolution) / z_step) + 1))
    floor = int(votes[:floor_limit].argmax())
    upper_start = floor + math.ceil(1.65 / z_step)
    upper_votes = votes[upper_start:]
    if not upper_votes.size or not upper_votes.any():
        return None
    main_ceiling = int(upper_start + upper_votes.argmax())

    # Prefer locally observed ceilings so that changes of roof height do not
    # automatically become boundaries. Ignore sparsely sampled plenum outliers.
    plausible = votes >= .035 * votes[main_ceiling]
    plausible[:upper_start] = False
    roof_points = valid & horizontal & plausible[levels.clip(0, _MAX_HEIGHT_BINS - 1)]
    ceiling = np.full(scope.size, -1, np.int16)
    np.maximum.at(ceiling, ids[roof_points], levels[roof_points])
    ceiling = ceiling.reshape(scope.shape)
    known = ceiling >= upper_start
    if known.any() and not known.all():
        near, nearest = ndimage.distance_transform_edt(~known, return_indices=True)
        ceiling = np.where(near <= .8 / resolution, ceiling[tuple(nearest)], main_ceiling)
    elif not known.any():
        ceiling.fill(main_ceiling)
    ceiling = ndimage.median_filter(ceiling, size=3, mode="nearest")
    local_ceiling = ceiling.ravel()[ids]

    wall_hits = valid & vertical & (levels > floor) & (levels < local_ceiling)
    ids, levels = ids[wall_hits], levels[wall_hits]
    # Eq. (8) prints (hc-hp)/(hc-hf), which weights floor hits more.
    # Follow the accompanying prose and use the complementary weight.
    upper_weight = np.clip(
        (z[wall_hits] - (z_origin + floor * z_step)) / ((local_ceiling[wall_hits] - floor) * z_step),
        0., 1.,
    )
    density = np.bincount(ids, weights=upper_weight, minlength=scope.size).reshape(scope.shape)
    occupied = np.zeros(scope.size, np.uint64)
    np.bitwise_or.at(occupied, ids, np.left_shift(np.uint64(1), levels.astype(np.uint64)))
    occupied = occupied.reshape(scope.shape)

    support = np.zeros(scope.shape, np.uint8)
    bottom = np.full(scope.shape, _MAX_HEIGHT_BINS, np.uint8)
    top = np.zeros(scope.shape, np.uint8)
    free_run = np.zeros(scope.shape, np.uint8)
    deficit = ceiling - floor
    free_height = math.ceil(_FREE_HEIGHT / z_step)
    for level in range(int(ceiling.max()) - 1, floor, -1):
        hits = (occupied & (np.uint64(1) << np.uint64(level))) != 0
        support += hits
        bottom[hits] = level
        top[hits & (top == 0)] = level
        empty = (level < ceiling) & ~hits
        free_run[~empty] = 0
        free_run += empty
        # The first sufficient run encountered from above defines accessible height.
        found = (free_run == free_height) & (deficit == ceiling - floor)
        deficit[found] = ceiling[found] - level - free_height

    # A short elevated lintel can extend a wall run but cannot seed one. Low
    # furniture and a single horizontal layer cannot supply wall support.
    upper = top >= floor + .64 * (ceiling - floor)
    persistent = upper & ((top.astype(np.int16) - bottom + 1) * z_step >= .9)
    persistent &= support >= np.maximum(3, .25 * (ceiling - floor))
    response = np.where(scope & upper & (support >= 2), deficit * z_step * density, 0.)
    positive = response[response > 0]
    if not len(positive):
        return None
    # Limit the leverage of very dense scan overlap, then normalize as Eq. (10).
    cap = float(np.quantile(positive, .99))
    response = np.minimum(response / cap, 1.).astype(np.float32)
    heights = dict(
        vertical_resolution_m=float(z_step),
        floor_z_m=float(z_origin + (floor + .5) * z_step),
        main_ceiling_z_m=float(z_origin + (main_ceiling + .5) * z_step),
        observed_ceiling_columns=int(known.sum()),
        free_height_m=_FREE_HEIGHT,
    )
    return response, persistent, heights


def _directions_and_ridges(evidence):
    """Weighted PCA in a 3×3 neighborhood and perpendicular suppression."""
    x, y = np.mgrid[-1:2, -1:2]
    mass = ndimage.correlate(evidence, np.ones((3, 3)), mode="constant")
    mass = np.maximum(mass, np.finfo(np.float32).eps)
    mean_x = ndimage.correlate(evidence, x, mode="constant") / mass
    mean_y = ndimage.correlate(evidence, y, mode="constant") / mass
    xx = ndimage.correlate(evidence, x * x, mode="constant") / mass - mean_x**2
    yy = ndimage.correlate(evidence, y * y, mode="constant") / mass - mean_y**2
    xy = ndimage.correlate(evidence, x * y, mode="constant") / mass - mean_x * mean_y
    angle = .5 * np.arctan2(2 * xy, xx - yy)
    direction = (np.floor(angle / (np.pi / 4) + .5).astype(np.int8) % 4)
    coherence = np.hypot(xx - yy, 2 * xy) / np.maximum(xx + yy, 1e-6)

    padded = np.pad(evidence, 1)
    nx, ny = evidence.shape
    ridge = np.zeros(evidence.shape, bool)
    prominence = np.zeros(evidence.shape, np.float32)
    for index in range(4):
        dx, dy = _DIRECTIONS[(index + 2) % 4]
        previous = padded[1 - dx:1 - dx + nx, 1 - dy:1 - dy + ny]
        following = padded[1 + dx:1 + dx + nx, 1 + dy:1 + dy + ny]
        selected = (direction == index) & (evidence >= previous) & (evidence >= following)
        ridge |= selected
        prominence[selected] = (evidence - np.minimum(previous, following))[selected]
    return direction, coherence, ridge, prominence


def _overlap(shape, dx, dy):
    source = tuple(slice(max(0, -step), min(n, n - step)) for n, step in zip(shape, (dx, dy)))
    target = tuple(slice(max(0, step), min(n, n + step)) for n, step in zip(shape, (dx, dy)))
    return source, target


def _observed_runs(evidence, persistent, resolution):
    direction, coherence, ridge, prominence = _directions_and_ridges(evidence)
    nonzero = evidence[evidence > 0]
    median = float(np.median(nonzero))
    threshold = median + float(np.median(np.abs(nonzero - median)))
    anchors = ridge & persistent & (coherence >= .15) & (prominence >= threshold)
    candidates = (evidence >= 1 / 255) & (ridge | persistent)
    positions = np.flatnonzero(candidates)
    ids = np.full(evidence.size, -1, np.int32)
    ids[positions] = np.arange(len(positions), dtype=np.int32)
    ids = ids.reshape(evidence.shape)
    starts, ends = [], []
    for index, (dx, dy) in enumerate(_DIRECTIONS):
        source, target = _overlap(evidence.shape, dx, dy)
        # Permit one orientation bin of change to follow curved walls, but
        # do not grow laterally from one line to a parallel line.
        aligned = ((direction - index) % 4 != 2) | ((coherence < .15) & persistent)
        selected = candidates[source] & candidates[target] & aligned[source] & aligned[target]
        starts.append(ids[source][selected])
        ends.append(ids[target][selected])
    rows, cols = np.concatenate(starts), np.concatenate(ends)
    graph = coo_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(len(positions), len(positions))).tocsr()
    n, labels = connected_components(graph, directed=False)
    structural = np.bincount(labels, weights=persistent.ravel()[positions], minlength=n)
    sizes = np.bincount(labels, minlength=n)
    anchor_counts = np.bincount(labels, weights=anchors.ravel()[positions], minlength=n)
    # Protect coherent observed wall runs even if similar evidence values
    # give the entire run zero prominence over its neighbors.
    supported = (structural >= max(2, math.ceil(.35 / resolution))) | (
        (anchor_counts > 0) & (structural >= 2) & (sizes * resolution >= .35)
    )
    observed = np.zeros(evidence.size, bool)
    observed[positions] = supported[labels]
    return observed.reshape(evidence.shape), direction, anchors, threshold


def _wall_endpoints(observed, persistent, direction):
    endpoints = np.zeros(observed.shape, bool)
    for index, (dx, dy) in enumerate(_DIRECTIONS):
        forward = np.zeros(observed.shape, bool)
        backward = np.zeros_like(forward)
        for step in ((dx, dy), _DIRECTIONS[(index - 1) % 4], _DIRECTIONS[(index + 1) % 4]):
            sx, sy = step if dx * step[0] + dy * step[1] > 0 else (-step[0], -step[1])
            source, target = _overlap(observed.shape, sx, sy)
            forward[source] |= observed[target]
            backward[target] |= observed[source]
        endpoints |= (direction == index) & (~forward | ~backward)
    return np.argwhere(endpoints & observed & persistent)


def _wall_bank(cell, towards, observed, evidence, resolution, minimum_span):
    """Fit the observed run behind an endpoint, using weighted TLS."""
    radius = max(2, math.ceil(.65 / resolution))
    begin = np.maximum(cell - radius, 0)
    end = np.minimum(cell + radius + 1, observed.shape)
    region = tuple(slice(a, b) for a, b in zip(begin, end))
    xy = np.argwhere(observed[region]) + begin
    delta = (xy - cell) * resolution
    along = delta @ towards
    across = delta @ np.array([-towards[1], towards[0]])
    support = (along <= .15 * resolution) & (np.abs(across) <= max(.12, .85 * resolution))
    if support.sum() < 2 or np.ptp(along[support]) < minimum_span:
        return None
    delta = delta[support]
    weights = evidence[tuple(xy[support].T)]
    center = np.average(delta, axis=0, weights=weights)
    centered = delta - center
    _, axes = np.linalg.eigh((centered * weights[:, None]).T @ centered)
    tangent = axes[:, 1]
    if abs(tangent @ towards) < math.cos(math.radians(20)):
        return None
    return center, tangent


def _line_pixels(start, end):
    """Integer Bresenham line, with both endpoints."""
    x, y = map(int, start)
    last_x, last_y = map(int, end)
    dx, dy = abs(last_x - x), -abs(last_y - y)
    sx, sy = (1 if last_x > x else -1), (1 if last_y > y else -1)
    error = dx + dy
    result = [(x, y)]
    while (x, y) != (last_x, last_y):
        twice = 2 * error
        if twice >= dy:
            error += dy
            x += sx
        if twice <= dx:
            error += dx
            y += sy
        result.append((x, y))
    return np.asarray(result)


def _link_gaps(observed, persistent, evidence, direction, scope, resolution, max_jump):
    """Prefer short links, then join collinear observed banks within the jump cap."""
    added = np.zeros(observed.shape, bool)
    short_count = jump_count = 0
    if max_jump < 2 * resolution - 1e-9:
        return added, short_count, jump_count
    structural = observed & persistent
    endpoints = _wall_endpoints(observed, persistent, direction)
    if len(endpoints) < 2:
        return added, short_count, jump_count
    # A bounded nearest-neighbor query prevents dense junctions from producing
    # a quadratic number of candidate gaps.
    tree = cKDTree(endpoints * resolution)
    distances, nearest = tree.query(tree.data, k=min(9, len(endpoints)), distance_upper_bound=max_jump + 1e-9)
    source, offset = np.nonzero((distances > resolution * math.sqrt(2) + 1e-9) & np.isfinite(distances))
    pairs = np.unique(np.sort(np.column_stack((source, nearest[source, offset])), axis=1), axis=0)
    lengths = np.linalg.norm(endpoints[pairs[:, 1]] - endpoints[pairs[:, 0]], axis=1) * resolution
    used = np.zeros(len(endpoints), bool)
    short_limit = max(.25, 2.5 * resolution) + 1e-9
    for candidate in np.argsort(lengths, kind="stable"):
        i, j = pairs[candidate]
        if used[i] or used[j]:
            continue
        a, b = endpoints[i], endpoints[j]
        direction_ab = (b - a) * resolution / lengths[candidate]
        short = lengths[candidate] <= short_limit
        minimum_span = max(.23 if short else .42, .95 * resolution)
        banks = (
            _wall_bank(a, direction_ab, structural, evidence, resolution, minimum_span),
            _wall_bank(b, -direction_ab, structural, evidence, resolution, minimum_span),
        )
        if any(bank is None for bank in banks):
            continue
        (center_a, tangent_a), (center_b, tangent_b) = banks
        normal_a = np.array([-tangent_a[1], tangent_a[0]])
        normal_b = np.array([-tangent_b[1], tangent_b[0]])
        gap = (b - a) * resolution
        tolerance = max(.13, .95 * resolution)
        if (abs(tangent_a @ tangent_b) < math.cos(math.radians(20))
                or abs((gap - center_a) @ normal_a) > tolerance
                or abs((-gap - center_b) @ normal_b) > tolerance):
            continue
        line = _line_pixels(a, b)[1:-1]
        index = tuple(line.T)
        if not scope[index].all() or observed[index].any() or added[index].any():
            continue
        added[index] = True
        used[[i, j]] = True
        if short:
            short_count += 1
        else:
            jump_count += 1
    return added, short_count, jump_count


def wall_boundaries(xyz, normals, origin_xy, resolution, shape, footprint, max_jump):
    """Return an x-first Boolean wall raster and JSON-compatible diagnostics."""
    xyz, footprint = np.asarray(xyz), np.asarray(footprint, dtype=bool)
    normals = None if normals is None else np.asarray(normals)
    origin_xy = np.asarray(origin_xy, dtype=float)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or footprint.ndim != 2
            or tuple(shape) != footprint.shape or not footprint.size
            or (normals is not None and normals.shape != xyz.shape)
            or origin_xy.shape != (2,) or not np.isfinite(origin_xy).all()
            or not np.isfinite(resolution) or resolution <= 0
            or not np.isfinite(max_jump) or max_jump < 0):
        raise ValueError("Wall linking requires aligned point data, an XY raster and physical distances")

    audit = dict(
        anchors=0, observed_wall_cells=0, short_links=0, jump_links=0, linked_wall_cells=0,
        wall_cells=0, max_jump_m=float(max_jump), density_weight="ceiling_biased_per_paper_prose",
    )
    empty = np.zeros(footprint.shape, bool)
    if not footprint.any() or not len(xyz):
        return empty, audit
    scope = ndimage.binary_dilation(footprint, structure=np.ones((3, 3), bool))
    finite = np.isfinite(xyz).all(axis=1)
    point_ids = np.flatnonzero(finite)
    xy = np.floor((xyz[point_ids, :2] - origin_xy) / resolution).astype(np.int64)
    inside = ((xy >= 0) & (xy < shape)).all(axis=1)
    xy, point_ids = xy[inside], point_ids[inside]
    inside = scope[tuple(xy.T)]
    xy, point_ids = xy[inside], point_ids[inside]
    if not len(point_ids):
        return empty, audit
    ids = xy[:, 0] * footprint.shape[1] + xy[:, 1]
    del xy
    if len(point_ids) < len(xyz):
        xyz = xyz[point_ids]
        if normals is not None:
            normals = normals[point_ids]
    column_evidence = _column_evidence(xyz, normals, ids, scope, resolution)
    if column_evidence is None:
        return empty, audit
    evidence, persistent, heights = column_evidence
    observed, direction, anchors, threshold = _observed_runs(evidence, persistent, resolution)
    added, short_count, jump_count = _link_gaps(observed, persistent, evidence, direction, scope, resolution, max_jump)
    walls = observed | added
    audit.update(
        heights,
        evidence_cells=int(np.count_nonzero(evidence)), persistent_wall_candidates=int(persistent.sum()),
        anchors=int(anchors.sum()), anchor_threshold=float(threshold), observed_wall_cells=int(observed.sum()),
        short_links=short_count, jump_links=jump_count, linked_wall_cells=int(added.sum()),
        wall_cells=int(walls.sum()),
    )
    return walls, audit
