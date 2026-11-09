"""Conservative, metre-based cleanup of the potential-field room labels.

Wide-opening cores describe room topology, independently of furniture-sized
holes in the free-space footprint.  They are deliberately not expanded across
narrow passages: a room component must overlap a core to join its merge group.
"""
from collections import deque

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label
from scipy.signal import find_peaks
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _structure_footprint(grid, interior, *, heights=None, protected_walls=None):
    """Separate persistent walls from shorter obstacles in one storey.

    A prominent occupied-height plane pair estimates the main floor/ceiling;
    interior-height quantiles are a fallback for incomplete surfaces. The
    middle height band excludes floor and ceiling; a column occupied in at
    least 40% of that band over a 1.2 m vertical span is treated as a wall.  Tall cabinets
    may therefore remain barriers, which is safer than erasing a real wall.
    """
    if heights is not None:
        return _local_structure_footprint(grid, interior, heights, protected_walls)
    free = interior.any(axis=2)
    walls = np.zeros(free.shape, dtype=bool)
    band = None
    if free.any():
        bottom = np.argmax(interior, axis=2)[free]
        top = interior.shape[2] - 1 - np.argmax(interior[:, :, ::-1], axis=2)[free]
        low = int(np.floor(np.quantile(bottom, .1)))
        high = int(np.ceil(np.quantile(top, .9)))
        counts = grid.busy.sum(axis=(0, 1))
        peaks = find_peaks(np.r_[0, counts, 0], prominence=.12 * counts.max(),
                           distance=max(1, int(round(.5 / grid.voxel))))[0] - 1
        pairs = []
        for a in peaks:
            for b in peaks:
                if (1.8 <= (b - a) * grid.voxel <= 4.5
                        and min(counts[a], counts[b]) >= .15 * counts.max()):
                    overlap = int(np.count_nonzero(grid.busy[:, :, a] & grid.busy[:, :, b]))
                    if overlap >= .25 * min(counts[a], counts[b]):
                        pairs.append((int(a), int(b), overlap))
        if pairs:
            low, high, _ = max(pairs, key=lambda pair: pair[2])
        trim = max(1, int(np.ceil(.25 / grid.voxel)))
        start, stop = low + trim, high - trim + 1
        if stop - start >= 2:
            band = (start, stop)
            occupied = grid.busy[:, :, start:stop]
            bottom_hit = np.argmax(occupied, axis=2)
            top_hit = occupied.shape[2] - 1 - np.argmax(occupied[:, :, ::-1], axis=2)
            walls = ((np.mean(occupied, axis=2) >= .4)
                     & ((top_hit - bottom_hit) * grid.voxel >= 1.2))
            # Spurious interior below the floor or above the ceiling must not
            # reconnect rooms over/under a genuine partition wall.
            free = np.mean(interior[:, :, start:stop], axis=2) >= .6
    # Only fill enclosed holes whose deepest point is <=0.6 m from observed
    # interior. A large unobserved room must not become an invented shortcut.
    # Wall columns are removed even if classified interior exists above them.
    holes = binary_fill_holes(free | walls) & ~free & ~walls
    filled = np.zeros(free.shape, dtype=bool)
    if holes.any():
        hole_ids, hole_count = label(holes)
        depth = distance_transform_edt(~free) * grid.voxel
        maximum = np.zeros(hole_count + 1, dtype=float)
        np.maximum.at(maximum, hole_ids[holes], depth[holes])
        allowed_holes = maximum <= .6 + 1e-12
        allowed_holes[0] = False
        filled = allowed_holes[hole_ids]
    footprint = (free | filled) & ~walls
    return footprint, walls, band


def _local_structure_footprint(grid, interior, heights, protected_walls):
    """Use observed local storeys without weakening the wall veto."""
    legacy, old_walls, band = _structure_footprint(grid, interior)
    floor, ceiling = np.asarray(heights["floor_z"]), np.asarray(heights["ceiling_z"])
    valid = np.asarray(heights["valid"], bool) & (np.asarray(heights["confidence"]) >= .5)
    protected = old_walls.copy() if protected_walls is None else np.asarray(protected_walls, bool)
    if any(a.shape != grid.shape[:2] for a in (floor, ceiling, valid, protected)):
        raise ValueError("Structural footprint maps must match XY occupancy")
    valid &= np.isfinite(floor) & np.isfinite(ceiling)
    z = np.arange(grid.shape[2])[None, None, :]
    band_mask = ((z > floor[:, :, None] + .25 / grid.voxel)
                 & (z < ceiling[:, :, None] - .25 / grid.voxel))
    amount = band_mask.sum(2)
    valid &= amount >= 2
    free = legacy.copy()
    local_free = (interior & band_mask).sum(2) >= .6 * np.maximum(amount, 1)
    free[valid] = local_free[valid]
    walls = protected | (old_walls & ~valid)
    free &= ~walls
    holes = binary_fill_holes(free | walls) & ~free & ~walls
    if holes.any():
        ids, count = label(holes)
        distance = distance_transform_edt(~free) * grid.voxel
        maximum = np.zeros(count + 1)
        np.maximum.at(maximum, ids[holes], distance[holes])
        allowed = maximum <= .6 + 1e-12
        allowed[0] = False
        free |= allowed[ids]
    return free & ~walls, walls, band


def _components(cells, labels, valid, shape, partitions=None):
    """Connected components with equal original labels, using four neighbors."""
    active = np.flatnonzero((labels >= 0) & valid)
    nodes = np.full(shape, -1, dtype=np.int32)
    nodes[tuple(cells[active].T)] = np.arange(len(active))
    original = np.full(shape, -1, dtype=np.int64)
    original[tuple(cells[active].T)] = labels[active]
    rows, cols = [], []
    for axis in range(2):
        lo = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        hi = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        left, right = nodes[lo], nodes[hi]
        adjacent = (left >= 0) & (right >= 0) & (original[lo] == original[hi])
        if partitions is not None:
            adjacent &= partitions[lo] == partitions[hi]
        rows.append(left[adjacent])
        cols.append(right[adjacent])
    row, col = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(row), dtype=bool), (row, col)),
                       shape=(len(active), len(active))).tocsr()
    count, component = connected_components(graph, directed=False)
    return active, component, count


def _extend_labels(cells, labels, footprint):
    """Flood unassigned interior locally, keeping every existing room seed."""
    raster = np.full(footprint.shape, -1, dtype=np.int64)
    valid = footprint[tuple(cells.T)]
    seeds = (labels >= 0) & valid
    raster[tuple(cells[seeds].T)] = labels[seeds]
    queue = deque(map(tuple, cells[seeds]))
    nx, ny = footprint.shape
    while queue:
        x, y = queue.popleft()
        for a, b in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= a < nx and 0 <= b < ny and footprint[a, b] and raster[a, b] < 0:
                raster[a, b] = raster[x, y]
                queue.append((a, b))
    return raster[tuple(cells.T)]


def _merge_neighbors(cells, active, component, count, component_core, mergeable,
                      cores, opening, voxel):
    """Union only components sharing a contiguous, physically wide interface."""
    raster = np.full(cores.shape, -1, dtype=np.int32)
    raster[tuple(cells[active].T)] = component
    pairs, positions = [], []
    for axis in range(2):
        lo = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        hi = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        a, b = raster[lo], raster[hi]
        possible = (a >= 0) & (b >= 0) & (a != b)
        q = np.argwhere(possible)
        aa, bb = a[possible], b[possible]
        use = (mergeable[aa] & mergeable[bb]
               & (component_core[aa] == component_core[bb])
               & (cores[lo][possible] > 0)
               & (cores[lo][possible] == component_core[aa])
               & (cores[lo][possible] == cores[hi][possible]))
        pairs.append(np.column_stack((np.minimum(aa[use], bb[use]), np.maximum(aa[use], bb[use]))))
        positions.append(q[use])
    parent = np.arange(count, dtype=np.int32)

    def root(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    pairs, positions = np.concatenate(pairs), np.concatenate(positions)
    if len(pairs):
        keys = pairs[:, 0].astype(np.int64) * count + pairs[:, 1]
        order = np.argsort(keys)
        keys, positions = keys[order], positions[order]
        starts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1, len(keys)]
        for begin, end in zip(starts[:-1], starts[1:]):
            q = positions[begin:end]
            lo, hi = q.min(0), q.max(0) + 1
            interface = np.zeros(tuple(hi - lo), bool)
            local = q - lo
            interface[tuple(local.T)] = True
            segments, nsegments = label(interface, structure=np.ones((3, 3), bool))
            # Geometric span avoids treating a curled narrow interface as a
            # wide opening merely because its accumulated edge length is high.
            width = max((np.linalg.norm(np.ptp(np.argwhere(segments == part), axis=0)) + 1) * voxel
                        for part in range(1, nsegments + 1))
            if width >= opening - 1e-12:
                a, b = divmod(int(keys[begin]), count)
                parent[root(b)] = root(a)
    return np.array([root(part) for part in range(count)], dtype=np.int32)


def refine_rooms(grid, interior, cells, labels, min_room_area=2., merge_opening=1.6,
                 projection_rejected=None, *, heights=None, protected_walls=None, partition_owners_out=None):
    """Refine per-stack labels and return ``(labels, diagnostics)``.

    ``min_room_area`` is square metres of classified free-space footprint,
    measured after supported merges. ``merge_opening`` is the minimum passage
    width in metres connecting a wide-opening core. Zero disables each step;
    setting both to zero returns the input labels exactly for legacy compares.

    Existing labels first grow into reachable unassigned interior by local
    four-neighbor propagation, so area measures the room footprint rather than
    only its high-potential seeds. Disconnected pieces are then separated. Merge candidates
    must have >=80% of their core overlap in one core, >=15% of their complete
    component inside that core, and >=0.25 m² of core overlap. This prevents a
    tiny overlap from making one component a bridge between distinct rooms.
    Small residual pieces may attach by four-neighbor propagation through the
    same core, never through a wall or a different core. Exterior/unobserved
    cells are not assigned by a global nearest-neighbor search.

    Optional ``projection_rejected`` is a Boolean output buffer per input cell.
    It identifies area-pruned columns, including extended cells. Structural
    wall columns are excluded: wall surfaces still need nearby room labels.
    """
    labels = np.asarray(labels)
    cells = np.asarray(cells, dtype=np.int64)
    if (not np.isfinite(min_room_area) or min_room_area < 0
            or not np.isfinite(merge_opening) or merge_opening < 0):
        raise ValueError("Room area and merge opening must be finite and nonnegative")
    if cells.shape != (len(labels), 2) or interior.shape != grid.busy.shape:
        raise ValueError("Cell labels or interior mask have incompatible shapes")
    if projection_rejected is not None:
        if projection_rejected.shape != labels.shape or projection_rejected.dtype != bool:
            raise ValueError("Projection rejection buffer must be boolean and match labels")
        projection_rejected[:] = False
    if partition_owners_out is not None:
        if partition_owners_out.shape != grid.shape[:2] or partition_owners_out.dtype != np.int32:
            raise ValueError("Partition output must be an int32 XY map")
        partition_owners_out[:] = 0
    diagnostics = dict(enabled=bool(min_room_area or merge_opening),
                       min_room_area_m2=float(min_room_area), merge_opening_m=float(merge_opening),
                       input_rooms=int(len(np.unique(labels[labels >= 0]))),
                       input_components=0, wide_cores=0, merged_components=0,
                       pruned_components=0, pruned_cells=0, attached_cells=0,
                       preserved_components=0,
                       extended_cells=0,
                       wall_cells=0, wall_band_voxels=None)
    if not min_room_area and not merge_opening:
        diagnostics["output_rooms"] = diagnostics["input_rooms"]
        return labels.copy(), diagnostics
    result = np.full(len(labels), -1, dtype=np.int32)
    footprint, walls, band = _structure_footprint(grid, interior, heights=heights, protected_walls=protected_walls)
    diagnostics["wall_band_voxels"] = list(band) if band is not None else None
    diagnostics["wall_cells"] = int(walls.sum())
    original_labels = labels
    # Preserve observed space outside the main storey for area cleanup and
    # projection. Only the height-supported footprint may form merge cores.
    # A global height band must not erase a lower landing or a ramp.
    assignment_footprint = (interior.any(axis=2) | footprint) & ~walls
    labels = _extend_labels(cells, labels, assignment_footprint)
    diagnostics["extended_cells"] = int(np.count_nonzero((original_labels < 0) & (labels >= 0)))
    valid = assignment_footprint[tuple(cells.T)]
    cores = np.zeros(footprint.shape, dtype=np.int32)
    if merge_opening:
        distance = distance_transform_edt(np.pad(footprint, 1))[1:-1, 1:-1] * grid.voxel
        cores, ncores = label(footprint & (distance >= merge_opening / 2))
        diagnostics["wide_cores"] = int(ncores)
    partition_seeds = cores.copy()
    if heights is not None and merge_opening:
        sizes_m2 = np.bincount(cores.ravel()) * grid.voxel ** 2
        strong = sizes_m2 >= max(2., min_room_area) - 1e-12
        strong[0] = False
        partition_seeds[~strong[cores]] = 0
        diagnostics["partition_seed_min_area_m2"] = float(max(2., min_room_area))
        diagnostics["partition_seed_count"] = int(strong.sum())
    partitions = None
    if heights is not None and merge_opening:
        from .partitions import core_partition_supported
        # Missing height support is not a boundary: infer ownership through
        # observed assignment space, still excluding every structural wall.
        partitions, partition_info = core_partition_supported(partition_seeds, assignment_footprint, walls, grid.voxel)
        diagnostics["partition_evidence"] = partition_info
        if partition_owners_out is not None:
            partition_owners_out[:] = partitions
    diagnostics["structural_core_partition"] = partitions is not None
    active, component, count = _components(cells, labels, valid, footprint.shape, partitions)
    diagnostics["input_components"] = int(count)
    if not len(active):
        diagnostics["output_rooms"] = 0
        diagnostics["pruned_cells"] = int(np.count_nonzero((original_labels >= 0) | (labels >= 0)))
        return result, diagnostics
    sizes = np.bincount(component, minlength=count)
    cell_cores = cores[tuple(cells[active].T)]
    component_core = np.zeros(count, dtype=np.int32)
    mergeable = np.zeros(count, dtype=bool)
    overlap = np.flatnonzero(cell_cores > 0)
    if len(overlap):
        pairs, amounts = np.unique(np.column_stack((component[overlap], cell_cores[overlap])),
                                  axis=0, return_counts=True)
        totals = np.bincount(pairs[:, 0], weights=amounts, minlength=count)
        # Iterate over overlap pairs, not image pixels or every possible core.
        strongest = np.zeros(count, dtype=np.int64)
        for (part, core), amount in zip(pairs, amounts):
            if amount > strongest[part]:
                strongest[part] = amount
                component_core[part] = core
        dominant = strongest >= .8 * totals
        component_core[~dominant] = 0
        mergeable = (dominant & (component_core > 0) & (strongest >= .15 * sizes)
                     & (strongest * grid.voxel ** 2 >= .25 - 1e-12))
    # A shared eroded core already establishes a physically wide path. Within
    # that one core, an arbitrary upstream label edge need not itself be wide.
    # Distinct cores, structural walls and ambiguous overlaps remain excluded.
    contact_width = min(merge_opening, max(.4, 2 * grid.voxel)) if heights is not None else merge_opening
    diagnostics["same_core_contact_m"] = float(contact_width)
    groups = _merge_neighbors(cells, active, component, count, component_core, mergeable,
                              cores, contact_width, grid.voxel)
    _, groups = np.unique(groups, return_inverse=True)
    diagnostics["merged_components"] = int(count - len(np.unique(groups)))
    group_sizes = np.bincount(groups, weights=sizes)
    keep_group = group_sizes * grid.voxel ** 2 >= min_room_area - 1e-12
    keep = keep_group[groups[component]]
    result[active[keep]] = groups[component[keep]]
    small_components = ~keep_group[groups]

    if merge_opening and small_components.any() and keep.any():
        # Multi-source local propagation. Allowed core identities are constant
        # across each dominant-overlap component; ambiguous pieces block it.
        allowed = cores.copy()
        allowed[tuple(cells[active].T)] = component_core[component]
        owner = np.full(footprint.shape, -1, dtype=np.int32)
        owner[tuple(cells[active[keep]].T)] = result[active[keep]]
        queue = deque(map(tuple, cells[active[keep & (component_core[component] > 0)]]))
        nx, ny = footprint.shape
        while queue:
            x, y = queue.popleft()
            core = allowed[x, y]
            for a, b in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if (0 <= a < nx and 0 <= b < ny and footprint[a, b]
                        and allowed[a, b] == core and owner[a, b] < 0):
                    owner[a, b] = owner[x, y]
                    queue.append((a, b))
        candidate = active[small_components[component]]
        attached = owner[tuple(cells[candidate].T)]
        result[candidate] = attached
        diagnostics["attached_cells"] = int(np.count_nonzero(attached >= 0))

    rejected = ((original_labels >= 0) | (labels >= 0)) & (result < 0)
    if projection_rejected is not None:
        projection_rejected[:] = rejected & valid
    diagnostics["pruned_cells"] = int(rejected.sum())
    remaining = np.bincount(component, weights=(result[active] >= 0), minlength=count)
    diagnostics["pruned_components"] = int(np.count_nonzero(remaining == 0))
    # The main height band cannot establish partitions on a lower landing or
    # an incomplete scan outside that storey. Preserve original identity for
    # those retained pieces instead of inventing a room ID per scan gap.
    source = np.empty(count, dtype=labels.dtype)
    source[component] = labels[active]
    height_support = np.bincount(component, weights=footprint[tuple(cells[active].T)], minlength=count)
    uncertain = height_support < .15 * sizes
    group_members = np.bincount(groups)
    retained = keep_group[groups]
    domains = label(assignment_footprint)[0] if heights is not None else None
    component_domain = np.zeros(count, np.int32)
    if domains is not None:
        observed_domains = domains[tuple(cells[active].T)]
        if partitions is not None:
            pairs = np.column_stack([observed_domains, partitions[tuple(cells[active].T)]])
            _, observed_domains = np.unique(pairs, axis=0, return_inverse=True)
        component_domain[component] = observed_domains
    for original in np.unique(source[retained]):
        # A core-supported partition must survive even when the upstream
        # clustering assigned the same ID on both sides of a doorway.
        parts = np.flatnonzero((source == original) & retained
                               & ((component_core == 0) if partitions is not None else True))
        if not len(parts):
            continue
        domain_values = np.unique(component_domain[parts]) if domains is not None else [0]
        for domain in domain_values:
            local = parts[component_domain[parts] == domain] if domains is not None else parts
            anchor = local[np.argmax(sizes[local])]
            for part in local[uncertain[local] & (group_members[groups[local]] == 1)]:
                if part != anchor:
                    result[result == groups[part]] = groups[anchor]
                    diagnostics["preserved_components"] += 1
    present = np.unique(result[result >= 0])
    if len(present):
        assigned = result >= 0
        result[assigned] = np.searchsorted(present, result[assigned])
    diagnostics["output_rooms"] = int(len(present))
    return result, diagnostics
