"""Lift stable coarse room seeds onto the fine XY grid for internal topology."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


_MIN_STABLE_WIDTH_M = .80


def _coarse_indices(fine_size, coarse_size, resolution, coarse_resolution):
    """Match a fine voxel's centre to the containing coarse voxel."""
    centres = (np.arange(fine_size, dtype=float) + .5) * (resolution / coarse_resolution)
    indices = np.floor(np.nextafter(centres, np.inf)).astype(np.intp)
    return indices[indices < coarse_size]


def _retain_components(components, observed, count, resolution, min_room_area):
    """Retain multiple children only when each has stable measured interior."""
    measured = np.bincount(components[observed], minlength=count + 1)
    measured[0] = 0
    if not measured.any():
        return []
    if count == 1:
        return [1]
    stable = []
    for child, bounds in enumerate(ndimage.find_objects(components), start=1):
        area = measured[child] * resolution ** 2
        width = min(axis.stop - axis.start for axis in bounds) * resolution
        if measured[child] and area + 1e-9 >= min_room_area and width + 1e-9 >= _MIN_STABLE_WIDTH_M:
            stable.append(child)
    # Keep an observed representative of a coarse room even when its fine
    # coverage is too sparse to create any new, independent room seeds.
    return stable or [int(measured.argmax())]


def refine_seed_labels(coarse_labels, coarse_walls, coarse_doors, observed, walls, doors,
                       resolution, coarse_resolution, min_room_area):
    """Return connected fine seed labels, both barrier masks, and diagnostics.

    Both grids have the same XY origin and x-first indexing. Lifted cells
    without fine observations are support for topology. Keep them internal
    and mask them using the original observations before point assignment.
    """
    coarse_labels = np.asarray(coarse_labels)
    coarse_walls, coarse_doors, observed, walls, doors = (
        np.asarray(raster, dtype=bool) for raster in (coarse_walls, coarse_doors, observed, walls, doors)
    )
    if (coarse_labels.ndim != 2 or coarse_labels.dtype.kind != "i"
            or coarse_walls.shape != coarse_labels.shape or coarse_doors.shape != coarse_labels.shape
            or observed.ndim != 2 or walls.shape != observed.shape or doors.shape != observed.shape
            or not np.isfinite(resolution) or not np.isfinite(coarse_resolution)
            or not 0 < resolution <= coarse_resolution
            or not np.isfinite(min_room_area) or min_room_area < 0):
        raise ValueError("Coarse seed support requires aligned XY grids and valid metric thresholds")

    indices = [_coarse_indices(fine, coarse, resolution, coarse_resolution)
               for fine, coarse in zip(observed.shape, coarse_labels.shape)]
    covered = tuple(slice(0, len(axis)) for axis in indices)
    coarse_cells = np.ix_(*indices)
    lifted = np.full(observed.shape, -1, coarse_labels.dtype)
    lifted[covered] = coarse_labels[coarse_cells]
    refined_walls, refined_doors = walls.copy(), doors.copy()
    refined_walls[covered] |= coarse_walls[coarse_cells]
    refined_doors[covered] |= coarse_doors[coarse_cells]
    active = (lifted >= 0) & ~(refined_walls | refined_doors)

    output = np.full(observed.shape, -1, np.int32)
    coarse_ids = np.unique(coarse_labels[coarse_labels >= 0])
    owner_ids = np.unique(lifted[active])
    room_count = split_rooms = dropped_components = 0
    if len(owner_ids):
        owners = np.zeros(observed.shape, np.int32)
        owners[active] = np.searchsorted(owner_ids, lifted[active]) + 1
        for owner, bounds in enumerate(ndimage.find_objects(owners), start=1):
            components, count = ndimage.label(owners[bounds] == owner)
            retained = _retain_components(components, observed[bounds], count, resolution, min_room_area)
            mapping = np.full(count + 1, -1, np.int32)
            mapping[retained] = np.arange(room_count, room_count + len(retained))
            inside = components > 0
            region = output[bounds]
            region[inside] = mapping[components[inside]]
            room_count += len(retained)
            split_rooms += int(len(retained) > 1)
            dropped_components += count - len(retained)

    diagnostics = dict(
        coarse_resolution_m=float(coarse_resolution), fine_resolution_m=float(resolution),
        coarse_room_count=len(coarse_ids), seed_room_count=room_count,
        split_coarse_rooms=split_rooms, removed_unstable_components=dropped_components,
        fine_observed_seed_cells=int(((output >= 0) & observed).sum()),
        virtual_support_cells=int(((output >= 0) & ~observed).sum()),
        added_coarse_wall_cells=int((refined_walls & ~walls).sum()),
        added_coarse_door_cells=int((refined_doors & ~doors).sum()),
    )
    return output, refined_walls, refined_doors, diagnostics
