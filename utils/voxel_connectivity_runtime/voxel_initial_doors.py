"""Verify projected Li door patches before using them as room boundaries."""

import math

import numpy as np
from scipy import ndimage

from .voxel_connectivity_review import _floor_level
from .voxel_door_candidates import _bits, _coherent_lintel_fraction, _grounded_wall


_CONNECTED = np.ones((3, 3), bool)


def _components(mask):
    labels, _ = ndimage.label(mask, structure=_CONNECTED)
    for index, box in enumerate(ndimage.find_objects(labels), 1):
        yield np.argwhere(labels[box] == index) + [axis.start for axis in box]


def _bank_extent(depth, width, resolution):
    positions = np.unique(np.round(depth[(depth >= -resolution / 2) & (depth <= .75)
                                         & width], 5))
    return (len(positions) > 1 and positions[0] <= max(.20, resolution)
            and np.ptp(positions) >= .34 - 1e-9
            and np.diff(positions).max() <= max(.24, 1.5 * resolution))


def _open_floor_bypass(xy, t, n, clear_floor, along, across, resolution):
    """Require measured paths around both ends of a suspected obstruction."""
    low, high = across.min() - .40, across.max() + .40
    across_strip = (n >= low) & (n <= high)
    for end, sign in ((along.min(), -1), (along.max(), 1)):
        beyond = (t - end) * sign
        inside = across_strip & (beyond >= .10) & (beyond <= .55)
        take = inside & clear_floor
        if take.sum() * resolution ** 2 < .65 * .45 * (high - low):
            return False
        points = xy[take]
        offset = points.min(axis=0)
        local = points - offset
        mask = np.zeros(tuple(local.max(axis=0) + 1), bool)
        mask[tuple(local.T)] = True
        labels = ndimage.label(mask)[0][tuple(local.T)]
        near, far = n[take] <= low + .18, n[take] >= high - .18
        if not np.intersect1d(labels[near], labels[far]).size:
            return False
    return True


def _contradicts_door(xy, t, n, observed, along, across, h, v, underfoot, floor, z_step, resolution):
    """Find a measured low surface with continuous floor around it."""
    levels = np.arange(floor + math.ceil(.50 / z_step), min(64, floor + math.floor(1.45 / z_step) + 1),
                       dtype=np.uint64)
    if not len(levels):
        return False
    surface = ((underfoot[:, None] >> levels) & 1) != 0
    if len(levels) > 1:
        surface = surface[:, :-1] | surface[:, 1:]
    if surface.sum(axis=0).max() < max(3, .50 * len(underfoot), .30 / resolution):
        return False
    at = tuple(xy.T)
    body = _bits(floor + math.ceil(.25 / z_step), floor + math.floor(1.52 / z_step))
    floor_returns = (h[at] & _bits(floor - 1, floor + 1)) != 0
    clear_floor = observed & floor_returns & ((v[at] & body) == 0)
    return _open_floor_bypass(xy, t, n, clear_floor, along, across, resolution)


def _verify_axis(door, patch, context, h, v, z_step, resolution, tangent):
    along = patch @ tangent
    normal = np.array([-tangent[1], tangent[0]])
    across = patch @ normal
    width = np.ptp(along) + resolution
    if not .55 - 1e-9 <= width <= 3.20 or np.ptp(across) > .65:
        return "implausible_aperture"

    xy, offsets, observed = context
    t, n = offsets @ tangent, offsets @ normal
    lateral = (t >= along.min() - resolution / 2) & (t <= along.max() + resolution / 2)
    sides = [
        observed & lateral & (n >= across.max() + .10) & (n <= across.max() + .65),
        observed & lateral & (n <= across.min() - .10) & (n >= across.min() - .65),
    ]
    levels = [_floor_level(h, v, xy[:, 0], xy[:, 1], side, z_step) for side in sides]
    if any(level is None for level in levels):
        return "unknown_floor"
    (a, top_a), (b, top_b) = levels
    if abs(a - b) * z_step > .25 + z_step / 2 or min(top_a - a, top_b - b) * z_step < 1.60:
        return "different_or_unconfirmed_floors"
    floor = round((a + b) / 2)

    # Use occupancy at shared heights along the opening, even if the
    # projection includes more than one raster row across the wall.
    bins = np.floor((along - along.min()) / resolution + .5).astype(int)
    overhead = np.zeros(bins.max() + 1, np.uint64)
    underfoot = np.zeros_like(overhead)
    np.bitwise_or.at(overhead, bins, v[tuple(door.T)])
    np.bitwise_or.at(underfoot, bins, h[tuple(door.T)])
    ground = (underfoot & _bits(floor - 1, floor + 1)) != 0
    passage = _bits(floor + math.ceil(.25 / z_step), floor + math.floor(1.52 / z_step))
    clear = (overhead & passage) == 0
    ground_fraction, clear_fraction = ground.mean(), clear.mean()
    if ground_fraction < .25 or clear_fraction < .65:
        return "unconfirmed_ground_or_clearance"
    if _coherent_lintel_fraction(np.where(clear, overhead, np.uint64(0)), floor, z_step, width) < .50:
        if ground_fraction >= .65 and clear_fraction >= .80 and _contradicts_door(
                xy, t, n, observed, along, across, h, v, underfoot, floor, z_step, resolution):
            return "measured_low_surface_in_open_interior"
        return "unconfirmed_lintel"

    grounded = _grounded_wall(v[tuple(xy.T)], floor, z_step)
    thickness = max(.13, .8 * resolution)
    collinear, perpendicular = [], []
    for end, sign in ((along.min(), -1), (along.max(), 1)):
        depth = (t[grounded] - end) * sign
        cross = n[grounded] - (across.min() + across.max()) / 2
        collinear.append(_bank_extent(depth, np.abs(cross) <= thickness, resolution))
        at_end = np.abs(depth) <= thickness
        perpendicular.append(any(_bank_extent(cross * side, at_end, resolution) for side in (-1, 1)))
    if all(collinear) or (collinear[0] and perpendicular[1]) or (collinear[1] and perpendicular[0]):
        return "confirmed"
    return "unconfirmed_wall_banks"


def verify_initial_doors(candidates, observed, walls, height_rasters, resolution):
    """Return confirmed and contradicted masks, leaving uncertain patches intact."""
    count = int(candidates.sum())
    report = dict(status="completed", candidate_pixels=count, confirmed_pixels=0, removed_pixels=0,
                  retained_pixels=count, candidate_count=0, confirmed_count=0, uncertain_count=0,
                  removed_count=0, uncertain_reasons={}, removed_reasons={})
    if not count:
        report["status"] = "no_candidates"
        return candidates, candidates, report
    confirmed = np.zeros_like(candidates)
    contradicted = np.zeros_like(candidates)
    if height_rasters is None:
        _, n = ndimage.label(candidates, structure=_CONNECTED)
        report["status"] = "unsupported_height_range"
        report["candidate_count"] = report["uncertain_count"] = n
        report["uncertain_reasons"]["unsupported_height_range"] = n
        return confirmed, contradicted, report

    h, v, z_step = height_rasters
    padding = math.ceil(.85 / resolution)
    for door in _components(candidates):
        report["candidate_count"] += 1
        center = door.mean(axis=0)
        patch = (door - center) * resolution
        _, axes = np.linalg.eigh(patch.T @ patch)
        low = np.maximum(door.min(axis=0) - padding, 0)
        high = np.minimum(door.max(axis=0) + padding + 1, candidates.shape)
        box = tuple(slice(a, b) for a, b in zip(low, high))
        scope = observed[box] | walls[box]
        xy = np.argwhere(scope) + low
        at = tuple(xy.T)
        context = (xy, (xy - center) * resolution, observed[at] & ~walls[at] & ~candidates[at])
        outcomes = [_verify_axis(door, patch, context, h, v, z_step, resolution, axis) for axis in axes.T[::-1]]
        if "confirmed" in outcomes:
            confirmed[tuple(door.T)] = True
            report["confirmed_count"] += 1
            report["confirmed_pixels"] += len(door)
        elif "measured_low_surface_in_open_interior" in outcomes:
            contradicted[tuple(door.T)] = True
            report["removed_count"] += 1
            report["removed_pixels"] += len(door)
            reasons = report["removed_reasons"]
            reason = "measured_low_surface_in_open_interior"
            reasons[reason] = reasons.get(reason, 0) + 1
        else:
            report["uncertain_count"] += 1
            reasons = report["uncertain_reasons"]
            reason = outcomes[0]
            reasons[reason] = reasons.get(reason, 0) + 1
    report["retained_pixels"] -= report["removed_pixels"]
    return confirmed, contradicted, report


def _neighbor_owners(labels, doors, walls, points, direction, resolution):
    owners = np.full(len(points), -1, labels.dtype)
    unknown = np.zeros(len(points), np.int32)
    pitch = np.linalg.norm(direction) * resolution
    reach = math.ceil(1.10 / pitch)
    allowed_unknown = max(1, math.floor(.15 / pitch))
    active = np.arange(len(points))
    for step in range(1, reach + 1):
        xy = points[active] + direction * step
        inside = ((xy >= 0) & (xy < labels.shape)).all(axis=1)
        active = active[inside]
        if not len(active):
            break
        at = tuple(xy[inside].T)
        values = labels[at]
        owners[active] = values
        is_door = doors[at]
        unknown[active] += ~is_door
        active = active[(values < 0) & (is_door | (~walls[at] & (unknown[active] <= allowed_unknown)))]
    return owners


def confirmed_initial_door_pairs(labels, doors, walls, resolution):
    """Protect the pair supported on opposite sides of each confirmed door."""
    pairs = set()
    if not doors.any():
        return pairs
    for points in _components(doors):
        support = {}
        for direction in map(np.asarray, ((1, 0), (0, 1), (1, 1), (1, -1))):
            opposite = np.column_stack([
                _neighbor_owners(labels, doors, walls, points, side * direction, resolution) for side in (-1, 1)
            ])
            valid = (opposite >= 0).all(axis=1) & (opposite[:, 0] != opposite[:, 1])
            if not valid.any():
                continue
            owners = np.sort(opposite[valid], axis=1)
            lateral = points[valid] @ np.array([-direction[1], direction[0]])
            evidence = np.unique(np.column_stack((owners, lateral)), axis=0)
            unique, counts = np.unique(evidence[:, :2], axis=0, return_counts=True)
            for pair, count in zip(unique, counts):
                pair = tuple(map(int, pair))
                width = count * resolution / np.linalg.norm(direction)
                support[pair] = max(support.get(pair, 0.), width)
        if support:
            pair = max(support, key=support.get)
            if support[pair] >= .22 - 1e-9:
                pairs.add(pair)
    return pairs
