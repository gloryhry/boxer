"""Review Li room boundaries with local, single-return voxel evidence.

Empty bins are unknown without scan rays. A merge requires a broad, directly
supported interface and compatible local floors. Narrow or uncertain seams
remain boundaries, and existing unlabeled pixels retain their values.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy import ndimage


_MAX_GAP_M = .35
_WALL_GUARD_M = .75
_BROAD_M = 1.50
_TORSO_HEIGHTS = np.array([.30, .62, .94, 1.26, 1.58, 1.90])
_HEIGHT_BINS = np.arange(64, dtype=np.uint64)


def _contacts(labels, walls, resolution):
    """Find short-gap candidates and independently guard thicker blocked rays."""
    size = int(labels.max()) + 1
    max_gap = math.floor(_MAX_GAP_M / resolution + 1e-9)
    guard_gap = math.floor(_WALL_GUARD_M / resolution + 1e-9)
    rows = []
    pairs = []
    guarded = set()
    count = 0
    for axis in (0, 1):
        raster = np.moveaxis(labels, axis, 0)
        barriers = np.moveaxis(walls, axis, 0)
        n = raster.shape[0]
        coord = np.arange(n, dtype=np.int32)[:, None]
        next_label = np.where(raster >= 0, coord, n)
        np.minimum.accumulate(next_label[::-1], axis=0, out=next_label[::-1])
        distance = next_label[1:] - coord[:-1]
        nearby = distance <= max_gap + 1
        thick_wall = (distance <= guard_gap + 1) & ~nearby & barriers[1:]
        starts, lateral = np.nonzero(
            (raster[:-1] >= 0) & (raster[:-1] != raster[1:])
            & (next_label[1:] < n) & (nearby | thick_wall)
        )
        if not len(starts):
            continue
        ends = next_label[starts + 1, lateral]
        left, right = raster[starts, lateral], raster[ends, lateral]
        codes = np.minimum(left, right).astype(np.int64) * size + np.maximum(left, right)
        long = (left != right) & (ends - starts > max_gap + 1)
        if long.any():
            wall_count = np.cumsum(barriers, axis=0, dtype=np.int32)
            start, end, side = starts[long], ends[long], lateral[long]
            crosses_wall = wall_count[end - 1, side] - wall_count[start, side] == end - start - 1
            protected, counts = np.unique(codes[long][crosses_wall], return_counts=True)
            guarded.update(divmod(int(code), size) for code in protected[counts * resolution >= .30 - 1e-9])
        distinct = (left != right) & ~long
        amount = int(distinct.sum())
        if not amount:
            continue
        count += amount
        rows.append(np.column_stack((np.full(amount, axis), starts[distinct],
                                     ends[distinct], lateral[distinct])).astype(np.int32))
        pairs.append(codes[distinct])
    if not rows:
        return {}, 0, guarded
    rays, codes = np.concatenate(rows), np.concatenate(pairs)
    order = np.argsort(codes, kind="stable")
    codes = codes[order]
    breaks = np.r_[0, np.flatnonzero(np.diff(codes)) + 1, len(codes)]
    rays = rays[order]
    return {divmod(int(codes[i]), size): rays[i:j] for i, j in zip(breaks[:-1], breaks[1:])}, count, guarded


def _height_rasters(working, normals, origin, resolution, shape):
    """Store distinct horizontal and nonhorizontal return heights per column."""
    sampled_z = working[::max(1, len(working) // 120_000), 2]
    low, high = np.quantile(sampled_z, [.001, .999])
    z_origin = float(low - .30)
    z_step = max(.13, float((high - low + .6) / 63))
    if z_step > .30:
        return None
    horizontal = np.zeros(math.prod(shape), np.uint64)
    vertical = np.zeros_like(horizontal)
    for start in range(0, len(working), 250_000):
        block = slice(start, start + 250_000)
        points, normal = working[block], normals[block]
        xy = np.floor((points[:, :2] - origin) / resolution)
        z = np.floor((points[:, 2] - z_origin) / z_step)
        inside = ((xy >= 0) & (xy < shape)).all(axis=1) & (z >= 0) & (z < 64)
        xy, z, normal = xy[inside].astype(np.int64), z[inside].astype(np.uint64), normal[inside]
        ids = xy[:, 0] * shape[1] + xy[:, 1]
        bits = np.left_shift(np.uint64(1), z)
        length = np.sqrt(np.einsum("ij,ij->i", normal, normal))
        nz = np.abs(normal[:, 2])
        flat = (length > 0) & (nz >= .85 * length)
        upright = (length > 0) & (nz <= .65 * length)
        np.bitwise_or.at(horizontal, ids[flat], bits[flat])
        np.bitwise_or.at(vertical, ids[upright], bits[upright])
    return horizontal.reshape(shape), vertical.reshape(shape), z_step


def _floor_level(horizontal, vertical, axial, lateral, room_side):
    """Find a locally repeated low horizontal level, with returns above it."""
    flat = horizontal[axial, lateral][room_side]
    upright = vertical[axial, lateral][room_side]
    if len(flat) < 3:
        return None
    hits = ((flat[:, None] >> _HEIGHT_BINS) & 1) != 0
    counts = hits.sum(axis=0)
    totals = np.bitwise_or.reduce(flat | upright)
    top = int(totals).bit_length() - 1
    # Demand horizontal samples on several distinct room-side columns. A
    # neighboring-bin pair tolerates noise or a slightly uneven floor.
    pairs = (hits[:, :-1] | hits[:, 1:]).sum(axis=0)
    enough = np.flatnonzero(pairs >= max(3, math.ceil(.12 * len(flat)), .25 * pairs.max()))
    if not len(enough):
        return None
    level = int(enough[0])
    level += int(counts[level + 1] > counts[level])
    return level, top


def _strip_groups(centers, lateral, resolution, *, staggered=False):
    low = np.array([centers.min(), lateral.min()])
    xy = np.column_stack((centers, lateral)) - low
    shape = tuple(xy.max(axis=0) + 1)
    mask = np.zeros(shape, bool)
    mask[tuple(xy.T)] = True
    if staggered:
        # Nearby rays can sample the same uneven interface at different
        # offsets. Use the expanded mask only for grouping measured rays.
        along = max(1, math.floor(.05 / resolution + 1e-9))
        across = math.floor(.20 / resolution + 1e-9)
        mask = ndimage.maximum_filter(mask, size=(2 * across + 1, 2 * along + 1), mode="constant")
    elif resolution <= .10 + 1e-9:
        # Join a one-pixel interruption along the interface only.
        padded = np.pad(mask, ((0, 0), (1, 1)))
        mask |= padded[:, :-2] & padded[:, 2:]
    groups = ndimage.label(mask, structure=np.ones((3, 3), bool))[0][tuple(xy.T)]
    order = np.argsort(groups, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(groups[order])) + 1, len(order)]
    return [order[i:j] for i, j in zip(boundaries[:-1], boundaries[1:])]


def _longest_strip(first, last, lateral, resolution, *, staggered=False):
    if not len(first):
        return 0., np.empty(0, np.intp)
    best = np.empty(0, np.intp)
    width = 0.
    for group in _strip_groups((first + last) // 2, lateral, resolution, staggered=staggered):
        positions = np.unique(lateral[group])
        span = (int(positions[-1]) - int(positions[0]) + 1) * resolution
        support = len(positions) * resolution
        if support >= .78 * span and support > width:
            best, width = group, support
    return width, best


def _band_masks(level, step):
    indices = np.floor(level + _TORSO_HEIGHTS / step + .5).astype(int)
    low, high = np.maximum(0, indices - 1), np.minimum(63, indices + 1)
    return np.array([((1 << (int(b) + 1)) - (1 << int(a))) if a <= b else 0
                     for a, b in zip(low, high)], dtype=np.uint64)


def _score_axis(rays, labels, observed, wall, doors, horizontal, vertical, step, resolution, *, staggered=False):
    first, last, lateral = rays.T
    span, strongest = _longest_strip(first, last, lateral, resolution, staggered=staggered)
    report = dict(contact_width_m=round(span, 3), clear_width_m=0., samples=len(first),
                  supported_samples=0, open_samples=0, blocked_samples=0, reason="narrow_or_fragmented")
    if span + 1e-9 < _BROAD_M:
        return report
    first, last, lateral = rays[strongest].T
    supported = np.zeros(len(first), bool)
    open_bands = np.zeros((len(first), 3), bool)
    structural = np.zeros(len(first), bool)
    blocked = np.zeros(len(first), bool)
    floors = np.full(len(first), -1, np.int16)
    sample_block = max(2, math.ceil(.80 / resolution))
    depth = max(1, math.ceil(.60 / resolution))
    offset = np.arange(depth + 1)
    n = labels.shape[0]
    order = np.argsort(lateral, kind="stable")
    strips = lateral[order] // sample_block
    cuts = np.r_[0, np.flatnonzero(np.diff(strips)) + 1, len(order)]
    for start, stop in zip(cuts[:-1], cuts[1:]):
        take = order[start:stop]
        a, b, t = first[take], last[take], lateral[take]
        left = np.maximum(a[:, None] - offset, 0)
        right = np.minimum(b[:, None] + offset, n - 1)
        columns = t[:, None]
        left_side = (labels[left, columns] == labels[a, t, None]) & (left == a[:, None] - offset)
        right_side = (labels[right, columns] == labels[b, t, None]) & (right == b[:, None] + offset)
        floor_a = _floor_level(horizontal, vertical, left, columns, left_side)
        floor_b = _floor_level(horizontal, vertical, right, columns, right_side)
        if floor_a is None or floor_b is None:
            continue
        level_a, top_a = floor_a
        level_b, top_b = floor_b
        if abs(level_a - level_b) * step > .26 + 1e-9:
            continue
        floor = int(round((level_a + level_b) / 2))
        if (min(top_a - level_a, top_b - level_b) * step < 1.65
                or floor + math.ceil(_TORSO_HEIGHTS[-1] / step) >= 64):
            continue
        width = int(np.max(b - a))
        path = np.minimum(a[:, None] + np.arange(width + 1), b[:, None])
        occupied = vertical[path, columns]
        band_masks = _band_masks(floor, step)
        hits = (occupied[:, :, None] & band_masks) != 0
        bands = hits.any(axis=1)
        # Allow shelves/lamp shadows at some heights while demanding a
        # common-height crossing through multiple bands along each full ray.
        clear = ~bands
        middle_pass = clear[:, 1:4] & clear[:, 2:5]
        floor_mask = np.uint64((1 << (floor + 2)) - (1 << max(0, floor - 1)))
        observed_path = observed[path, columns].mean(axis=1) >= .75
        supported[take] = observed_path
        structural[take] = (hits.sum(axis=2) >= 4).any(axis=1) | (bands.sum(axis=1) >= 5)
        barriers = wall[path, columns] | doors[path, columns]
        blocked[take] = barriers.any(axis=1)
        floors[take] = floor
        # A modeled wall/door with no direct observation around the barrier
        # stays unknown. Accept weak barriers only where a floor is observed.
        local_floor = (((horizontal[path, columns] & floor_mask) != 0) & barriers).any(axis=1)
        feasible = observed_path & ~structural[take] & (clear.sum(axis=1) >= 3) & (~blocked[take] | local_floor)
        open_bands[take] = feasible[:, None] & middle_pass

    report.update(supported_samples=int(supported.sum()), open_samples=int(open_bands.any(axis=1).sum()),
                  blocked_samples=int((blocked | structural).sum()))
    if not supported.any():
        report["reason"] = "uncertain_or_different_local_floors"
        return report
    good = np.empty(0, np.intp)
    clear_width = 0.
    for band in range(open_bands.shape[1]):
        candidates = np.flatnonzero(open_bands[:, band])
        width, indices = _longest_strip(first[candidates], last[candidates], lateral[candidates], resolution,
                                        staggered=staggered)
        if width > clear_width:
            good, clear_width = candidates[indices], width
            report["clear_height_band_m"] = _TORSO_HEIGHTS[band + 1:band + 3].tolist()
    report["clear_width_m"] = round(clear_width, 3)
    if not len(good):
        report["reason"] = "vertical_obstruction_or_insufficient_support"
        return report
    fraction_near = float((last[good] - first[good] > 1).mean())
    fraction_blocked = float(blocked[good].mean())
    report.update(near_gap_fraction=round(fraction_near, 3), barrier_fraction=round(fraction_blocked, 3))
    required_width = _BROAD_M + .25 * (fraction_near > .30) + .30 * (fraction_blocked > .25)
    if clear_width + 1e-9 < required_width:
        report["reason"] = "narrow_or_fragmented_clearance"
        return report

    # Look beyond both ends of a clear strip for tall wall returns. A wide
    # but framed doorway is a valid separator too.
    lo, hi = int(lateral[good].min()), int(lateral[good].max())
    bank_width = max(1, math.floor(.55 / resolution))
    axis_width = max(1, math.ceil(.22 / resolution))
    banks = []
    for end, side in ((lo, -1), (hi, 1)):
        edge = good[lateral[good] == end]
        middle = int(np.median((first[edge] + last[edge]) // 2))
        floor = int(np.median(floors[edge]))
        indices = end + side * np.arange(1, bank_width + 1)
        indices = indices[(indices >= 0) & (indices < vertical.shape[1])]
        values = vertical[max(0, middle - axis_width):middle + axis_width + 1, indices]
        strong = ((values[:, :, None] & _band_masks(floor, step)) != 0).sum(axis=2) >= 4
        banks.append(np.count_nonzero(strong.any(axis=0)) * resolution >= .25 - 1e-9)
    if all(banks) and clear_width <= 2.20 + 1e-9:
        report["reason"] = "framed_portal"
        return report
    if staggered and (fraction_near > .30 or fraction_blocked > .10):
        report["reason"] = "staggered_connection_without_direct_support"
        return report
    report["reason"] = "broad_local_connection"
    return report


def _apply_merges(labels, pairs, accepted):
    # Treat unresolved directly adjacent pairs as cannot-links. Two allowed
    # merges cannot indirectly erase another measured or unreviewed seam.
    if not accepted:
        return labels, 0
    members = {}
    forbidden = {}
    accepted_pairs = {pair for pair, _ in accepted}
    for a, b in pairs:
        if (a, b) not in accepted_pairs:
            forbidden.setdefault(a, set()).add(b)
            forbidden.setdefault(b, set()).add(a)
    parent = {}

    def root(room):
        while room in parent:
            room = parent[room]
        return room

    merges = 0
    for (a, b), report in accepted:
        a, b = sorted((root(a), root(b)))
        if a == b:
            report["reason"] = "already_merged"
            continue
        group_a, group_b = members.get(a, {a}), members.get(b, {b})
        deny_a, deny_b = forbidden.get(a, set()), forbidden.get(b, set())
        if group_a & deny_b or group_b & deny_a:
            report["reason"] = "conflicting_boundary"
            continue
        parent[b] = a
        members[a] = group_a | group_b
        forbidden[a] = deny_a | deny_b
        report["accepted"] = True
        merges += 1
    if not merges:
        return labels, 0
    ids = np.unique(labels[labels >= 0])
    owners = [root(int(room)) for room in ids]
    _, compact = np.unique(owners, return_inverse=True)
    mapping = np.full(int(ids[-1]) + 2, -1, labels.dtype)
    mapping[ids + 1] = compact
    return mapping[labels + 1], merges


def _review_pass(working, labels, initial, walls, origin, resolution, height_rasters, protected_pairs, *,
                 staggered=False):
    started = time.perf_counter()
    barriers = set(protected_pairs)
    audit = dict(evidence="local_single_return_voxel_support", seconds=0., contacts=0, candidate_pairs=0,
                 evaluated_pairs=[], accepted_merges=0)

    def finish(reason, result=labels):
        audit.update(status=reason, seconds=round(time.perf_counter() - started, 4))
        return result, audit, barriers, height_rasters

    if not labels.size or labels.max() < 1:
        return finish("too_few_rooms")
    if not len(working):
        return finish("no_geometry")
    pairs, audit["contacts"], thick_walls = _contacts(labels, walls, resolution)
    barriers.update(thick_walls)
    audit["candidate_pairs"] = len(pairs)
    audit["thick_wall_pair_count"] = len(thick_walls)
    broad_pairs = sorted((pair for pair in pairs if len(pairs[pair]) * resolution >= _BROAD_M - 1e-9),
                         key=lambda pair: (-len(pairs[pair]), pair))
    if not broad_pairs:
        return finish("no_broad_candidates")
    if height_rasters is None:
        height_rasters = _height_rasters(np.asarray(working), np.asarray(initial["normals"]),
                                         np.asarray(origin), resolution, labels.shape)
    if height_rasters is None:
        return finish("unsupported_height_range")
    horizontal, vertical, step = height_rasters
    audit["vertical_resolution_m"] = step
    masks = (labels, initial["observed_footprint"], walls, initial["door_columns"], horizontal, vertical)
    accepted = []
    for pair in broad_pairs:
        if pair in protected_pairs:
            reason = "preserved_boundary" if staggered else "recovered_door_guard"
            audit["evaluated_pairs"].append(dict(rooms=list(pair), reason=reason, accepted=False))
            continue
        if pair in thick_walls:
            audit["evaluated_pairs"].append(dict(rooms=list(pair), reason="thick_wall_guard", accepted=False))
            continue
        rays = pairs[pair]
        axes = []
        for axis in (0, 1):
            active = rays[rays[:, 0] == axis, 1:]
            if not len(active):
                continue
            score = _score_axis(active, *(np.moveaxis(mask, axis, 0) for mask in masks), step,
                                resolution, staggered=staggered)
            axes.append(score | dict(axis=axis))
        strongest = max(axes, key=lambda row: (row["reason"] == "framed_portal",
                                              row["reason"] == "broad_local_connection",
                                              row["clear_width_m"], row["contact_width_m"]))
        report = dict(rooms=list(pair), accepted=False, **strongest)
        audit["evaluated_pairs"].append(report)
        if report["reason"] == "broad_local_connection":
            accepted.append((pair, report))
        elif (report["reason"] == "framed_portal"
              or (report["supported_samples"] and not report["open_samples"]
                  and report["blocked_samples"] * resolution >= .30 - 1e-9
                  and report["blocked_samples"] >= .80 * report["supported_samples"])):
            barriers.add(pair)

    accepted.sort(key=lambda row: (-row[1]["clear_width_m"], row[0]))
    forbidden_candidates = pairs.keys() | thick_walls | protected_pairs
    result, audit["accepted_merges"] = _apply_merges(labels, forbidden_candidates, accepted)
    return finish("completed", result)


def review_boundaries(working, labels_xy, initial, walls, origin, resolution, log=None, *,
                      height_rasters=None, protected_pairs=()):
    """Review local boundaries without painting walls or unobserved cells."""
    started = time.perf_counter()
    labels = np.asarray(labels_xy)
    protected_pairs = {tuple(sorted(pair)) for pair in protected_pairs}
    result, audit, barriers, evidence = _review_pass(
        working, labels, initial, walls, origin, resolution, height_rasters, protected_pairs,
    )
    evaluated = len(audit["evaluated_pairs"])
    if resolution < .10 - 1e-9 and audit["status"] == "completed":
        # A merge exposes a longer shared interface to the next pass. Carry
        # measured separators through each label renumbering.
        passes = audit["staggered_reviews"] = []
        previous_merges = audit["accepted_merges"]
        while True:
            if barriers and previous_merges:
                mapping = np.full(int(labels.max()) + 1, -1, dtype=labels.dtype)
                assigned = labels >= 0
                mapping[labels[assigned]] = result[assigned]
                barriers = {tuple(sorted((int(mapping[a]), int(mapping[b])))) for a, b in barriers
                            if 0 <= a < len(mapping) and 0 <= b < len(mapping)}
            labels = result
            result, current, barriers, _ = _review_pass(
                working, labels, initial, walls, origin, resolution, evidence, barriers, staggered=True,
            )
            passes.append(current)
            previous_merges = current["accepted_merges"]
            audit["accepted_merges"] += previous_merges
            evaluated += len(current["evaluated_pairs"])
            if not previous_merges:
                break
    audit["seconds"] = round(time.perf_counter() - started, 4)
    if log is not None:
        log(f"连通复核 {audit['accepted_merges']} 次合并，检查 {evaluated} 对，{audit['seconds']:.2f} s")
    return result, audit
