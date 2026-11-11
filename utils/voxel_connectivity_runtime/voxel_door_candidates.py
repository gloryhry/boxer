"""Missed-door proposals from measured wall banks and local floors.

The input rasters encode single returns, so empty bins are only a weak
opening cue. A proposal also needs measured vertical walls at both jambs and
floor support through the aperture. At a corner, a perpendicular wall can
support one jamb when a coherent lintel spans the opening. The caller verifies
whole-room topology before changing labels.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from .li_anchor_walls import _line_pixels


_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
_MIN_DOOR_M = .55
_MAX_DOOR_M = 1.55
_DUPLICATE_DISTANCE_M = .28


def _bits(low, high):
    low, high = max(0, low), min(63, high)
    return np.uint64(((1 << (high + 1)) - (1 << low)) if high >= low else 0)


def _run_starts(bits, length, z_step, *, bridge_missing=True):
    # Bridge one thin missing slice, then locate contiguous vertical runs.
    bridged = bits
    if bridge_missing and z_step <= .20:
        bridged = bits | ((bits << np.uint64(1)) & (bits >> np.uint64(1)))
    starts = bridged.copy()
    for offset in range(1, length):
        starts &= bridged >> np.uint64(offset)
    return starts


def _bank_seeds(vertical, scope, z_step):
    active = np.flatnonzero((vertical != 0) & scope)
    values = vertical.ravel()[active]
    starts = _run_starts(values, math.ceil(.95 / z_step), z_step)
    return np.column_stack(np.unravel_index(active[starts != 0], vertical.shape))


def _directional_pairs(seeds, labels, walls, resolution):
    """Yield consecutive measured bank pixels on each raster line."""
    for dx, dy in _DIRECTIONS:
        line = seeds[:, 0] * dy - seeds[:, 1] * dx
        along = seeds[:, 0] if dx else seeds[:, 1]
        order = np.lexsort((along, line))
        a, b = order[:-1], order[1:]
        stride = math.hypot(dx, dy) * resolution
        width = (along[b] - along[a] - 1) * stride
        valid = (line[a] == line[b]) & (width >= _MIN_DOOR_M - 1e-9) & (width <= _MAX_DOOR_M + 1e-9)
        a, b, width = seeds[a[valid]], seeds[b[valid]], width[valid]
        if not len(a):
            continue
        midpoint = (a + b) // 2
        room = labels[tuple(midpoint.T)]
        keep = room >= 0
        a, b, width, room = a[keep], b[keep], width[keep], room[keep]
        known = walls[tuple(a.T)].astype(np.int8) + walls[tuple(b.T)]
        for support, aperture, owner, first, last in zip(known, width, room, a, b):
            yield float(support), float(aperture), int(owner), first, last


def _side_points(aperture, normal, labels, observed, room, resolution):
    depths = np.array([.24, .39, .57])
    steps = np.rint(depths[:, None] * normal / resolution).astype(int)
    points = (aperture[None, :, :] + steps[:, None, :]).reshape(-1, 2)
    inside = ((points >= 0) & (points < labels.shape)).all(axis=1)
    points = np.unique(points[inside], axis=0)
    supported = (labels[tuple(points.T)] == room) & observed[tuple(points.T)]
    return points[supported] if len(points) and supported.mean() >= .60 else points[:0]


def _grounded_wall(bits, floor, z_step):
    """Check for one measured continuous lower-to-middle wall run per XY cell."""
    length = math.ceil(1.30 / z_step)
    starts = _run_starts(bits, length, z_step)
    first = max(0, floor + math.ceil(1.65 / z_step) - length + 1)
    last = min(64 - length, floor + math.floor(.65 / z_step))
    solid = np.zeros(bits.shape, bool)
    for start in range(first, last + 1):
        possible = (starts & (np.uint64(1) << np.uint64(start))) != 0
        if not possible.any():
            continue
        occupied = ((bits[possible, None] >> np.arange(start, start + length, dtype=np.uint64)) & 1).sum(axis=1)
        solid[possible] |= occupied >= math.ceil(.72 * length)
    return solid


def _bank_span(endpoint, outward, vertical, scope, floor, z_step, resolution, *, require_contact=False):
    reach = max(2, math.ceil(.72 / resolution))
    begin = np.maximum(endpoint - reach, 0)
    end = np.minimum(endpoint + reach + 1, vertical.shape)
    region = tuple(slice(a, b) for a, b in zip(begin, end))
    pixels = np.argwhere((vertical[region] != 0) & scope[region]) + begin
    offset = (pixels - endpoint) * resolution
    forward = offset @ outward
    across = offset @ np.array([-outward[1], outward[0]])
    inline = ((forward >= -.05) & (forward <= .74)
              & (np.abs(across) <= max(.12, .75 * resolution)))
    pixels, offset, forward = pixels[inline], offset[inline], forward[inline]
    grounded = _grounded_wall(vertical[tuple(pixels.T)], floor, z_step)
    if require_contact:
        # For a corner substitute the return wall must meet the exact jamb.
        # A separate, nearby perpendicular line cannot support the opening.
        local = pixels - begin
        connected = np.zeros(tuple(end - begin), bool)
        connected[tuple(local[grounded].T)] = True
        groups, _ = ndimage.label(connected, structure=np.ones((3, 3), bool))
        at_jamb = groups[tuple(endpoint - begin)]
        if not at_jamb:
            return 0.
        grounded &= groups[tuple(local.T)] == at_jamb
    if grounded.sum() < 2:
        return 0.
    offsets = offset[grounded]
    centered = offsets - offsets.mean(axis=0)
    _, axes = np.linalg.eigh(centered.T @ centered)
    if abs(axes[:, -1] @ outward) < math.cos(math.radians(22)):
        return 0.
    positions = np.unique(np.round(forward[grounded], 5))
    span = float(np.ptp(positions))
    if (positions[0] > max(.17, 1.1 * resolution)
            or np.max(np.diff(positions), initial=0.) > max(.25, 1.7 * resolution)
            or span < max(.36, .90 * resolution)):
        return 0.
    return span


def _coherent_lintel_fraction(hits, floor, z_step, width):
    """Find a measured lintel at a common height crossing the opening centre."""
    first = floor + math.ceil(1.60 / z_step)
    last = min(63, floor + math.floor(2.45 / z_step))
    levels = np.arange(first, last + 1, dtype=np.uint64)
    present = ((hits[:, None] >> levels) & 1) != 0
    if z_step <= .20 and len(levels) > 1:
        # A return on a bin boundary may fall into either adjacent bin.
        present = present[:, :-1] | present[:, 1:]
    pitch = width / len(hits)
    midpoint = (len(hits) - 1) / 2
    largest = 0
    for profile in present.T:
        measured = np.flatnonzero(profile)
        if len(measured) < 2:
            continue
        breaks = np.flatnonzero(np.diff(measured) * pitch > max(.20, pitch) + 1e-9) + 1
        for span in np.split(measured, breaks):
            if span[0] < midpoint < span[-1] and len(span) * pitch >= .30 - 1e-9:
                largest = max(largest, len(span))
    return largest / len(hits)


def _has_corner_lintel(walls, horizontal, vertical, aperture, normal, floor, z_step, width, resolution, passage):
    if _coherent_lintel_fraction(vertical[tuple(aperture.T)], floor, z_step, width) >= .50:
        return True
    # The clear cut and its measured lintel can land in adjacent grid rows
    # when the door frame and nearby wall face share one raster cell.
    offset = np.rint(normal).astype(int)
    if np.linalg.norm(offset) * resolution > .15 + 1e-9:
        return False
    floor_bits = _bits(floor - 1, floor + 1)
    for direction in (-1, 1):
        line = aperture + direction * offset
        if ((line < 0) | (line >= vertical.shape)).any():
            continue
        at = tuple(line.T)
        hits = vertical[at]
        # Only recover evidence hidden by the wall projection. A nearby
        # overhead beam above an otherwise open row is not enough.
        supported = walls[at] & ((horizontal[at] & floor_bits) != 0) & ((hits & passage) == 0)
        hits[~supported] = 0
        if _coherent_lintel_fraction(hits, floor, z_step, width) >= .50:
            return True
    return False


def _wall_continuation(cut, labels, walls, observed, horizontal, vertical, floor, z_step, room,
                       *, collinear_sides=(True, True)):
    """Find measured lower wall pixels along the two outward jamb rays."""
    direction = np.sign(cut[-1] - cut[0])
    floor_surface = _bits(floor - 1, floor + 1)
    first_hit = _bits(floor - max(1, math.floor(.20 / z_step)),
                      floor + max(1, math.floor(.26 / z_step)))
    run_length = max(2, math.ceil(.75 / z_step - 1e-9))
    extension = []
    nx, ny = labels.shape
    for endpoint, outward, collinear in zip((cut[0], cut[-1]), (-direction, direction), collinear_sides):
        if not collinear:
            continue
        x, y = map(int, endpoint)
        dx, dy = map(int, outward)
        x, y = x + dx, y + dy
        while 0 <= x < nx and 0 <= y < ny:
            owner = labels[x, y]
            if owner >= 0 and owner != room:
                break
            if not walls[x, y]:
                if not observed[x, y] or horizontal[x, y] & floor_surface:
                    break
                runs = _run_starts(vertical[x, y], run_length, z_step, bridge_missing=False)
                if not runs & first_hit:
                    break
                if owner == room:
                    extension.append((x, y))
            x, y = x + dx, y + dy
    return np.asarray(extension, dtype=np.intp).reshape(-1, 2)


def _measure_gap(pair, labels, walls, observed, horizontal, vertical, scope, z_step, resolution):
    # Import at review time so the shared local floor estimator remains
    # usable by the caller even if it imports this module during startup.
    from .voxel_connectivity_review import _floor_level

    known, width, room, a, b = pair
    cut = _line_pixels(a, b)
    aperture = cut[1:-1]
    at = tuple(aperture.T)
    aperture_rooms = labels[at]
    active = aperture_rooms == room
    seen = observed[at]
    if (len(aperture) < 2 or seen.mean() < .75 or ((aperture_rooms >= 0) & ~active).any()):
        return None, "unobserved_or_already_separated"
    wall_pixels = walls[at]
    partly_masked = active.mean() < .80 or wall_pixels.mean() > .20
    if partly_masked and (not active.any() or (active & wall_pixels).any()
                          or not (active | (wall_pixels & seen)).all()):
        return None, "unobserved_or_already_separated"
    tangent = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-tangent[1], tangent[0]])
    sides = [_side_points(aperture, sign * normal, labels, observed, room, resolution) for sign in (-1, 1)]
    if min(map(len, sides)) < 3:
        return None, "insufficient_passage_support"
    levels = [_floor_level(horizontal, vertical, points[:, 0], points[:, 1], np.ones(len(points), bool), z_step)
              for points in sides]
    if any(level is None for level in levels):
        return None, "unknown_floor"
    (floor_a, top_a), (floor_b, top_b) = levels
    if abs(floor_a - floor_b) * z_step > .25 + z_step / 2 or min(top_a - floor_a, top_b - floor_b) * z_step < 1.60:
        return None, "different_or_unconfirmed_floors"
    floor = int(round((floor_a + floor_b) / 2))
    floor_bits = _bits(floor - 1, floor + 1)
    on_floor = (horizontal[at] & floor_bits) != 0
    if on_floor.mean() < .18:
        return None, "no_floor_through_aperture"

    # Require a low and a middle passage band without repeated nonhorizontal
    # hits in the aperture. A fixture above them may supply a lintel cue.
    lower = _bits(floor + math.ceil(.25 / z_step), floor + math.floor(.80 / z_step))
    middle = _bits(floor + math.ceil(.85 / z_step), floor + math.floor(1.52 / z_step))
    hits = vertical[at]
    clear = (hits & (lower | middle)) == 0
    clear_fraction = float(clear.mean())
    if clear_fraction < .65:
        return None, "obstructed_at_door_height"
    lintel = _bits(floor + math.ceil(1.60 / z_step), floor + math.floor(2.45 / z_step))
    if partly_masked:
        # The top-down wall mask can include a real lintel. Keep those cells
        # blocked, but let the remaining aperture complete the same cut.
        open_below = seen & on_floor & clear & ((hits & lintel) != 0)
        if ((wall_pixels & ~open_below).any()
                or _coherent_lintel_fraction(np.where(clear, hits, np.uint64(0)), floor, z_step, width) < .50):
            return None, "closed_barrier"
    spans = [
        _bank_span(endpoint, outward, vertical, scope, floor, z_step, resolution)
        for endpoint, outward in ((a, -tangent), (b, tangent))
    ]
    collinear_sides = tuple(span > 0 for span in spans)
    kind = "collinear"
    if not all(collinear_sides):
        if not any(collinear_sides):
            return None, "no_grounded_collinear_banks"
        short = collinear_sides.index(False)
        endpoint = (a, b)[short]
        perpendicular = max(
            _bank_span(endpoint, direction, vertical, scope, floor, z_step, resolution, require_contact=True)
            for direction in (-normal, normal)
        )
        if perpendicular < .45 - 1e-9:
            return None, "corner_without_perpendicular_wall"
        if not _has_corner_lintel(walls, horizontal, vertical, aperture, normal, floor, z_step, width, resolution,
                                  lower | middle):
            return None, "corner_without_coherent_lintel"
        kind = "corner"
        spans[short] = perpendicular
    lintel_fraction = float(((hits & lintel) != 0).mean())
    score = (2. * min(spans) + clear_fraction + .40 * float(on_floor.mean())
             + .30 * lintel_fraction + .12 * known - .25 * abs(width - 1.))
    continuation = _wall_continuation(cut, labels, walls, observed, horizontal, vertical, floor, z_step, room,
                                      collinear_sides=collinear_sides)
    return dict(room=room, cut=cut, wall_continuation=continuation, kind=kind, score=float(score), width_m=round(width, 4),
                bank_span_m=round(min(spans), 4), supported_passage_fraction=round(clear_fraction, 4),
                lintel_fraction=round(lintel_fraction, 4)), "measured_portal"


def _deduplicate(proposed, resolution):
    """Keep the strongest representation of each geometrically repeated cut."""
    selected = []
    buckets = {}
    for candidate, priority in sorted(
            proposed, key=lambda item: (item[0]["kind"] == "corner", -item[0]["score"], item[1])):
        a, b = candidate["cut"][[0, -1]]
        midpoint = (a + b) * (.5 * resolution)
        tangent = (b - a) / np.linalg.norm(b - a)
        x, y = np.floor(midpoint / _DUPLICATE_DISTANCE_M).astype(int)
        room = candidate["room"]
        duplicate = any(
            np.linalg.norm(midpoint - center) <= _DUPLICATE_DISTANCE_M and abs(tangent @ direction) >= .85
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            for center, direction in buckets.get((room, x + dx, y + dy), ())
        )
        if not duplicate:
            selected.append(candidate)
            buckets.setdefault((room, x, y), []).append((midpoint, tangent))
    return selected


def propose_door_cuts(labels, walls, observed, horizontal, vertical, z_step, resolution):
    """Return candidate cuts in x-first XY order, leaving labels untouched."""
    report = dict(status="completed", candidate_count=0, raw_candidates=0, evaluated=0,
                  bank_seed_count=0, rejected={})

    if not labels.size or not (labels >= 0).any():
        report["status"] = "no_labeled_space"
        return [], report
    scope = observed | walls
    seeds = _bank_seeds(vertical, scope, z_step)
    report["bank_seed_count"] = len(seeds)
    if len(seeds) < 4:
        report["status"] = "no_wall_banks"
        return [], report
    proposed = []
    for pair in _directional_pairs(seeds, labels, walls, resolution):
        candidate, reason = _measure_gap(pair, labels, walls, observed, horizontal, vertical, scope, z_step, resolution)
        report["evaluated"] += 1
        if candidate is not None:
            known, width, room, first, last = pair
            priority = (-known, abs(width - 1.), room, *first, *last)
            proposed.append((candidate, priority))
        else:
            rejected = report["rejected"]
            rejected[reason] = rejected.get(reason, 0) + 1

    selected = _deduplicate(proposed, resolution)
    report.update(raw_candidates=report["evaluated"], candidate_count=len(selected))
    return selected, report
