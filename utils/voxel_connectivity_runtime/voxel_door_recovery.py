"""Validate measured doorway proposals against full-label XY connectivity."""

from collections import Counter
from itertools import chain, combinations
import math
import time

import numpy as np
from scipy import ndimage

from .voxel_door_candidates import propose_door_cuts


_MIN_INTERIOR_AREA = 1.8
_MIN_INTERIOR_WIDTH = .8
_MIN_CORE_RADIUS = .35
_MIN_CORE_AREA = .18


def _ordered_cuts(candidates, labels):
    """Group valid, distinct proposals in deterministic evidence order."""
    ordered = sorted(candidates, key=lambda candidate: (
        candidate.get('kind') == 'corner', -candidate['score'], candidate['room'],
    ))
    rooms, seen = {}, set()
    for candidate in ordered:
        line = np.asarray(candidate['cut'])
        room = int(candidate['room'])
        score, width = float(candidate['score']), float(candidate['width_m'])
        if (room < 0 or line.ndim != 2 or line.shape[1] != 2 or len(line) < 2
                or line.dtype.kind not in 'iu' or (line < 0).any() or (line >= labels.shape).any()
                or not np.isfinite(score) or not np.isfinite(width) or width <= 0):
            continue
        line = line.astype(np.intp)
        if np.any(np.max(np.abs(np.diff(line, axis=0)), axis=1) != 1):
            continue
        active = line[labels[tuple(line.T)] == room]
        if not len(active):
            continue
        continuation = np.asarray(candidate.get('wall_continuation', np.empty((0, 2), np.intp)))
        if (continuation.ndim == 2 and continuation.shape[1] == 2 and continuation.dtype.kind in 'iu'
                and (continuation >= 0).all() and (continuation < labels.shape).all()):
            continuation = np.unique(continuation.astype(np.intp), axis=0)
            continuation = continuation[labels[tuple(continuation.T)] == room]
            aperture = set(map(tuple, line))
            keep = np.array([tuple(cell) not in aperture for cell in continuation], dtype=bool)
            continuation = continuation[keep]
        else:
            continuation = np.empty((0, 2), np.intp)
        key = (room, tuple(sorted(map(tuple, active))), tuple(map(tuple, continuation)))
        if key in seen:
            continue
        seen.add(key)
        rooms.setdefault(room, []).append(dict(line=line, active=active, continuation=continuation,
                                               score=score, width_m=width,
                                               kind=candidate.get('kind', 'collinear')))
    return rooms


def _room_boxes(labels, rooms):
    # Find extents using compact proposal IDs, even when the source IDs are sparse.
    ids = np.asarray(sorted(rooms), dtype=labels.dtype)
    compact = np.searchsorted(ids, labels)
    np.minimum(compact, len(ids) - 1, out=compact)
    compact[ids[compact] != labels] = -1
    compact += 1
    return dict(zip(ids, ndimage.find_objects(compact)))


def _interior(component, seen, walls, resolution, min_area):
    observed = component & seen
    area = float(observed.sum() * resolution ** 2)
    width = float(min(component.shape) * resolution)
    if area < min_area - 1e-9 or width < _MIN_INTERIOR_WIDTH - 1e-9:
        return None

    # Close only small raster holes for measuring the interior. This does not
    # add pixels to the final labels or participate in the topology test.
    radius = max(1, round(.10 / resolution))
    padding = radius + 1
    window = np.s_[padding:-padding, padding:-padding]
    padded = np.pad(observed, padding)
    closed = ndimage.binary_closing(padded, iterations=radius)[window]
    measured = (observed | closed) & ~walls
    clearance = ndimage.distance_transform_edt(np.pad(measured, 1))[1:-1, 1:-1] * resolution
    core_area = float(np.count_nonzero(clearance >= _MIN_CORE_RADIUS - 1e-9) * resolution ** 2)
    if core_area < _MIN_CORE_AREA - 1e-9:
        return None
    return dict(observed_area_m2=round(area, 3), width_m=round(width, 3), core_area_m2=round(core_area, 3))


def _corridor_direction(components, room, line, width, resolution):
    """Identify a narrow straight continuation immediately beyond a cut."""
    center = (line[0] + line[-1]) / 2
    tangent = (line[-1] - line[0]).astype(float)
    length = np.linalg.norm(tangent)
    if not length:
        return 0
    tangent /= length
    normal = np.array([-tangent[1], tangent[0]])
    radius = int(math.ceil(3.2 / resolution))
    low = np.maximum(np.floor(center).astype(int) - radius, 0)
    high = np.minimum(np.ceil(center).astype(int) + radius + 1, components.shape)
    box = tuple(slice(a, b) for a, b in zip(low, high))
    xy = (np.argwhere(components[box] == room) + low - center) * resolution
    longitudinal = xy @ normal
    lateral = xy @ tangent
    side = 1 if np.count_nonzero(longitudinal > .65) >= np.count_nonzero(longitudinal < -.65) else -1
    depth = longitudinal * side
    run = (depth >= .65) & (depth <= 2.75) & (np.abs(lateral) <= 3.)
    if np.count_nonzero(run & (depth >= 2.20)) * resolution ** 2 < .16:
        return 0
    low, center, high = np.quantile(lateral[run], [.05, .5, .95])
    narrow = high - low + resolution <= max(1.65, width + .30) + 1e-9
    return side if narrow and abs(center) <= .45 else 0


def _label_without(room_mask, result, cells, audit):
    coords = tuple(cells.T)
    audit['topology_trials'] += 1
    audit['topology_cells'] += room_mask.size
    room_mask[coords] = False
    count = ndimage.label(room_mask, output=result)
    room_mask[coords] = True
    return count


def _trial(components, count, room_mask, result, cuts, seen, walls, resolution, min_area, audit,
           continuation=None):
    """Assess whether one or two lines add exactly one stable partition."""
    apertures = [cut['local_active'] for cut in cuts]
    segments = apertures if continuation is None else [*apertures, continuation]
    cells = np.concatenate(segments)
    parents = np.unique(components[tuple(cells.T)])
    if len(parents) != 1 or parents[0] == 0:
        return None, 'overlapping_or_inactive_cut', 0
    parent = int(parents[0])
    number = _label_without(room_mask, result, cells, audit)
    if number == count:
        return None, 'does_not_disconnect', parent
    if number != count + 1:
        return None, 'unstable_component_count', parent
    if continuation is not None:
        # Keep the measured walls fixed and prove each aperture is necessary.
        # A restoration alone must never justify clearing an open doorway.
        for index in range(len(apertures)):
            partial = np.concatenate([continuation, *apertures[:index], *apertures[index + 1:]])
            if _label_without(room_mask, result, partial, audit) != count:
                return None, 'nonessential_portal', parent
        _label_without(room_mask, result, cells, audit)
        walls = walls.copy()
        walls[tuple(continuation.T)] = True
    children = np.flatnonzero(np.bincount(result[components == parent])[1:]) + 1
    if len(children) != 2:
        return None, 'unstable_component_count', parent

    boxes = ndimage.find_objects(result)
    evidence = []
    for child in children:
        box = boxes[child - 1]
        stable = _interior(result[box] == child, seen[box], walls[box], resolution, min_area)
        if stable is None:
            return None, 'insufficient_stable_interior', parent
        evidence.append(stable)
    for cut in cuts:
        directions = [_corridor_direction(result, child, cut['local_line'], cut['width_m'], resolution)
                      for child in children]
        if directions[0] * directions[1] == -1:
            return None, 'straight_corridor_continuation', parent
    return evidence, 'accepted', parent


def _next_split(components, count, room_mask, result, remaining, seen, walls, resolution, min_area, audit, rejected):
    redundant = []
    for index, cut in enumerate(remaining):
        evidence, reason, parent = _trial(components, count, room_mask, result, [cut], seen, walls,
                                          resolution, min_area, audit)
        if evidence is not None:
            return evidence, (index,), None
        rejected[reason] += 1
        if reason == 'does_not_disconnect':
            redundant.append((index, parent))

    # A room accessible through two measured portals requires both. Only
    # try pairs that individually failed to disconnect the same component.
    repair_pairs = []
    for (a, parent_a), (b, parent_b) in combinations(redundant, 2):
        first, second = remaining[a], remaining[b]
        separation = np.linalg.norm(first['center'] - second['center']) * resolution
        if parent_a != parent_b or separation < .45 - 1e-9:
            continue
        evidence, reason, _ = _trial(components, count, room_mask, result, [first, second], seen, walls,
                                     resolution, min_area, audit)
        if evidence is not None:
            return evidence, (a, b), None
        rejected[reason] += 1
        if reason == 'does_not_disconnect' and (len(first['local_continuation']) or len(second['local_continuation'])):
            repair_pairs.append((a, b))

    # Ordinary door cuts take precedence. Only try a measured continuation
    # when the corresponding aperture trial left a path around its banks.
    repair_singles = ((index,) for index, _ in redundant if len(remaining[index]['local_continuation']))
    for indices in chain(repair_singles, repair_pairs):
        cuts = [remaining[index] for index in indices]
        continuation = np.unique(np.concatenate([cut['local_continuation'] for cut in cuts]), axis=0)
        aperture = {tuple(cell) for cut in cuts for cell in cut['local_line']}
        if any(tuple(cell) in aperture for cell in continuation):
            rejected['continuation_overlaps_aperture'] += 1
            continue
        evidence, reason, _ = _trial(components, count, room_mask, result, cuts, seen, walls,
                                     resolution, min_area, audit, continuation=continuation)
        if evidence is not None:
            return evidence, indices, continuation
        rejected[reason] += 1
    return None, (), None


def recover_door_splits(labels, walls, observed, horizontal, vertical, z_step, resolution, min_room_area):
    """Return room labels, protected doorway cuts, restored walls, and audit.

    The largest surviving component keeps its old ID. Other components receive
    new IDs above the previous maximum. Only accepted cut and measured wall
    pixels owned by the old label are cleared. Existing walls and unlabeled
    cells stay unassigned.
    """
    started = time.perf_counter()
    labels = np.asarray(labels)
    walls, observed = np.asarray(walls, dtype=bool), np.asarray(observed, dtype=bool)
    if (labels.ndim != 2 or labels.dtype.kind != 'i' or labels.dtype.itemsize < 4
            or walls.shape != labels.shape or observed.shape != labels.shape
            or np.shape(horizontal) != labels.shape or np.shape(vertical) != labels.shape
            or not np.isfinite(resolution) or resolution <= 0 or not np.isfinite(z_step) or z_step <= 0
            or not np.isfinite(min_room_area) or min_room_area < 0):
        raise ValueError('Door recovery requires compatible signed labels, XY evidence, and metric thresholds')
    protected = np.zeros(labels.shape, bool)
    restored = np.zeros(labels.shape, bool)
    output = labels
    rejected = Counter()
    audit = dict(status='completed', accepted_door_count=0, accepted_corner_door_count=0,
                 accepted_split_count=0, accepted_component_splits=0,
                 topology_trials=0, topology_cells=0, protected_pairs=[], accepted=[], restored_wall_pixel_count=0)

    def finish(status):
        audit.update(status=status, rejected=dict(rejected), seconds=round(time.perf_counter() - started, 4))
        return output, protected, restored, audit

    largest_id = int(labels.max(initial=-1))
    if largest_id < 0:
        return finish('no_rooms')

    candidates, proposal = propose_door_cuts(labels, walls, observed, horizontal, vertical, z_step, resolution)
    audit['proposal'] = proposal
    audit['candidate_count'] = len(candidates)
    groups = _ordered_cuts(candidates, labels)
    if not groups:
        return finish('no_candidates')
    boxes = _room_boxes(labels, groups)
    next_id = largest_id + 1
    min_area = max(_MIN_INTERIOR_AREA, min_room_area)
    for room, cuts in groups.items():
        box = boxes[room]
        original = labels[box] == room
        components, count = ndimage.label(original)
        if count != 1:
            rejected['disconnected_original_label'] += 1
            continue
        offset = np.array([part.start for part in box])
        for cut in cuts:
            cut['local_line'] = cut['line'] - offset
            cut['local_active'] = cut['active'] - offset
            cut['local_continuation'] = cut['continuation'] - offset
            cut['center'] = cut['local_line'].mean(0)
        accepted = []
        room_mask = original.copy()
        trial_result = np.empty_like(components)
        while cuts:
            evidence, indices, continuation = _next_split(components, count, room_mask, trial_result, cuts,
                observed[box], walls[box], resolution, min_area, audit, rejected)
            if evidence is None:
                break
            components, trial_result = trial_result, components
            count += 1
            selected = [cuts[index] for index in indices]
            for cut in selected:
                room_mask[tuple(cut['local_active'].T)] = False
            if continuation is not None:
                room_mask[tuple(continuation.T)] = False
                continuation = continuation + offset
            accepted.append((selected, evidence, continuation))
            cuts = [cut for index, cut in enumerate(cuts) if index not in indices]
        if accepted:
            required_id = next_id + count - 2
            dtype = np.dtype(np.int64) if required_id > np.iinfo(output.dtype).max else output.dtype
            if output is labels or dtype != output.dtype:
                output = output.astype(dtype, copy=True)
            sizes = np.bincount(components.ravel())[1:]
            largest = int(np.argmax(sizes)) + 1
            mapping = np.full(count + 1, -1, dtype=dtype)
            component_ids = np.arange(1, count + 1)
            mapping[component_ids[component_ids != largest]] = np.arange(next_id, next_id + count - 1)
            mapping[largest] = room
            next_id += count - 1
            output_room = output[box]
            output_room[original] = mapping[components[original]]
            siblings = sorted(map(int, mapping[1:]))
            audit['accepted_split_count'] += 1
            audit['accepted_component_splits'] += len(accepted)
            audit['protected_pairs'].extend([a, b] for a, b in combinations(siblings, 2))
            for selected, evidence, continuation in accepted:
                lines = []
                for cut in selected:
                    indices = tuple(cut['line'].T)
                    allowed = (labels[indices] == room) | (labels[indices] < 0)
                    protected[tuple(cut['line'][allowed].T)] = True
                    lines.append(dict(kind=cut['kind'], width_m=round(cut['width_m'], 3),
                                      score=round(cut['score'], 3), removed_pixels=len(cut['active']),
                                      aperture_endpoints_xy=cut['line'][[0, -1]].tolist()))
                cells = []
                if continuation is not None:
                    restored[tuple(continuation.T)] = True
                    cells = continuation.tolist()
                    audit['restored_wall_pixel_count'] += len(cells)
                audit['accepted'].append(dict(source_room=room, result_rooms=siblings, doors=lines,
                                              interiors=evidence, restored_wall_cells=cells))
                audit['accepted_door_count'] += len(selected)
                audit['accepted_corner_door_count'] += sum(cut['kind'] == 'corner' for cut in selected)
    return finish('completed')
