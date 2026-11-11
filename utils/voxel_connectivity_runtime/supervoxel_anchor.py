"""Web adapter for the Li et al. (2025) supervoxel/anchor-pixel pipeline."""

from collections import deque
from numbers import Real
from pathlib import Path
import json
import time

import numpy as np
from PIL import Image
from scipy import ndimage

from .li_anchor_walls import wall_boundaries
from .li_supervoxels import initial_partition
from utils.potential_field_runtime.pointcloud import room_color


METHOD_VERSION = "li-supervoxel-anchor-v1"
DEFAULT_PARAMS = {
    "sa_resolution": .1,
    "sa_seed_size": 1.,
    "sa_max_jump": 1.1,
    "sa_min_room_area": 1.,
    "min_room_points": 0,
}
_RANGES = {
    "sa_resolution": (.03, .5),
    "sa_seed_size": (.2, 5.),
    "sa_max_jump": (.1, 5.),
    "sa_min_room_area": (0., 1_000_000.),
    "min_room_points": (0, 100_000_000),
}
_MAX_PIXELS = 4_000_000
_MAX_WORK_POINTS = 2_500_000
_VC_MAX_PIXELS = 16_000_000
_VC_MAX_WORK_POINTS = 8_000_000
_MIN_FINE_SEED_RETENTION = .25
_CHUNK = 500_000
_DOWNSAMPLE_M = .05
_MIN_ROOM_WIDTH_M = .8
_SURFACE_RADIUS_M = .30


def parameters(params):
    values = {}
    for key, default in DEFAULT_PARAMS.items():
        value = params.get(key, default)
        lower, upper = _RANGES[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not lower <= value <= upper:
            raise ValueError(f"{key} 需在 {lower:g}–{upper:g} 之间")
        if key == "min_room_points" and int(value) != value:
            raise ValueError("min_room_points 需为整数")
        values[key] = int(value) if key == "min_room_points" else float(value)
    return values


def supervoxel_anchor_status():
    return dict(ready=True, reason="", defaults=DEFAULT_PARAMS.copy())


def _combine_voxels(keys, counts, sums):
    unique, inverse = np.unique(keys, return_inverse=True)
    return (
        unique,
        np.bincount(inverse, weights=counts),
        np.column_stack([np.bincount(inverse, weights=sums[:, axis]) for axis in range(3)]),
    )


def _voxel_centroids(points, low, high, step, max_work_points=_MAX_WORK_POINTS):
    # Sort only observed voxels; the enclosing three-dimensional grid is sparse.
    extent = (high - low) / step
    if np.any(extent >= 2**52):
        raise ValueError("场景跨度过大，请核对点云坐标单位及离群点")
    nx, ny, nz = (int(length) + 1 for length in extent)
    if nx * ny * nz > np.iinfo(np.int64).max:
        raise ValueError("场景跨度过大，请核对点云坐标单位及离群点")
    groups = []
    pending = 0
    for start in range(0, len(points), _CHUNK):
        block = points[start:start + _CHUNK]
        voxels = np.floor((block - low) / step).astype(np.int64)
        keys = (voxels[:, 0] * ny + voxels[:, 1]) * nz + voxels[:, 2]
        unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        sums = np.column_stack([np.bincount(inverse, weights=block[:, axis]) for axis in range(3)])
        groups.append((unique, counts, sums))
        pending += len(unique)
        if pending >= 2 * max_work_points:
            combined = _combine_voxels(*(np.concatenate(parts) for parts in zip(*groups)))
            pending = len(combined[0])
            if pending > max_work_points:
                return None
            groups = [combined]
    if len(groups) == 1:
        _, counts, sums = groups[0]
    else:
        _, counts, sums = _combine_voxels(*(np.concatenate(parts) for parts in zip(*groups)))
    if len(counts) > max_work_points:
        return None
    return sums / counts[:, None]


def _sample_geometry(points, low, high, max_work_points=_MAX_WORK_POINTS):
    step = _DOWNSAMPLE_M
    for _ in range(6):
        working = _voxel_centroids(points, low, high, step, max_work_points)
        if working is not None:
            return working, step
        step *= 1.4
    raise ValueError("超体素工作点数超过保护上限，请缩小场景范围后重试")


def _grow_labels(labels, walkable):
    """Expand through unlabeled observed cells using four-neighbor distances."""
    unassigned = walkable & (labels < 0)
    sources = np.argwhere((labels >= 0) & ndimage.binary_dilation(unassigned))
    queue = deque(map(tuple, sources.tolist()))
    nx, ny = labels.shape
    while queue:
        x, y = queue.popleft()
        room = labels[x, y]
        for a, b in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= a < nx and 0 <= b < ny and walkable[a, b] and labels[a, b] < 0:
                labels[a, b] = room
                queue.append((a, b))


def _partition_map(footprint, observed, walls, doors, resolution, min_area):
    blocked = walls | doors
    walkable = observed & ~blocked
    components, count = ndimage.label(footprint & walkable)
    area = np.bincount(components.ravel())
    ids = np.full(count + 1, -1, dtype=np.int32)
    room_count = 0
    for label, slices in enumerate(ndimage.find_objects(components), start=1):
        if (area[label] * resolution**2 + 1e-9 >= min_area
                and min(part.stop - part.start for part in slices) * resolution + 1e-9 >= _MIN_ROOM_WIDTH_M):
            ids[label] = room_count
            room_count += 1
    label_map = ids[components]
    if room_count:
        _grow_labels(label_map, walkable)
    return label_map, room_count, count


def _assign_surfaces(labels, projected_regions, walls, doors, resolution, room_count):
    """Inherit an unambiguous nearby room without propagating through walls."""
    candidates = (labels < 0) & (projected_regions > 0) & ~doors
    radius = int(_SURFACE_RADIUS_M / resolution + 1e-9)
    if not room_count or not radius or not candidates.any():
        return 0
    x, y = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    neighborhood = (x*x + y*y) * resolution**2 <= _SURFACE_RADIUS_M**2 + 1e-9
    smallest = ndimage.minimum_filter(np.where(labels >= 0, labels, room_count),
                                       footprint=neighborhood, mode="constant", cval=room_count)
    largest = ndimage.maximum_filter(labels, footprint=neighborhood, mode="constant", cval=-1)
    candidates &= (largest >= 0) & (smallest == largest)

    known = labels >= 0
    traversable = (projected_regions > 0) & ~(walls | doors)
    reached = known
    for _ in range(radius):
        reachable = ndimage.binary_dilation(reached)
        reached = reachable & traversable
    candidates &= reachable

    region_count = int(projected_regions.max()) + 1
    first = np.full(region_count, room_count, dtype=np.int32)
    last = np.full(region_count, -1, dtype=np.int32)
    np.minimum.at(first, projected_regions[known], labels[known])
    np.maximum.at(last, projected_regions[known], labels[known])
    candidate_regions = projected_regions[candidates]
    nearby = largest[candidates]
    consensus = (first[candidate_regions] == nearby) & (last[candidate_regions] == nearby)
    consensus |= walls[candidates] & (last[candidate_regions] < 0)
    candidates[candidates] = consensus
    labels[candidates] = largest[candidates]
    return int(np.count_nonzero(candidates))


def _project(points, label_map, origin, resolution):
    labels = np.empty(len(points), dtype=np.int32)
    for start in range(0, len(points), _CHUNK):
        block = slice(start, start + _CHUNK)
        xy = np.floor((points[block, :2] - origin) / resolution).astype(np.int64)
        labels[block] = label_map[xy[:, 0], xy[:, 1]]
    return labels


def _observed_input_columns(points, shape, origin, resolution):
    observed = np.zeros(shape, dtype=bool)
    for start in range(0, len(points), _CHUNK):
        xy = np.floor((points[start:start + _CHUNK, :2] - origin) / resolution).astype(np.int64)
        observed[xy[:, 0], xy[:, 1]] = True
    return observed


def _fallback_seed_labels(points, working, initial, build_coarse, origin, resolution, coarse_resolution,
                          values, walls, verify_doors):
    from .voxel_seed_support import _coarse_indices, refine_seed_labels

    coarse = build_coarse(coarse_resolution)
    observed = coarse["observed_footprint"]
    coarse_walls, _ = wall_boundaries(
        working, initial["normals"], origin, coarse_resolution, observed.shape, observed, values["sa_max_jump"],
    )
    if verify_doors is not None:
        verify_doors(working, coarse, coarse_walls, origin, coarse_resolution)
    coarse_doors = coarse["door_columns"]
    coarse_labels, coarse_count, _ = _partition_map(
        coarse["footprint"], observed, coarse_walls, coarse_doors, coarse_resolution, values["sa_min_room_area"],
    )
    diagnostics = dict(coarse_resolution_m=coarse_resolution, coarse_room_count=coarse_count)
    if verify_doors is not None:
        diagnostics["initial_door_verification"] = coarse["door_verification"]
    if not coarse_count:
        return None, diagnostics
    input_observed = _observed_input_columns(points, walls.shape, origin, resolution)
    labels, walls, doors, refinement = refine_seed_labels(
        coarse_labels, coarse_walls, coarse_doors, input_observed, walls, initial["door_columns"],
        resolution, coarse_resolution, values["sa_min_room_area"],
    )
    diagnostics.update(refinement)
    room_count = int(labels.max()) + 1
    if not room_count:
        return None, diagnostics
    confirmed = None
    if verify_doors is not None:
        confirmed = initial["confirmed_door_columns"]
        coarse_confirmed = coarse["confirmed_door_columns"]
        if coarse_confirmed.any():
            indices = [_coarse_indices(fine, coarse_size, resolution, coarse_resolution)
                       for fine, coarse_size in zip(confirmed.shape, coarse_confirmed.shape)]
            covered = tuple(slice(0, len(axis)) for axis in indices)
            confirmed = confirmed.copy()
            confirmed[covered] |= coarse_confirmed[np.ix_(*indices)]
    return (labels, room_count, walls, doors, input_observed, confirmed), diagnostics


def run_supervoxel_anchor(xyz, params, output_dir, log=None, stage=None):
    """Run the CPU pipeline and return full-order point labels plus map artifacts."""
    return _run_anchor_pipeline(xyz, parameters(params), output_dir, log, stage)


def _run_anchor_pipeline(xyz, values, output_dir, log, stage, *,
                         method="supervoxel_anchor", method_version=METHOD_VERSION, review=None,
                         robust_seeds=False, verify_doors=None):
    log = log or (lambda message: None)
    stage = stage or (lambda name: None)
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    points = np.asarray(xyz)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 21:
        raise ValueError("超体素锚点分割至少需要 21 个有效的三维点")
    low = points.min(axis=0).astype(np.float64)
    high = points.max(axis=0).astype(np.float64)
    if not np.all(np.isfinite([low, high])):
        raise ValueError("点云包含非有限坐标")
    resolution = values["sa_resolution"]
    origin = low[:2] - resolution
    extents = (high[:2] - origin) / resolution
    max_pixels = _VC_MAX_PIXELS if robust_seeds else _MAX_PIXELS
    grid_limit_error = f"二维网格超过 {max_pixels // 10_000} 万像素，请增大地图分辨率或检查离群点"
    if np.any(extents >= max_pixels):
        raise ValueError(grid_limit_error)
    shape = tuple(int(size) + 2 for size in extents)
    if shape[0] * shape[1] > max_pixels:
        raise ValueError(grid_limit_error)

    stage("体素下采样")
    max_work_points = _VC_MAX_WORK_POINTS if robust_seeds else _MAX_WORK_POINTS
    working, step = _sample_geometry(points, low, high, max_work_points)
    if len(working) < 21:
        raise ValueError("体素下采样后有效点不足，无法估计局部几何")
    log(f"工作点 {len(working):,} / {len(points):,}，下采样体素 {step:.3f} m")
    stage("超体素初分割与门洞候选")
    coarse_resolution = max(.10, 2 * step, 1.5 * resolution)
    initial = initial_partition(
        working, origin, resolution, shape, values["sa_seed_size"], stage_log=log,
        fallback_resolution=coarse_resolution if robust_seeds else None,
    )
    build_coarse = initial.pop("fallback", None)
    stage("墙体证据与锚点连接")
    walls, wall_diagnostics = wall_boundaries(working, initial["normals"], origin, resolution,
                                              shape, initial["observed_footprint"], values["sa_max_jump"])
    if verify_doors is not None:
        stage("核验初始门洞")
        verify_doors(working, initial, walls, origin, resolution, cache_heights=True)
    stage("连通房间提取与三维回投影" if review is None else "提取初始房间")
    footprint = initial["footprint"]
    observed = initial["observed_footprint"]
    doors = initial["door_columns"]
    labels_xy, room_count, components = _partition_map(footprint, observed, walls, doors, resolution,
                                                       values["sa_min_room_area"])
    seed_diagnostics = None
    input_observed = None
    if build_coarse is not None:
        fine_stats = initial["diagnostics"]
        retained = fine_stats["footprint_pixels"] / max(1, fine_stats["projected_pixels"])
        seed_diagnostics = dict(mode="native", fine_room_count=room_count, fine_seed_retention=retained)
        if room_count == 0 or retained < _MIN_FINE_SEED_RETENTION:
            stage("粗栅格稳定种子与细图细化")
            attempts = []
            fine_supported_cells = int(np.count_nonzero(labels_xy >= 0))
            for scale in (1., 1.5, 2.25):
                result, diagnostics = _fallback_seed_labels(
                    points, working, initial, build_coarse, origin, resolution,
                    coarse_resolution * scale, values, walls, verify_doors,
                )
                attempts.append(diagnostics)
                if result is not None and diagnostics["fine_observed_seed_cells"] > fine_supported_cells:
                    labels_xy, room_count, walls, doors, input_observed, confirmed = result
                    initial["door_columns"] = doors
                    if confirmed is not None:
                        initial["confirmed_door_columns"] = confirmed
                    seed_diagnostics.update(diagnostics, mode="coarse_to_fine")
                    log(f"稳定种子：{diagnostics['coarse_resolution_m']:.3f} m/px 提取 "
                        f"{diagnostics['coarse_room_count']} 个区域，细化到 {resolution:.3f} m/px 后保留 {room_count} 个区域")
                    break
            seed_diagnostics["attempts"] = attempts
    del build_coarse
    review_stats = None
    protected_boundaries = doors
    if review is not None:
        stage("门洞补切与多高度连通复核")
        np.save(output / "initial_label_map.npy", labels_xy.T)
        initial_room_count = room_count
        labels_xy, recovered_doors, restored_walls, review_stats = review(
            working, labels_xy, initial, walls, origin, resolution, log=log)
        protected_boundaries = doors | recovered_doors | restored_walls
        room_count = int(labels_xy.max()) + 1
        review_stats.update(initial_room_count=initial_room_count, final_room_count=room_count)
        log(f"体素连通复核：初始 {initial_room_count} 个区域 → 保留 {room_count} 个")
        stage("三维回投影")
    surface_pixels = _assign_surfaces(labels_xy, initial["projected_regions"], walls, protected_boundaries,
                                      resolution, room_count)
    if input_observed is not None:
        labels_xy[~input_observed] = -1
    sizes = np.bincount(labels_xy[labels_xy >= 0], minlength=room_count)
    rooms = [dict(id=room, cell_count=int(size), area_m2=float(size * resolution**2))
             for room, size in enumerate(sizes)]
    labels = _project(points, labels_xy, origin, resolution)
    elapsed = time.perf_counter() - started
    transform = dict(width=shape[0], height=shape[1], origin=origin.tolist(), resolution=resolution,
                     row_order="y_up", method=method, preprocessing_version=method_version)
    warnings = []
    if step > _DOWNSAMPLE_M:
        warnings.append(f"工作点数达到保护阈值，下采样体素自动增至 {step:.3f} m。")
    if not rooms:
        warnings.append("没有区域达到房间面积与宽度要求。稀疏点云可尝试增大地图分辨率数值，再结合中间结构图检查。")
    stats = dict(method_version=method_version, native_room_count=len(rooms), inference_seconds=elapsed,
                 input_point_count=len(points), working_point_count=len(working), downsample_m=step,
                 map_dimensions=list(shape), map_resolution=resolution, min_room_width_m=_MIN_ROOM_WIDTH_M,
                 observed_pixel_count=int(np.count_nonzero(observed)), component_count=components,
                 surface_assigned_pixels=surface_pixels, surface_radius_m=_SURFACE_RADIUS_M,
                 supervoxels=initial["diagnostics"], anchor_walls=wall_diagnostics, warnings=warnings)
    if seed_diagnostics is not None:
        stats["seed_support"] = seed_diagnostics
    if verify_doors is not None:
        stats["initial_door_verification"] = initial["door_verification"]
    if review_stats is not None:
        stats["review"] = review_stats
        if review_stats["status"] == "unsupported_height_range":
            warnings.append("场景高度跨度超出复核范围，边界沿用初分割。")
    label_map = labels_xy.T
    np.save(output / "label_map.npy", label_map)
    label_map.astype("<i4", copy=False).tofile(output / "label_map.bin")
    colors = np.asarray([(80, 80, 80), *(room_color(room["id"]) for room in rooms)], dtype=np.uint8)
    Image.fromarray(colors[label_map[::-1] + 1]).save(output / "segmentation.png")
    structure = dict(observed=observed.T, initial_footprint=footprint.T, doors=doors.T, walls=walls.T)
    if verify_doors is not None:
        structure["confirmed_initial_doors"] = initial["confirmed_door_columns"].T
    if input_observed is not None:
        structure["input_observed"] = input_observed.T
    if review is not None:
        structure["recovered_doors"] = recovered_doors.T
        structure["restored_walls"] = restored_walls.T
    np.savez_compressed(output / "structure.npz", **structure)
    for name, payload in (("inference.json", stats), ("transform.json", transform)):
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                   encoding="utf-8")
    log(f"地图 {shape[0]} × {shape[1]}，连通分量 {components}，房间 {len(rooms)}")
    downloads = [dict(name=name, path=path) for name, path in (
        ("体素连通标签图 NPY" if review is not None else "超体素锚点标签图 NPY", "label_map.npy"),
        ("二维分割预览 PNG", "segmentation.png"),
        ("初分割、墙体及门洞 NPZ", "structure.npz"), ("地图坐标变换 JSON", "transform.json"),
        ("体素连通复核诊断 JSON" if review is not None else "超体素锚点诊断 JSON", "inference.json"),
    )]
    if review is not None:
        downloads.append(dict(name="复核前二维标签 NPY", path="initial_label_map.npy"))
    return labels, dict(statistics=stats, transform=transform, rooms=rooms, label_map_path="label_map.bin",
                        downloads=downloads)
