"""Room partitioning with multilevel checks of uncertain boundaries."""

from functools import partial
import time

import numpy as np

from .supervoxel_anchor import DEFAULT_PARAMS, _run_anchor_pipeline, parameters
from .voxel_connectivity_review import _height_rasters, review_boundaries
from .voxel_door_recovery import recover_door_splits
from .voxel_initial_doors import confirmed_initial_door_pairs, verify_initial_doors


METHOD_VERSION = "voxel-connectivity-v10"


def voxel_connectivity_status():
    return dict(ready=True, reason="", defaults=DEFAULT_PARAMS.copy(), method_version=METHOD_VERSION)


def _verify_seed_doors(working, initial, walls, origin, resolution, *, cache_heights=False, log=None):
    """Check projected door hints, keeping uncertain barriers for seed partitioning."""
    raw = initial["door_columns"]
    rasters = None
    if raw.any():
        rasters = _height_rasters(working, initial["normals"], origin, resolution, walls.shape)
        if cache_heights:
            initial["review_height_rasters"] = rasters
    confirmed, contradicted, report = verify_initial_doors(
        raw, initial["observed_footprint"], walls, rasters, resolution,
    )
    seeds = initial.pop("door_seed_indices")
    initial["footprint"].flat[seeds] = contradicted.flat[seeds]
    initial["door_columns"] = raw & ~contradicted
    initial["confirmed_door_columns"] = confirmed
    initial["door_verification"] = report
    if cache_heights and log is not None:
        log(f"初始门洞三维核验确认 {report['confirmed_count']} 处，"
            f"保留证据不足候选 {report['uncertain_count']} 处，移除 {report['removed_count']} 处")


def _refine_boundaries(working, labels, initial, walls, origin, resolution, *, min_room_area, log):
    started = time.perf_counter()
    has_rooms = labels.max() >= 0
    cached = "review_height_rasters" in initial
    rasters = initial.pop("review_height_rasters", None)
    if has_rooms and not cached:
        rasters = _height_rasters(working, initial["normals"], origin, resolution, labels.shape)
    if not has_rooms or rasters is None:
        audit = dict(
            status="unsupported_height_range" if has_rooms else "too_few_rooms",
            seconds=round(time.perf_counter() - started, 4),
            accepted_door_count=0, accepted_corner_door_count=0, accepted_split_count=0,
            accepted_merges=0, restored_wall_pixel_count=0,
        )
        empty = np.zeros_like(walls)
        return labels, empty, empty, audit

    horizontal, vertical, z_step = rasters
    labels, cuts, restored_walls, door_audit = recover_door_splits(
        labels, walls, initial["observed_footprint"], horizontal, vertical,
        z_step, resolution, min_room_area,
    )
    restored_wall_pixels = int(np.count_nonzero(restored_walls))
    if log is not None:
        log(f"门洞补切：确认 {door_audit['accepted_door_count']} 处开口"
            f"（转角门 {door_audit['accepted_corner_door_count']} 处），"
            f"拆分 {door_audit['accepted_split_count']} 个初始区域，恢复 {restored_wall_pixels} 个实测墙格")
    barriers = walls | cuts | restored_walls
    initial_pairs = confirmed_initial_door_pairs(labels, initial["confirmed_door_columns"], barriers, resolution)
    protected_pairs = initial_pairs | {tuple(pair) for pair in door_audit["protected_pairs"]}
    labels, audit = review_boundaries(working, labels, initial, barriers, origin, resolution, log,
                                      height_rasters=rasters, protected_pairs=protected_pairs)
    audit.update(
        door_recovery=door_audit,
        accepted_door_count=door_audit["accepted_door_count"],
        accepted_corner_door_count=door_audit["accepted_corner_door_count"],
        accepted_split_count=door_audit["accepted_split_count"],
        restored_wall_pixel_count=restored_wall_pixels,
        initial_door_guard_pairs=[list(pair) for pair in sorted(initial_pairs)],
        seconds=round(time.perf_counter() - started, 4),
    )
    return labels, cuts, restored_walls, audit


def run_voxel_connectivity(xyz, params, output_dir, log=None, stage=None):
    """Return room labels after local structural and connectivity checks."""
    values = parameters(params)
    review = partial(_refine_boundaries, min_room_area=values["sa_min_room_area"])
    return _run_anchor_pipeline(xyz, values, output_dir, log, stage, method="voxel_connectivity",
                                method_version=METHOD_VERSION, review=review, robust_seeds=True,
                                verify_doors=partial(_verify_seed_doors, log=log))
