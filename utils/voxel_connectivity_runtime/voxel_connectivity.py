"""Room partitioning with multilevel checks of uncertain boundaries."""

from functools import partial
import time

import numpy as np

from .supervoxel_anchor import DEFAULT_PARAMS, _run_anchor_pipeline, parameters
from .voxel_connectivity_review import _height_rasters, review_boundaries
from .voxel_door_recovery import recover_door_splits


METHOD_VERSION = "voxel-connectivity-v8"


def voxel_connectivity_status():
    return dict(ready=True, reason="", defaults=DEFAULT_PARAMS.copy(), method_version=METHOD_VERSION)


def _refine_boundaries(working, labels, initial, walls, origin, resolution, *, min_room_area, log):
    started = time.perf_counter()
    has_rooms = labels.max() >= 0
    rasters = (_height_rasters(working, initial["normals"], origin, resolution, labels.shape)
               if has_rooms else None)
    if rasters is None:
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
    labels, audit = review_boundaries(
        working, labels, initial, walls | cuts | restored_walls, origin, resolution, log,
        height_rasters=rasters, protected_pairs=door_audit["protected_pairs"],
    )
    audit.update(
        door_recovery=door_audit,
        accepted_door_count=door_audit["accepted_door_count"],
        accepted_corner_door_count=door_audit["accepted_corner_door_count"],
        accepted_split_count=door_audit["accepted_split_count"],
        restored_wall_pixel_count=restored_wall_pixels,
        seconds=round(time.perf_counter() - started, 4),
    )
    return labels, cuts, restored_walls, audit


def run_voxel_connectivity(xyz, params, output_dir, log=None, stage=None):
    """Return room labels after local structural and connectivity checks."""
    values = parameters(params)
    review = partial(_refine_boundaries, min_room_area=values["sa_min_room_area"])
    return _run_anchor_pipeline(xyz, values, output_dir, log, stage, method="voxel_connectivity",
                                method_version=METHOD_VERSION, review=review, robust_seeds=True)
