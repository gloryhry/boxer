"""Run the Room Atlas potential-field core on a metre-scale, Z-up cloud."""

import json
import math
import threading
import time
from contextlib import contextmanager

import numpy as np

from .pointcloud import estimate_normals, room_color
from .roomseg import pipeline

_PIPELINE_LOCK = threading.Lock()


def _check_grid(xyz, voxel, max_grid_voxels, log):
    with np.errstate(over="ignore"):
        spans = np.ptp(xyz, axis=0) / voxel
    if not np.isfinite(spans).all():
        raise ValueError("Point cloud extent is too large for voxelization")
    shape = tuple(math.ceil(span) + 3 for span in spans)
    count = math.prod(shape)
    log(f"体素网格估计 {shape}，共 {count:,} 个体素")
    if count > max_grid_voxels:
        raise ValueError(
            f"网格过大（{count:,} > {max_grid_voxels:,}）；请增大体素或裁切场景。"
        )


def _prepare_normals(xyz, normals, log):
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float64)
        if normals.shape != xyz.shape:
            raise ValueError("Point coordinates and normals must have matching shapes")
        lengths = np.linalg.norm(normals, axis=1)
        valid = np.isfinite(normals).all(axis=1) & (lengths > 0.1)
        if np.mean(valid) >= 0.99:
            log("使用输入点云中的有效法向量")
            return normals / np.maximum(lengths[:, None], 1e-12)
    return estimate_normals(xyz, log=log)


def _run_core(xyz, normals, options, log):
    # Room Atlas supplies the stack limit and stage logging around the core.
    # Keep those hooks scoped so the copied core stays identical to its source.
    with _PIPELINE_LOCK:
        original_timed = pipeline._timed
        original_visibility = pipeline.visibility

        @contextmanager
        def timed(name, bench):
            log(f"开始 {name}")
            with original_timed(name, bench):
                yield
            _, seconds, peak_memory = bench[-1]
            log(f"完成 {name}：{seconds:.3f} s；峰值内存 {peak_memory:.1f} MB")

        def guarded_visibility(grid, top_z, cells, targets=None, **kwargs):
            if not len(cells):
                raise ValueError(
                    "未找到室内空闲空间。请检查墙体、天花板、竖直方向、单位和体素大小。"
                )
            max_stacks = options["max_stacks"]
            if len(cells) > max_stacks:
                raise ValueError(
                    f"室内栈数 {len(cells):,} 超过计算保护阈值 {max_stacks:,}。"
                    "请增大体素或裁切场景。"
                )
            target_count = len(cells) if targets is None else len(targets)
            log(f"{len(cells):,} 个室内栈，可见性目标 {target_count}")
            return original_visibility(grid, top_z, cells, targets, **kwargs)

        pipeline._timed = timed
        pipeline.visibility = guarded_visibility
        started = time.perf_counter()
        try:
            labels, art = pipeline.segment(
                xyz,
                normals,
                voxel=options["voxel"],
                vis_targets=options["vis_targets"],
                min_room_area=options["min_room_area"],
                merge_opening=options["merge_opening"],
                max_projection_distance=options["max_projection_distance"],
                ceiling_aware=options["ceiling_aware"],
                structure_aware=False,
                boundary_review=False,
            )
        finally:
            pipeline._timed = original_timed
            pipeline.visibility = original_visibility
        return labels, art, time.perf_counter() - started


def segment(xyz, normals, options, *, log=print):
    """Return (labels, rooms, summary, normals_used) for a validated job.

    The caller resolves the YAML defaults and job overrides in ``options``.
    Coordinates, labels and normals use the same point order.
    """
    started = time.perf_counter()
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3:
        raise ValueError("Point cloud must contain at least three XYZ points")
    if not np.isfinite(xyz).all():
        raise ValueError("Point coordinates must be finite")
    options = dict(options)
    _check_grid(xyz, options["voxel"], options["max_grid_voxels"], log)
    normals = _prepare_normals(xyz, normals, log)
    raw_labels, art, inference_seconds = _run_core(xyz, normals, options, log)

    labels = np.asarray(raw_labels, dtype=np.int32).copy()
    if labels.shape != (len(xyz),):
        raise ValueError("算法标签长度与场景点集不一致")
    room_ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    filtered = counts < options["min_room_points"]
    filtered_points = int(counts[filtered].sum())
    if filtered.any():
        labels[np.isin(labels, room_ids[filtered])] = -1
        log(f"点数阈值隐藏 {int(filtered.sum())} 个候选、{filtered_points:,} 个已分配点。")

    rooms = []
    for room_id, count in zip(room_ids[~filtered], counts[~filtered]):
        identity = int(room_id)
        points = xyz[labels == identity]
        rooms.append(
            dict(
                id=identity,
                name=f"房间 {identity + 1:02d}",
                point_count=int(count),
                color=list(room_color(identity)),
                bounds=[points.min(0).tolist(), points.max(0).tolist()],
                center=points.mean(0).tolist(),
                confidence=None,
                area_m2=None,
            )
        )
    for title, key in (
        ("空间清理", "refinement"),
        ("高度采样", "vertical_sampling"),
        ("距离计算", "distance_computation"),
    ):
        log(f"{title} " + json.dumps(art[key], ensure_ascii=False))
    summary = dict(
        room_count=len(rooms),
        raw_room_count=len(room_ids),
        point_count=len(xyz),
        refinement=art["refinement"],
        vertical_sampling=art["vertical_sampling"],
        distance_computation=art["distance_computation"],
        local_heights=dict(enabled=False),
        structure=dict(enabled=False),
        multilevel_visibility=art["multilevel_visibility"],
        boundary_evidence=dict(enabled=False),
        boundary_review=art["boundary_review"],
        filtered_room_count=int(filtered.sum()),
        point_count_filtered_count=filtered_points,
        unassigned_count=int(np.count_nonzero(labels < 0)),
        assigned_ratio=float(np.mean(labels >= 0)),
        inference_seconds=inference_seconds,
        total_seconds=time.perf_counter() - started,
        peak_memory_mb=float(max(row[2] for row in art["bench"])),
        stages=[
            dict(name=name, seconds=seconds, peak_memory_mb=peak_memory)
            for name, seconds, peak_memory in art["bench"]
        ],
    )
    log(
        f"完成：原始 {len(room_ids)} 个实例 → 保留 {len(rooms)} 个房间；"
        f"标注覆盖率 {summary['assigned_ratio']:.1%}"
    )
    return labels, rooms, summary, normals
