"""Run the voxel-connectivity room segmentation bundled with Boxer."""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from potential_field import ARTIFACTS, _publish_artifacts, _watch_owner
from utils.potential_field_runtime.config import _ConfigLoader


PROJECT = Path(__file__).resolve().parent


class VoxelConnectivityConfigError(ValueError):
    """The configured voxel-connectivity YAML could not be loaded."""


class VoxelConnectivityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    resolution: float = Field(ge=.03, le=.50, allow_inf_nan=False)
    min_room_points: int = Field(ge=0, le=100_000_000)


def load_voxel_connectivity_config():
    """Read a fresh snapshot. Relative overrides resolve from the checkout."""
    path = Path(os.environ.get("BOXER_VC_CONFIG") or "config/voxel_connectivity.yaml")
    try:
        path = PROJECT / path.expanduser()
        values = yaml.load(path.read_text(encoding="utf-8"), Loader=_ConfigLoader)
        return VoxelConnectivityConfig.model_validate(values).model_dump()
    except (OSError, RuntimeError, ValueError, yaml.YAMLError) as error:
        raise VoxelConnectivityConfigError(f"体素连通配置无效（{path}）\n{error}") from error


def run(input_path, output_dir, options):
    import numpy as np

    from utils.potential_field_runtime.pointcloud import read_cloud, room_color
    from utils.voxel_connectivity_runtime.voxel_connectivity import run_voxel_connectivity

    started = time.perf_counter()
    source = Path(input_path).resolve(strict=True)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination / name).exists() for name in ARTIFACTS):
        raise ValueError("Output directory already contains room segmentation results")

    config = load_voxel_connectivity_config()
    params = VoxelConnectivityConfig.model_validate({**config, **options}).model_dump()
    xyz, normals = read_cloud(source)

    def log(message):
        print(message, flush=True)

    log(f"输入 {source.name}：{len(xyz):,} 个有效点；Z-up，单位米")
    log("参数 " + json.dumps(params, ensure_ascii=False))
    with tempfile.TemporaryDirectory(prefix=".voxel-connectivity-", dir=destination) as scratch:
        scratch = Path(scratch)
        raw_labels, artifacts = run_voxel_connectivity(
            xyz, {"sa_resolution": params["resolution"]}, scratch / "native",
            log=log, stage=lambda name: log(f"开始 {name}"),
        )
        raw_labels = np.asarray(raw_labels)
        if (raw_labels.shape != (len(xyz),) or not np.issubdtype(raw_labels.dtype, np.integer)
                or np.any(raw_labels < -1) or np.any(raw_labels > np.iinfo(np.int32).max)):
            raise ValueError("算法标签必须与全量点云一一对应，为 -1 或非负整数")
        labels = raw_labels.astype(np.int32, copy=True)
        native_rooms = {room["id"]: room for room in artifacts["rooms"]}
        room_ids, counts = np.unique(labels[labels >= 0], return_counts=True)
        if any(int(identity) not in native_rooms for identity in room_ids):
            raise ValueError("点标签引用了不存在的原生房间")

        filtered = counts < params["min_room_points"]
        filtered_count = int(counts[filtered].sum())
        if filtered.any():
            labels[np.isin(labels, room_ids[filtered])] = -1
            log(f"点数阈值隐藏 {int(filtered.sum())} 个候选、{filtered_count:,} 个已分配点。")

        rooms = []
        for room_id, count in zip(room_ids[~filtered], counts[~filtered]):
            identity = int(room_id)
            points = xyz[labels == identity]
            native_room = native_rooms[identity]
            rooms.append(dict(id=identity, name=f"房间 {identity + 1:02d}", point_count=int(count),
                              color=list(room_color(identity)), bounds=[points.min(0).tolist(), points.max(0).tolist()],
                              center=points.mean(0).tolist(), confidence=None,
                              area_m2=native_room["area_m2"], cell_count=native_room["cell_count"]))

        diagnostics = artifacts["statistics"]
        assigned = int(np.count_nonzero(labels >= 0))
        summary = dict(
            algorithm="voxel_connectivity", method_version=diagnostics["method_version"], parameters=params,
            room_count=len(rooms), raw_room_count=len(room_ids), native_room_count=len(native_rooms),
            filtered_room_count=int(filtered.sum()), point_count_filtered_count=filtered_count,
            point_count=len(xyz), assigned_count=assigned, unassigned_count=len(xyz) - assigned,
            assigned_ratio=assigned / len(xyz), inference_seconds=diagnostics["inference_seconds"],
            total_seconds=time.perf_counter() - started, diagnostics=diagnostics,
        )
        _publish_artifacts(scratch, destination, xyz, labels, normals, rooms, summary)
    log(f"完成：原始 {len(room_ids)} 个实例 → 保留 {len(rooms)} 个房间；标注覆盖率 {summary['assigned_ratio']:.1%}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Segment a +Z, metre-scale point cloud with voxel connectivity."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resolution", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--min-room-points", type=int, default=argparse.SUPPRESS)
    args = vars(parser.parse_args(argv))
    source = args.pop("input")
    destination = args.pop("output_dir")
    try:
        _watch_owner()
        run(source, destination, args)
    except ImportError as error:
        parser.exit(1, f"Voxel connectivity: {error}. Install requirements-potential-field.txt.\n")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Voxel connectivity: {error}\n")


if __name__ == "__main__":
    main()
