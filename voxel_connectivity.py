"""Run the voxel-connectivity room segmentation bundled with Boxer."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from potential_field import _watch_owner
from utils.potential_field_runtime.config import _ConfigLoader
from utils.voxel_connectivity_runtime import ARTIFACT_TYPES


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

    from utils.potential_field_runtime.pointcloud import read_cloud
    from utils.voxel_connectivity_runtime.room_polygons import room_polygons
    from utils.voxel_connectivity_runtime.voxel_connectivity import METHOD_VERSION, run_voxel_connectivity

    source = Path(input_path).resolve(strict=True)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    legacy_artifacts = ("result_colored.ply", "labels.npy", "rooms.json", "summary.json")
    if any((destination / name).exists() for name in (*ARTIFACT_TYPES, *legacy_artifacts)):
        raise ValueError("Output directory already contains room segmentation results")

    config = load_voxel_connectivity_config()
    params = VoxelConnectivityConfig.model_validate({**config, **options}).model_dump()
    xyz, _ = read_cloud(source)

    def log(message):
        print(message, flush=True)

    log(f"输入 {source.name}：{len(xyz):,} 个有效点；Z-up，单位米")
    log(f"方法 {METHOD_VERSION}")
    log("参数 " + json.dumps(params, ensure_ascii=False))
    with tempfile.TemporaryDirectory(prefix=".voxel-connectivity-", dir=destination) as scratch:
        scratch = Path(scratch)
        native = scratch / "native"
        raw_labels, artifacts = run_voxel_connectivity(
            xyz, {"sa_resolution": params["resolution"]}, native,
            log=log, stage=lambda name: log(f"开始 {name}"),
        )
        raw_labels = np.asarray(raw_labels)
        if (raw_labels.shape != (len(xyz),) or not np.issubdtype(raw_labels.dtype, np.integer)
                or np.any(raw_labels < -1) or np.any(raw_labels > np.iinfo(np.int32).max)):
            raise ValueError("算法标签必须与全量点云一一对应，为 -1 或非负整数")
        native_ids = {room["id"] for room in artifacts["rooms"]}
        room_ids, counts = np.unique(raw_labels[raw_labels >= 0], return_counts=True)
        if any(int(identity) not in native_ids for identity in room_ids):
            raise ValueError("点标签引用了不存在的原生房间")

        filtered = counts < params["min_room_points"]
        if filtered.any():
            log(f"点数阈值隐藏 {int(filtered.sum())} 个候选、{int(counts[filtered].sum()):,} 个已分配点。")

        label_map = np.load(native / "label_map.npy", mmap_mode="r", allow_pickle=False)
        polygons = room_polygons(label_map, artifacts["transform"], room_ids[~filtered])
        polygons.update(volume_type="vertical_prism", z_bounds_source="labeled_point_extents")
        for footprint in polygons["rooms"]:
            z = xyz[raw_labels == footprint["id"], 2]
            footprint["z_bounds_m"] = [float(z.min()), float(z.max())]

        name = "room_polygons.json"
        (scratch / name).write_text(
            json.dumps(polygons, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        os.replace(scratch / name, destination / name)
    coverage = counts[~filtered].sum() / len(xyz)
    log(f"完成：原始 {len(room_ids)} 个实例 → 保留 {len(polygons['rooms'])} 个房间；标注覆盖率 {coverage:.1%}")


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
