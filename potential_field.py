"""Run the potential-field implementation bundled with Boxer in a worker."""

import argparse
import json
import os
import tempfile
import threading
import time
from pathlib import Path


ARTIFACTS = ("result_colored.ply", "rooms.json", "summary.json")


def _watch_owner():
    owner = os.environ.get("BOXER_PF_OWNER_PID")
    if owner is None:
        return
    parent_pid = int(owner)
    if os.getppid() != parent_pid:
        raise RuntimeError("Potential-field service exited before the worker started")

    def watch():
        while os.getppid() == parent_pid:
            time.sleep(0.5)
        os._exit(1)

    threading.Thread(target=watch, name="potential-field-owner", daemon=True).start()


def run(input_path, output_dir, options):
    from utils.potential_field_runtime.config import (
        PotentialFieldConfig,
        load_potential_field_config,
    )
    from utils.potential_field_runtime.engine import segment
    from utils.potential_field_runtime.pointcloud import read_cloud, save_colored_ply

    source = Path(input_path).resolve(strict=True)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination / name).exists() for name in ARTIFACTS):
        raise ValueError("Output directory already contains potential-field results")

    config = load_potential_field_config()
    params = PotentialFieldConfig.model_validate({**config, **options}).model_dump()
    xyz, normals = read_cloud(source)
    print(f"输入 {source.name}：{len(xyz):,} 个有效点；Z-up，单位米", flush=True)
    print("参数 " + json.dumps(params, ensure_ascii=False), flush=True)
    labels, rooms, summary, normals = segment(xyz, normals, params)
    summary["parameters"] = params

    with tempfile.TemporaryDirectory(prefix=".potential-field-", dir=destination) as scratch:
        artifacts = Path(scratch)
        save_colored_ply(artifacts / "result_colored.ply", xyz, labels, normals)
        for name, value in (("rooms.json", rooms), ("summary.json", summary)):
            (artifacts / name).write_text(
                json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        for name in ARTIFACTS:
            os.replace(artifacts / name, destination / name)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Segment a +Z, metre-scale point cloud with the bundled potential field."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--voxel", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--min-room-points", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--min-room-area", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--merge-opening", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--max-projection-distance", type=float, default=argparse.SUPPRESS)
    args = vars(parser.parse_args(argv))
    source = args.pop("input")
    destination = args.pop("output_dir")
    try:
        _watch_owner()
        run(source, destination, args)
    except ImportError as error:
        parser.exit(1, f"Potential field: {error}. Install requirements-potential-field.txt.\n")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Potential field: {error}\n")


if __name__ == "__main__":
    main()
