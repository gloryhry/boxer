"""Run the existing Room Atlas potential-field implementation in a worker."""

import argparse
import os
import sys
import tempfile
import threading
import time
from pathlib import Path


ARTIFACTS = ("result_colored.ply", "rooms.json", "summary.json")


def project_path():
    location = os.environ.get("BOXER_PF_PROJECT")
    if not location:
        raise ValueError("Set BOXER_PF_PROJECT to the room_instance_seg checkout")
    project = Path(location).expanduser().resolve()
    if not (project / "backend" / "segmentation.py").is_file():
        raise ValueError(f"BOXER_PF_PROJECT is not a room_instance_seg checkout: {project}")
    return project


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
    source = Path(input_path).resolve(strict=True)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination / name).exists() for name in ARTIFACTS):
        raise ValueError("Output directory already contains potential-field results")

    sys.path.insert(0, str(project_path()))
    try:
        from backend.pointcloud import import_scene
        from backend.segmentation import create_run, execute_run
    finally:
        sys.path.pop(0)

    with tempfile.TemporaryDirectory(prefix=".roomseg-", dir=destination) as scratch:
        root = Path(scratch)
        scene = import_scene(
            root, source, up="z", scale=1.0, preview_points=100, copy_source=False
        )
        job = create_run(root, scene["id"], options, algorithm="potential_field")
        result = execute_run(root, job["id"])
        if result["status"] != "succeeded":
            raise RuntimeError(result.get("error") or f"Potential field {result['status']}")

        artifacts = root / "runs" / job["id"]
        for name in ARTIFACTS:
            if not (artifacts / name).is_file():
                raise RuntimeError(f"Potential field did not produce {name}")
        for name in ARTIFACTS:
            os.replace(artifacts / name, destination / name)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run room_instance_seg on a +Z, metre-scale point cloud."
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
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Potential field: {error}\n")


if __name__ == "__main__":
    main()
