"""Integration tests for the bundled voxel-connectivity worker and API."""

import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
VC_JOBS = "/api/voxel-connectivity/jobs"
PF_JOBS = "/api/potential-field/jobs"
ARTIFACTS = {"room_polygons.json"}
LEGACY_PLY = (
    b"ply\nformat ascii 1.0\nelement vertex 1\n"
    b"property float x\nproperty float y\nproperty float z\n"
    b"property int label\nend_header\n0 0 0 0\n"
)


def _write_two_rooms(path):
    import numpy as np

    # An 8 × 4 × 2.8 m enclosure with a full wall at x = 4 m.
    x, y, z = np.indices((101, 51, 36))
    surfaces = (
        (x == 0) | (x == 100) | (x == 50) | (y == 0) | (y == 50) | (z == 0) | (z == 35)
    )
    points = (np.column_stack((x[surfaces], y[surfaces], z[surfaces])) * .08).astype(
        "<f4"
    )
    # Give the right room a distinct Z extent in an unordered PCD that also
    # contains a row the point-cloud reader drops.
    points = np.concatenate((points, np.array([[6, 2, 3.25]], dtype="<f4")))
    points = points[np.random.default_rng(5).permutation(len(points))]
    records = np.insert(points, len(points) // 2, [np.nan, 0, 0], axis=0)
    count = len(records)
    header = (
        "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        f"WIDTH {count}\nHEIGHT 1\nPOINTS {count}\nDATA binary\n"
    )
    path.write_bytes(header.encode("ascii") + records.tobytes())
    return points


@unittest.skipUnless(
    all(
        find_spec(name)
        for name in ("numpy", "scipy", "plyfile", "PIL", "yaml", "shapely")
    ),
    "Install the voxel-connectivity runtime dependencies to exercise the bundled CLI",
)
class TestVoxelConnectivityCli(unittest.TestCase):
    def test_two_room_artifacts_use_fresh_yaml_and_explicit_overrides(self):
        import numpy as np
        from shapely.geometry import Point, shape

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "global_map.pcd"
            point_count = len(_write_two_rooms(source))
            config = root / "voxel_connectivity.yaml"
            config.write_text("resolution: 0.1\nmin_room_points: 0\n", encoding="utf-8")
            env = {**os.environ, "BOXER_VC_CONFIG": str(config)}
            env.pop("BOXER_VC_OWNER_PID", None)
            env.pop("BOXER_PF_OWNER_PID", None)

            def run(output, expected_params, *options):
                result = subprocess.run(
                    [
                        sys.executable, "-m", "voxel_connectivity", "--input",
                        str(source), "--output-dir", str(output), *options,
                    ],
                    cwd=REPO,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual({path.name for path in output.iterdir()}, ARTIFACTS)
                self.assertIn("voxel-connectivity-v10", result.stdout)
                self.assertIn(f"{point_count:,} 个有效点", result.stdout)
                params = next(
                    line.removeprefix("参数 ") for line in result.stdout.splitlines()
                    if line.startswith("参数 ")
                )
                self.assertEqual(json.loads(params), expected_params)
                polygons = json.loads((output / "room_polygons.json").read_text())
                self.assertEqual(set(polygons), {
                    "schema_version", "coordinate_frame", "units", "resolution_m",
                    "volume_type", "z_bounds_source", "rooms",
                })
                self.assertEqual(polygons["schema_version"], 1)
                self.assertEqual(polygons["coordinate_frame"], "scene_xy")
                self.assertEqual(polygons["units"], "m")
                self.assertEqual(polygons["volume_type"], "vertical_prism")
                self.assertEqual(polygons["z_bounds_source"], "labeled_point_extents")
                self.assertEqual(
                    polygons["resolution_m"], expected_params["resolution"]
                )
                return polygons

            polygons = run(
                root / "initial", {"resolution": .1, "min_room_points": 0}
            )
            self.assertEqual(len(polygons["rooms"]), 2)
            footprints = [(shape(room["geometry"]), room) for room in polygons["rooms"]]
            footprints.sort(key=lambda item: item[0].centroid.x)
            (left, left_room), (right, right_room) = footprints
            self.assertNotEqual(left_room["id"], right_room["id"])
            for footprint, room in footprints:
                self.assertTrue(footprint.is_valid)
                self.assertAlmostEqual(footprint.area, room["area_m2"])
            self.assertTrue(left.covers(Point(2, 2)))
            self.assertFalse(left.covers(Point(6, 2)))
            self.assertTrue(right.covers(Point(6, 2)))
            self.assertFalse(right.covers(Point(2, 2)))
            np.testing.assert_allclose(left_room["z_bounds_m"], [0., 2.8])
            np.testing.assert_allclose(right_room["z_bounds_m"], [0., 3.25])

            config.write_text(
                "resolution: 0.15\nmin_room_points: 0\n", encoding="utf-8"
            )
            polygons = run(
                root / "filtered", {"resolution": .15, "min_room_points": 100000},
                "--min-room-points", "100000",
            )
            self.assertEqual(polygons["rooms"], [])


WORKER = """
import argparse
import json
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('gate', type=Path)
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--resolution', type=float)
parser.add_argument('--min-room-points', type=int)
args = parser.parse_args()
assert args.input.read_bytes() == b'registered map'
print('Started test worker', flush=True)
print('Worker options ' + json.dumps({
    'resolution': args.resolution, 'min_room_points': args.min_room_points,
}), flush=True)
deadline = time.monotonic() + 10
while not args.gate.exists():
    if time.monotonic() >= deadline:
        raise RuntimeError('Test did not release worker')
    time.sleep(.02)
args.output_dir.mkdir(parents=True, exist_ok=True)
(args.output_dir / 'room_polygons.json').write_text(json.dumps({
    'schema_version': 1, 'coordinate_frame': 'scene_xy', 'units': 'm',
    'resolution_m': args.resolution,
    'volume_type': 'vertical_prism', 'z_bounds_source': 'labeled_point_extents',
    'rooms': [{'id': 0, 'area_m2': 1.44, 'z_bounds_m': [0, 2.8], 'geometry': {
        'type': 'Polygon',
        'coordinates': [[[0, 0], [1.2, 0], [1.2, 1.2], [0, 1.2], [0, 0]]],
    }}],
}))
"""


def _write_legacy_job(root, identity, files):
    directory = root / "voxel_connectivity" / "jobs" / identity
    output = directory / "output"
    output.mkdir(parents=True)
    for name, payload in files.items():
        (output / name).write_bytes(payload)
    job_url = f"{VC_JOBS}/{identity}"
    artifacts = {name: f"{job_url}/artifacts/{name}" for name in files}
    (directory / "job.json").write_text(json.dumps(dict(
        id=identity, map_id="registered", status="complete", options={},
        created_at="2025-01-01T00:00:00+00:00", finished_at="2025-01-01T00:00:01+00:00",
        artifacts=artifacts,
    )))
    return job_url, artifacts, files


@unittest.skipUnless(
    all(find_spec(name) for name in ("numpy", "cv2", "torch", "httpx")),
    "Install the Boxer server and HTTP client dependencies to exercise HTTP routes",
)
class TestVoxelConnectivityApi(unittest.TestCase):
    def test_registered_map_routes_share_worker_slot_and_publish_results(self):
        import numpy as np
        from fastapi import HTTPException
        from fastapi.testclient import TestClient

        from server import create_app
        from utils.potential_field_service import (
            PotentialFieldJobs,
            VoxelConnectivityJobs,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_root = root / "registered map"
            map_root.mkdir()
            (map_root / "global_map.pcd").write_bytes(b"registered map")
            gate = root / "release"
            command = [sys.executable, "-c", WORKER, str(gate)]
            historical_labels = io.BytesIO()
            np.save(
                historical_labels, np.array([-1], dtype=np.int32), allow_pickle=False
            )
            histories = [
                _write_legacy_job(root, "8" * 32, {
                    "result_colored.ply": LEGACY_PLY,
                    "rooms.json": b'[{"id": 0}]',
                    "summary.json": b'{"method_version": "voxel-connectivity-v8"}',
                }),
                _write_legacy_job(root, "a" * 32, {
                    "labels.npy": historical_labels.getvalue(),
                    "rooms.json": b"[]",
                    "summary.json": b'{"method_version": "voxel-connectivity-v10"}',
                    "room_polygons.json": (
                        b'{"schema_version": 1, "coordinate_frame": "scene_xy", '
                        b'"units": "m", "resolution_m": 0.1, '
                        b'"volume_type": "vertical_prism", '
                        b'"z_bounds_source": "labeled_point_extents", "rooms": []}'
                    ),
                }),
            ]
            pf_jobs = PotentialFieldJobs(root, command=command)
            vc_jobs = VoxelConnectivityJobs(root, command=command)
            scene = SimpleNamespace(
                loader=SimpleNamespace(root=map_root), check_source=lambda: None
            )

            def get_map(identity):
                if identity != "registered":
                    raise HTTPException(404, "Unknown map")
                return scene

            app = create_app(
                output_dir=root,
                manager=SimpleNamespace(get_map=get_map),
                map_root=root,
                potential_field_jobs=pf_jobs,
                voxel_connectivity_jobs=vc_jobs,
            )
            # The route test uses the real app and no Boxer model startup.
            client = TestClient(app)
            try:
                unknown_map = client.post(VC_JOBS, json={"map_id": "missing"})
                self.assertEqual(unknown_map.status_code, 404)
                bad = {"map_id": "registered", "options": {"resolution": 0.01}}
                self.assertEqual(client.post(VC_JOBS, json=bad).status_code, 422)
                options = {"resolution": 0.12, "min_room_points": 50}
                body = {"map_id": "registered", "options": options}
                response = client.post(VC_JOBS, json=body)
                self.assertEqual(response.status_code, 202, response.text)
                job = response.json()
                self.assertEqual(job["status"], "running")
                self.assertEqual(job["options"], options)
                job_url = f"{VC_JOBS}/{job['id']}"
                pending = client.get(f"{job_url}/artifacts/room_polygons.json")
                self.assertEqual(pending.status_code, 404)
                overlapping = client.post(PF_JOBS, json={"map_id": "registered"})
                self.assertEqual(overlapping.status_code, 409)
                self.assertEqual(client.get(PF_JOBS).json(), [])
                self.assertEqual(client.get(f"{PF_JOBS}/{job['id']}").status_code, 404)

                gate.touch()
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    job = client.get(job_url).json()
                    if job["status"] != "running":
                        break
                    time.sleep(.02)
                self.assertEqual(job["status"], "complete", job)
                self.assertEqual(set(job["artifacts"]), ARTIFACTS)
                polygon_url = job["artifacts"]["room_polygons.json"]
                self.assertEqual(polygon_url, f"{job_url}/artifacts/room_polygons.json")
                downloaded = client.get(polygon_url)
                self.assertEqual(downloaded.status_code, 200)
                self.assertEqual(downloaded.headers["content-type"], "application/json")
                polygons = downloaded.json()
                self.assertEqual(polygons["resolution_m"], options["resolution"])
                self.assertEqual(polygons["rooms"][0]["z_bounds_m"], [0, 2.8])
                old_names = (
                    "result_colored.ply", "labels.npy", "rooms.json", "summary.json"
                )
                for name in old_names:
                    self.assertEqual(
                        client.get(f"{job_url}/artifacts/{name}").status_code, 404
                    )
                for legacy_url, artifacts, files in histories:
                    self.assertEqual(
                        client.get(legacy_url).json()["artifacts"], artifacts
                    )
                    for name, url in artifacts.items():
                        legacy = client.get(url)
                        self.assertEqual(legacy.status_code, 200)
                        self.assertEqual(legacy.content, files[name])
                        media_type = (
                            "application/json" if name.endswith(".json")
                            else "application/octet-stream"
                        )
                        self.assertEqual(legacy.headers["content-type"], media_type)
                log = client.get(f"{job_url}/log").json()["text"]
                self.assertIn("Started test worker", log)
                self.assertIn(f"Worker options {json.dumps(options)}", log)
                internal = client.get(f"{job_url}/artifacts/job.json")
                self.assertEqual(internal.status_code, 404)
                self.assertEqual(client.get(VC_JOBS).json()[0]["id"], job["id"])
            finally:
                client.close()
                vc_jobs.close()
                pf_jobs.close()
