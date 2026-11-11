"""Integration tests for the bundled voxel-connectivity worker and API."""

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
ARTIFACTS = {"result_colored.ply", "rooms.json", "summary.json"}


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
    count = len(points)
    header = (
        "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        f"WIDTH {count}\nHEIGHT 1\nPOINTS {count}\nDATA binary\n"
    )
    path.write_bytes(header.encode("ascii") + points.tobytes())
    return points


@unittest.skipUnless(
    all(find_spec(name) for name in ("numpy", "scipy", "plyfile", "PIL", "yaml")),
    "Install the voxel-connectivity runtime dependencies to exercise the bundled CLI",
)
class TestVoxelConnectivityCli(unittest.TestCase):
    def test_two_room_artifacts_use_fresh_yaml_and_explicit_overrides(self):
        import numpy as np
        from plyfile import PlyData

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "global_map.pcd"
            points = _write_two_rooms(source)
            config = root / "voxel_connectivity.yaml"
            config.write_text("resolution: 0.1\nmin_room_points: 0\n", encoding="utf-8")
            env = {**os.environ, "BOXER_VC_CONFIG": str(config)}
            env.pop("BOXER_VC_OWNER_PID", None)
            env.pop("BOXER_PF_OWNER_PID", None)

            def run(output, *options):
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
                vertices = PlyData.read(output / "result_colored.ply")["vertex"]
                xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")])
                np.testing.assert_array_equal(xyz, points)
                rooms = json.loads((output / "rooms.json").read_text())
                summary = json.loads((output / "summary.json").read_text())
                return vertices["label"], rooms, summary

            labels, rooms, summary = run(root / "initial")
            self.assertEqual(summary["algorithm"], "voxel_connectivity")
            self.assertEqual(summary["method_version"], "voxel-connectivity-v8")
            self.assertEqual(
                summary["parameters"], {"resolution": 0.1, "min_room_points": 0}
            )
            self.assertEqual(summary["room_count"], 2)
            self.assertEqual(summary["point_count"], len(points))
            self.assertEqual(summary["unassigned_count"], int((labels < 0).sum()))
            ids, counts = np.unique(labels[labels >= 0], return_counts=True)
            self.assertEqual(
                {room["id"]: room["point_count"] for room in rooms},
                dict(zip(ids.tolist(), counts.tolist())),
            )
            left = labels[(points == [2, 2, 0]).all(axis=1)].item()
            right = labels[(points == [6, 2, 0]).all(axis=1)].item()
            self.assertGreaterEqual(min(left, right), 0)
            self.assertNotEqual(left, right)

            config.write_text(
                "resolution: 0.15\nmin_room_points: 0\n", encoding="utf-8"
            )
            labels, rooms, summary = run(
                root / "filtered", "--min-room-points", "100000"
            )
            self.assertEqual(
                summary["parameters"], {"resolution": 0.15, "min_room_points": 100000}
            )
            self.assertEqual(summary["room_count"], 0)
            self.assertEqual(summary["unassigned_count"], len(points))
            self.assertEqual(rooms, [])
            self.assertTrue((labels == -1).all())


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
deadline = time.monotonic() + 10
while not args.gate.exists():
    if time.monotonic() >= deadline:
        raise RuntimeError('Test did not release worker')
    time.sleep(.02)
args.output_dir.mkdir(parents=True, exist_ok=True)
(args.output_dir / 'result_colored.ply').write_bytes(
    b'ply\\nformat ascii 1.0\\nelement vertex 1\\n'
    b'property float x\\nproperty float y\\nproperty float z\\n'
    b'property int label\\nend_header\\n0 0 0 0\\n'
)
(args.output_dir / 'rooms.json').write_text(json.dumps([{'id': 0}]))
(args.output_dir / 'summary.json').write_text(json.dumps({
    'resolution': args.resolution, 'min_room_points': args.min_room_points,
}))
"""


@unittest.skipUnless(
    all(find_spec(name) for name in ("cv2", "torch", "httpx")),
    "Install the Boxer server and HTTP client dependencies to exercise HTTP routes",
)
class TestVoxelConnectivityApi(unittest.TestCase):
    def test_registered_map_routes_share_worker_slot_and_publish_results(self):
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
                pending = client.get(f"{job_url}/artifacts/rooms.json")
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
                for name, url in job["artifacts"].items():
                    self.assertEqual(url, f"{job_url}/artifacts/{name}")
                    downloaded = client.get(url)
                    self.assertEqual(downloaded.status_code, 200)
                    media_type = (
                        "application/octet-stream" if name.endswith(".ply")
                        else "application/json"
                    )
                    self.assertEqual(downloaded.headers["content-type"], media_type)
                summary = client.get(job["artifacts"]["summary.json"]).json()
                self.assertEqual(summary, options)
                log = client.get(f"{job_url}/log").json()["text"]
                self.assertIn("Started test worker", log)
                internal = client.get(f"{job_url}/artifacts/job.json")
                self.assertEqual(internal.status_code, 404)
                self.assertEqual(client.get(VC_JOBS).json()[0]["id"], job["id"])
            finally:
                client.close()
                vc_jobs.close()
                pf_jobs.close()
