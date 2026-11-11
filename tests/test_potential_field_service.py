"""Exercise the room-segmentation job lifecycle without model dependencies."""

import json
import os
import struct
import sys
import tempfile
import time
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from utils.potential_field_service import PotentialFieldJobRequest, PotentialFieldJobs


WORKER = """
import argparse
import json
import struct
import sys
import time
from pathlib import Path

mode = sys.argv.pop(1)
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--voxel', type=float)
args = parser.parse_args()
assert args.input.read_bytes() == b'test map'
print('Room segmentation started', flush=True)
if mode == 'wait':
    time.sleep(30)
args.output_dir.mkdir(parents=True, exist_ok=True)
(args.output_dir / 'rooms.json').write_text(json.dumps([{'id': 0}]))
(args.output_dir / 'summary.json').write_text(json.dumps({
    'input': str(args.input), 'voxel': args.voxel, 'room_count': 1,
}))
if mode != 'missing':
    (args.output_dir / 'result_colored.ply').write_bytes(
        b'ply\\nformat binary_little_endian 1.0\\nelement vertex 1\\n'
        b'property float x\\nproperty float y\\nproperty float z\\n'
        b'property uchar red\\nproperty uchar green\\nproperty uchar blue\\n'
        b'property int label\\nend_header\\n'
        + struct.pack('<fffBBBi', 0, 0, 0, 1, 2, 3, 0)
    )
"""


class TestPotentialFieldJobs(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.map_root = self.root / "registered map"
        self.map_root.mkdir()
        self.point_cloud = self.map_root / "global_map.pcd"
        self.point_cloud.write_bytes(b"test map")
        self.output = self.root / "output"

    def jobs(self, mode="success"):
        jobs = PotentialFieldJobs(
            self.output, command=[sys.executable, "-c", WORKER, mode]
        )
        self.addCleanup(jobs.close)
        return jobs

    def wait_for_terminal(self, jobs, job_id, timeout=8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            jobs.poll()
            job = jobs.info(job_id)
            if job["status"] in {"complete", "failed", "interrupted"}:
                return job
            time.sleep(0.02)
        self.fail(f"Job {job_id} never finished")

    def test_registered_map_and_options_produce_downloadable_results(self):
        jobs = self.jobs()
        request = PotentialFieldJobRequest(map_id="map1", options={"voxel": 0.28})
        created = jobs.start(request, self.map_root)
        self.assertEqual(created["map_id"], "map1")
        self.assertEqual(created["status"], "running")

        finished = self.wait_for_terminal(jobs, created["id"])
        self.assertEqual(finished["status"], "complete", finished)
        job_id = finished["id"]
        self.assertEqual(
            set(finished["artifacts"]),
            {"result_colored.ply", "rooms.json", "summary.json"},
        )
        point_cloud = jobs.artifact(job_id, "result_colored.ply").read_bytes()
        self.assertTrue(point_cloud.startswith(b"ply\n"))
        rooms = json.loads(jobs.artifact(job_id, "rooms.json").read_text())
        self.assertEqual(rooms, [{"id": 0}])
        summary = json.loads(jobs.artifact(job_id, "summary.json").read_text())
        self.assertEqual(summary["input"], str(self.point_cloud.resolve()))
        self.assertEqual(summary["voxel"], 0.28)
        self.assertIn("Room segmentation started", jobs.log(job_id))
        with self.assertRaises(HTTPException) as raised:
            jobs.artifact(job_id, "job.json")
        self.assertEqual(raised.exception.status_code, 404)

        jobs.close()
        reopened = self.jobs()
        self.assertEqual(reopened.info(job_id), finished)
        self.assertEqual([row["id"] for row in reopened.list()], [job_id])

    def test_missing_source_and_incomplete_results_are_rejected(self):
        jobs = self.jobs("missing")
        request = PotentialFieldJobRequest(map_id="map1")
        self.point_cloud.unlink()
        with self.assertRaises(HTTPException) as raised:
            jobs.start(request, self.map_root)
        self.assertEqual(raised.exception.status_code, 404)
        self.assertEqual(jobs.list(), [])
        self.point_cloud.write_bytes(b"test map")

        created = jobs.start(request, self.map_root)
        failed = self.wait_for_terminal(jobs, created["id"])
        self.assertEqual(failed["status"], "failed", failed)
        self.assertFalse(failed["artifacts"])
        with self.assertRaises(HTTPException) as raised:
            jobs.artifact(failed["id"], "rooms.json")
        self.assertEqual(raised.exception.status_code, 404)

    @unittest.skipUnless(
        all(
            find_spec(name) for name in ("numpy", "scipy", "sklearn", "plyfile", "yaml")
        ),
        "Install requirements-potential-field.txt to exercise the bundled worker",
    )
    def test_bundled_worker_segments_registered_room_without_external_checkout(self):
        from plyfile import PlyData

        # Six surfaces of a 4 × 3 × 2.6 metre room at 0.1 metre spacing.
        points = [
            (x / 10, y / 10, z / 10)
            for x in range(41)
            for y in range(31)
            for z in range(27)
            if x in (0, 40) or y in (0, 30) or z in (0, 26)
        ]
        count = len(points)
        header = (
            "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
            f"WIDTH {count}\nHEIGHT 1\nPOINTS {count}\nDATA binary\n"
        ).encode("ascii")
        coordinates = (coordinate for point in points for coordinate in point)
        payload = struct.pack(f"<{3 * count}f", *coordinates)
        self.point_cloud.write_bytes(header + payload)

        jobs = PotentialFieldJobs(self.output)
        self.addCleanup(jobs.close)
        with patch.dict(os.environ):
            for variable in (
                "BOXER_PF_PROJECT",
                "BOXER_PF_PYTHON",
                "BOXER_PF_CONFIG",
                "ROOMSEG_PF_CONFIG",
            ):
                os.environ.pop(variable, None)
            job = jobs.start(PotentialFieldJobRequest(map_id="room"), self.map_root)
        identity = job["id"]
        finished = self.wait_for_terminal(jobs, identity, timeout=30)
        self.assertEqual(finished["status"], "complete", jobs.log(identity))
        rooms = json.loads(jobs.artifact(identity, "rooms.json").read_text())
        self.assertEqual(len(rooms), 1)
        vertices = PlyData.read(jobs.artifact(identity, "result_colored.ply"))["vertex"]
        self.assertEqual(len(vertices), count)

    def test_active_job_blocks_second_job_and_can_be_stopped(self):
        jobs = self.jobs("wait")
        request = PotentialFieldJobRequest(map_id="map1")
        job_id = jobs.start(request, self.map_root)["id"]
        with self.assertRaises(HTTPException) as raised:
            jobs.start(request, self.map_root)
        self.assertEqual(raised.exception.status_code, 409)
        with self.assertRaises(HTTPException) as raised:
            jobs.artifact(job_id, "rooms.json")
        self.assertEqual(raised.exception.status_code, 404)

        self.assertEqual(jobs.stop(job_id)["status"], "stopping")
        finished = self.wait_for_terminal(jobs, job_id)
        self.assertEqual(finished["status"], "interrupted", finished)
        self.assertFalse(finished["artifacts"])
