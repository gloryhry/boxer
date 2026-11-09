"""Own independent potential-field tasks and publish completed artifacts."""

import contextlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from utils.xnav_result_store import _atomic_json

LOG = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[1]
API_PATH = "/api/potential-field/jobs"
ARTIFACTS = {
    "result_colored.ply": "application/octet-stream",
    "rooms.json": "application/json",
    "summary.json": "application/json",
}
ACTIVE = {"running", "stopping"}
JobStatus = Literal["running", "stopping", "complete", "failed", "interrupted"]


class PotentialFieldOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    voxel: float | None = Field(None, ge=0.02, le=5, allow_inf_nan=False)
    min_room_points: int | None = Field(None, ge=0, le=100_000_000, strict=True)
    min_room_area: float | None = Field(None, ge=0, le=1_000_000, allow_inf_nan=False)
    merge_opening: float | None = Field(None, ge=0, le=10, allow_inf_nan=False)
    max_projection_distance: float | None = Field(
        None, ge=0, le=10, allow_inf_nan=False
    )


class PotentialFieldJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    map_id: str = Field(min_length=1)
    options: PotentialFieldOptions = Field(default_factory=PotentialFieldOptions)


class PotentialFieldJobInfo(BaseModel):
    id: str
    map_id: str
    status: JobStatus
    options: dict[str, float | int]
    created_at: str
    finished_at: str | None = None
    error: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


class PotentialFieldJobs:
    def __init__(self, output_dir, *, command=None):
        self.root = Path(output_dir).expanduser().resolve() / "potential_field" / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.command = tuple(command) if command else None
        self.active_id = None
        self.process = None
        self.source = None
        self.stop_deadline = None
        self.closing = False
        self._recover()

    def _recover(self):
        for path in self.root.glob("*/job.json"):
            if path.parent.is_symlink() or path.is_symlink():
                continue
            try:
                job = PotentialFieldJobInfo.model_validate_json(
                    path.read_bytes()
                ).model_dump()
                if job["id"] != path.parent.name:
                    raise ValueError("Potential-field job ID does not match directory")
                if job["status"] in ACTIVE:
                    for name in ("staging", "output"):
                        shutil.rmtree(path.parent / name, ignore_errors=True)
                    job.update(
                        status="interrupted",
                        finished_at=_now(),
                        error="Service restarted before the job finished",
                        artifacts={},
                    )
                    _atomic_json(path, job)
                self.jobs[job["id"]] = job
            except (OSError, ValueError):
                LOG.exception(
                    "Could not recover potential-field job %s", path.parent.name
                )

    def _worker_command(self):
        if self.command is not None:
            return self.command
        python = Path(os.environ.get("BOXER_PF_PYTHON") or sys.executable)
        if (
            not python.is_absolute()
            or not python.is_file()
            or not os.access(python, os.X_OK)
        ):
            raise HTTPException(
                503, "BOXER_PF_PYTHON must be an absolute executable path"
            )
        return (str(python), "-u", "-m", "potential_field")

    def _save(self, job):
        _atomic_json(self.root / job["id"] / "job.json", job)
        self.jobs[job["id"]] = job

    def _get(self, identity):
        try:
            return self.jobs[identity]
        except KeyError:
            raise HTTPException(404, "Potential-field job not found") from None

    def start(self, request, map_root):
        with self.lock:
            self._poll()
            if self.closing:
                raise HTTPException(503, "Potential-field service is shutting down")
            if self.active_id is not None:
                raise HTTPException(409, "A potential-field job is already running")
            root = Path(map_root).resolve()
            map_cloud = root / "global_map.pcd"
            source = map_cloud.resolve()
            if not source.is_relative_to(root) or not source.is_file():
                raise HTTPException(404, "Registered map has no local global_map.pcd")
            command = self._worker_command()
            fingerprint = _fingerprint(source)
            identity = uuid.uuid4().hex
            directory = self.root / identity
            options = request.options.model_dump(exclude_none=True)
            args = [
                *command,
                "--input",
                str(source),
                "--output-dir",
                str(directory / "staging"),
            ]
            for name, value in options.items():
                args.extend([f"--{name.replace('_', '-')}", str(value)])
            job = PotentialFieldJobInfo(
                id=identity,
                map_id=request.map_id,
                status="running",
                created_at=_now(),
                options=options,
            ).model_dump()
            directory.mkdir()
            try:
                self._save(job)
            except OSError:
                shutil.rmtree(directory, ignore_errors=True)
                raise
            try:
                with (directory / "inference.log").open("wb") as log:
                    self.process = subprocess.Popen(
                        args,
                        cwd=REPO,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env={**os.environ, "BOXER_PF_OWNER_PID": str(os.getpid())},
                    )
            except OSError as error:
                self._finish(job, "failed", f"Could not start worker: {error}")
            else:
                self.active_id = identity
                self.source = (map_cloud, fingerprint)
            return self.jobs[identity].copy()

    def _finish(self, job, status, error=None):
        directory = self.root / job["id"]
        staging = directory / "staging"
        output = directory / "output"
        artifacts = {}
        if status == "complete":
            try:
                if staging.is_symlink():
                    raise ValueError("Invalid artifact directory")
                for name in ARTIFACTS:
                    path = staging / name
                    if (
                        path.is_symlink()
                        or not path.is_file()
                        or not path.stat().st_size
                    ):
                        raise ValueError(f"Missing or empty artifact: {name}")
                for name, kind in (("rooms.json", list), ("summary.json", dict)):
                    if not isinstance(json.loads((staging / name).read_bytes()), kind):
                        raise ValueError(f"Invalid artifact: {name}")
                source, fingerprint = self.source
                if _fingerprint(source) != fingerprint:
                    raise ValueError("global_map.pcd changed during segmentation")
                staging.rename(output)
                url = f"{API_PATH}/{job['id']}/artifacts"
                artifacts = {
                    name: f"{url}/{name}" for name in ARTIFACTS
                }
            except (OSError, ValueError) as exc:
                status, error = "failed", f"Could not publish worker output: {exc}"
        if status != "complete":
            shutil.rmtree(staging, ignore_errors=True)
        result = dict(
            job, status=status, finished_at=_now(), error=error, artifacts=artifacts
        )
        try:
            self._save(result)
        except OSError as exc:
            LOG.exception("Could not persist potential-field job %s", job["id"])
            if status == "complete":
                shutil.rmtree(output, ignore_errors=True)
            result.update(
                status="failed", error=f"Could not save job: {exc}", artifacts={}
            )
            self.jobs[job["id"]] = result
        self.active_id = self.process = self.source = self.stop_deadline = None

    def _signal(self, sig):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.process.pid, sig)

    def _poll(self):
        if self.process is None:
            return
        code = self.process.poll()
        if code is None:
            if (
                self.stop_deadline is not None
                and time.monotonic() >= self.stop_deadline
            ):
                self._signal(signal.SIGKILL)
                self.stop_deadline = None
            return
        job = self.jobs[self.active_id]
        if job["status"] == "stopping":
            self._finish(job, "interrupted", job["error"])
        elif code != 0:
            self._finish(
                job, "failed", f"Worker exited with status {code}. See job log."
            )
        else:
            self._finish(job, "complete")

    def poll(self):
        with self.lock:
            self._poll()

    def list(self):
        with self.lock:
            self._poll()
            return sorted(
                (job.copy() for job in self.jobs.values()),
                key=lambda job: job["created_at"],
                reverse=True,
            )

    def info(self, identity):
        with self.lock:
            self._poll()
            return self._get(identity).copy()

    def _request_stop(self, job, reason):
        if job["status"] == "running":
            self._save(dict(job, status="stopping", error=reason))
            self.stop_deadline = time.monotonic() + 10
            self._signal(signal.SIGTERM)

    def stop(self, identity):
        with self.lock:
            self._poll()
            job = self._get(identity)
            self._request_stop(job, "Stopped by request")
            return self._get(identity).copy()

    def log(self, identity):
        with self.lock:
            self._get(identity)
            path = self.root / identity / "inference.log"
        if path.is_symlink() or not path.is_file():
            return ""
        with path.open("rb") as stream:
            stream.seek(max(0, os.fstat(stream.fileno()).st_size - 65_536))
            return stream.read(65_536).decode("utf-8", errors="replace")

    def artifact(self, identity, filename):
        with self.lock:
            self._poll()
            job = self._get(identity)
            output = self.root / identity / "output"
            path = output / filename
            if (
                job["status"] != "complete"
                or filename not in ARTIFACTS
                or output.is_symlink()
                or path.is_symlink()
                or not path.is_file()
            ):
                raise HTTPException(404, "Potential-field artifact not found")
            return path

    def close(self):
        with self.lock:
            self.closing = True
            self._poll()
            if self.process is None:
                return
            job = self.jobs[self.active_id]
            reason = "Service is shutting down"
            try:
                self._request_stop(job, reason)
            except OSError:
                LOG.exception("Could not record shutdown for potential-field worker")
                self.jobs[job["id"]] = dict(job, status="stopping", error=reason)
                self._signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._signal(signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self.process.wait(timeout=5)
            self._poll()
