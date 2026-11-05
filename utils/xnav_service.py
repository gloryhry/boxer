"""Single-owner inference jobs and durable replay for the local HTTP service."""

import contextlib
import json
import os
import signal
import threading
import time
import uuid
from concurrent.futures import CancelledError, ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from loaders.xnav_loader import XNavMapLoader, is_xnav_map, processing_frame_id
from utils.taxonomy import load_text_labels
from utils.xnav_archive import create_map_archive
from utils.xnav_export import export_map_csv
from utils.xnav_pipeline import XNavReplay
from utils.xnav_result_store import _atomic_json
from utils.xnav_web_scene import MapScene
from utils.xnav_worker_client import InferenceWorker

REPO = Path(__file__).resolve().parents[1]
ACTIVE = {"starting", "running", "stopping"}
Status = Literal["starting", "running", "stopping", "complete", "failed", "interrupted"]


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MapRequest(RequestModel):
    path: str = Field(
        min_length=1,
        description="Server-local map path; full paths outside /nav_map are supported.",
    )


class InferenceOptions(RequestModel):
    labels: str = Field(default="indoor", min_length=1, max_length=8192)
    thresh2d: float = Field(default=0.25, ge=0, le=1)
    thresh3d: float = Field(default=0.5, ge=0, le=1)
    xnav_cameras: str = Field(default="all", min_length=1)
    xnav_camera_mode: Literal["native", "virtual_pinhole"] = "native"
    track: bool = True
    track_min_hits: int | None = Field(default=None, ge=1)
    start_n: int = Field(default=1, ge=1)
    skip_n: int = Field(default=1, ge=1)
    max_n: int = Field(default=99999, ge=1)
    detector_hw: int = Field(default=960, ge=2)
    force_precision: Literal["float32", "bfloat16"] | None = None
    xnav_execution: Literal["auto", "serial"] = "auto"
    xnav_detector_batch_size: Literal[1, 2] = 1
    xnav_dedup: Literal["conservative", "legacy"] = "conservative"
    force_cpu: bool = False
    no_sdp: bool = False
    no_csv: bool = False
    profile: bool = False
    ckpt: str | None = None

    @field_validator("labels")
    @classmethod
    def nonempty_labels(cls, value):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("At least one label is required")
        return ",".join(parts)

    @field_validator("xnav_cameras")
    @classmethod
    def nonempty_cameras(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("At least one camera is required")
        return value

    @field_validator("ckpt")
    @classmethod
    def optional_checkpoint(cls, value):
        if value is None:
            return None
        return value.strip() or None


class JobRequest(RequestModel):
    map_id: str
    options: InferenceOptions | None = None


class InferenceInfo(BaseModel):
    status: Literal["loading", "ready", "failed"]
    options: InferenceOptions
    device: str
    error: str | None = None


class ImportRequest(RequestModel):
    map_id: str
    results_path: str = Field(min_length=1)


class MapInfo(BaseModel):
    id: str
    path: str
    validation: dict
    cameras: list[dict]


class MapDirectory(BaseModel):
    name: str = Field(description="Map path relative to the directory root.")
    path: str = Field(description="Full server-local path accepted by POST /api/maps.")


class MapDirectories(BaseModel):
    root: str
    directories: list[MapDirectory]


def list_map_directories(root):
    """Discover map packages without loading their images, clouds or calibration."""
    root = Path(root).expanduser().absolute()

    def walk_error(error):
        if Path(error.filename) == root:
            if isinstance(error, FileNotFoundError):
                raise HTTPException(404, f"Map directory not found: {root}") from error
            raise error
        # A missing or unreadable subtree must not hide the other maps.

    directories = []
    for folder, children, _ in os.walk(root, onerror=walk_error, followlinks=False):
        path = Path(folder)
        if path == root:
            children[:] = [name for name in children if name != ".boxer_uploads"]
        if path == root:
            continue
        try:
            if is_xnav_map(path):
                directories.append(
                    {"name": path.relative_to(root).as_posix(), "path": str(path)}
                )
        except OSError:
            continue
    return {
        "root": str(root),
        "directories": sorted(directories, key=lambda item: item["name"]),
    }


class SceneInfo(BaseModel):
    center: list[float]
    radius: float
    point_count: int
    format: str
    up_axis: str
    unit: str


class JobInfo(BaseModel):
    id: str
    map_id: str
    status: Status
    readonly: bool
    created_at: str
    output_dir: str
    results_path: str
    options: dict
    processed_frames: int
    total_frames: int
    elapsed_seconds: float = 0
    tracking: bool = False
    device: str = "initializing"
    error: str | None = None


class SavedModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class BoxRecord(SavedModel):
    center: list[float]
    quaternion_wxyz: list[float]
    size: list[float]
    label: str
    sem_id: int
    prob: float
    track_id: int
    state: str | None = None


class Observation(SavedModel):
    camera_id: int
    view_id: str
    image_stamp_ns: str
    T_map_view: list[list[float]]
    boxes_2d: list[list[float]]
    scores_2d: list[float]
    labels_2d: list[str]
    detections: list[BoxRecord] = Field(default_factory=list)


class FrameSummary(BaseModel):
    index: int
    frame_id: int
    image_frame_id: int | None = None
    lidar_keyframe_id: int
    time_ns: str
    position: list[float]


class FrameData(SavedModel):
    index: int
    frame_id: int
    image_frame_id: int | None = None
    lidar_keyframe_id: int
    time_ns: str
    T_map_base: list[list[float]]
    observations: list[Observation]
    tracks: list[BoxRecord]
    detections: list[BoxRecord]


class FramePage(BaseModel):
    total: int
    offset: int
    frames: list[FrameSummary]


class Box3DRow(BaseModel):
    """Saved 3D box values using the boxer_3dbbs CSV column names."""

    time_ns: str = Field(description="Frame time in nanoseconds; global objects use 0.")
    tx_world_object: float
    ty_world_object: float
    tz_world_object: float
    qw_world_object: float
    qx_world_object: float
    qy_world_object: float
    qz_world_object: float
    scale_x: float
    scale_y: float
    scale_z: float
    name: str
    instance: int
    sem_id: int
    prob: float


class Box2DRow(BaseModel):
    """Saved 2D box values using the owl_2dbbs CSV column names."""

    time_ns: str = Field(description="Source image timestamp in nanoseconds.")
    frame_id: int
    sensor: str
    device: str
    img_width: int
    img_height: int
    x1: float
    y1: float
    x2: float
    y2: float
    name: str
    instance: int
    sem_id: int
    prob: float


class BoxPage(BaseModel):
    status: Status
    processed_frames: int = Field(ge=0)
    total: int = Field(ge=0, description="Box count in this committed snapshot.")
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=1000)


class Box3DPage(BoxPage):
    items: list[Box3DRow]


class Box2DPage(BoxPage):
    items: list[Box2DRow]


def _box3d_row(record, time_ns):
    tx, ty, tz = record["center"]
    qw, qx, qy, qz = record["quaternion_wxyz"]
    sx, sy, sz = record["size"]
    return {
        "time_ns": str(time_ns),
        "tx_world_object": tx,
        "ty_world_object": ty,
        "tz_world_object": tz,
        "qw_world_object": qw,
        "qx_world_object": qx,
        "qy_world_object": qy,
        "qz_world_object": qz,
        "scale_x": sx,
        "scale_y": sy,
        "scale_z": sz,
        "name": record["label"],
        "instance": record["track_id"],
        "sem_id": record["sem_id"],
        "prob": record["prob"],
    }


def _box2d_row(frame, observation, index, *, image_size, sem_map):
    x1, x2, y1, y2 = observation["boxes_2d"][index]
    labels, scores = observation["labels_2d"], observation["scores_2d"]
    name = labels[index] if index < len(labels) else "unknown"
    return {
        "time_ns": str(observation["image_stamp_ns"]),
        "frame_id": processing_frame_id(frame),
        "sensor": f"cam{observation['camera_id']}/{observation['view_id']}",
        "device": "x-nav",
        "img_width": image_size,
        "img_height": image_size,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "name": name,
        "instance": -1,
        "sem_id": sem_map.get(name, -1),
        "prob": scores[index] if index < len(scores) else 0.0,
    }


def _box3d_entries(replay, kind):
    if kind in ("3dbbox_tracked", "3dbbox_tracks"):
        if not replay.results["tracking"]["enabled"]:
            return
        if kind == "3dbbox_tracked":
            if (
                replay.results.get("status") == "complete"
                and "objects" in replay.results
            ):
                for record in replay.results["objects"]:
                    yield record, 0
            elif replay.frames:
                for record in replay.frames[-1]["tracks"]:
                    if record.get("state") != "TENTATIVE":
                        yield record, 0
            return
        key = "tracks"
    elif kind == "3dbbox":
        key = "detections"
    else:
        raise ValueError(f"Unknown box result kind: {kind}")
    for frame in replay.frames:
        for record in frame[key]:
            if key != "tracks" or record.get("state") != "TENTATIVE":
                yield record, frame["time_ns"]


def timestamps(value):
    """Preserve nanosecond integers across JSON/JavaScript's 53-bit boundary."""
    if isinstance(value, dict):
        return {
            key: str(item)
            if key.endswith("_ns") and item is not None
            else timestamps(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [timestamps(item) for item in value]
    return value


class JobManager:
    def __init__(self, root, *, stop_timeout=60, worker_factory=InferenceWorker):
        self.root = Path(root).expanduser().resolve()
        (self.root / "maps").mkdir(parents=True, exist_ok=True)
        (self.root / "jobs").mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.stop_timeout = stop_timeout
        self.map_records = {}
        self.maps = {}
        self.jobs = {}
        self.replays = {}
        self.processes = {}
        self.stop_deadlines = {}
        self.active_id = None
        self.worker_factory = worker_factory
        self.worker = None
        self.inference_status = "loading"
        self.applied_options = InferenceOptions().model_dump()
        self.pending_options = None
        self.inference_device = "initializing"
        self.inference_error = None
        self.finished = {}
        self._finalizer = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="xnav-finalize"
        )
        self._finalizations = {}
        self._closing = False
        self._recovering = False
        for path in (self.root / "maps").glob("*.json"):
            record = json.loads(path.read_text())
            self.map_records[record["id"]] = record
        for path in (self.root / "jobs").glob("*/job.json"):
            record = json.loads(path.read_text())
            if record["status"] in ACTIVE and not record["readonly"]:
                record.update(
                    status="interrupted",
                    error="Service restarted; inference was not resumed",
                )
                _atomic_json(path, record)
                # Only our atomic ZIP temporaries can be left by a killed server.
                for temporary in path.parent.glob(".3dbbox.zip.*.tmp"):
                    temporary.unlink(missing_ok=True)
                (path.parent / "3dbbox.zip").unlink(missing_ok=True)
            self.jobs[record["id"]] = record

    def _save(self, job):
        folder = self.root / "jobs" / job["id"]
        folder.mkdir(exist_ok=True)
        _atomic_json(folder / "job.json", job)

    def add_map(self, path):
        scene = MapScene(path)
        return self.register_map(scene)

    def register_map(self, scene, *, identity=None):
        """Register a fully validated scene without reading its assets twice."""
        with self.lock:
            for record in self.map_records.values():
                if (
                    record["path"] == str(scene.loader.root)
                    and record["signature"] == scene.signature
                ):
                    self.maps[record["id"]] = scene
                    return self.map_info(record["id"])
            identity = identity or uuid.uuid4().hex
            if identity in self.map_records:
                raise HTTPException(409, "Map identity already registered")
            record = {
                "id": identity,
                "path": str(scene.loader.root),
                "signature": scene.signature,
                "validation": scene.validation,
                "cameras": [
                    {"id": cid, "model": scene.loader.cameras[cid].model}
                    for cid in scene.loader.camera_ids
                ],
            }
            _atomic_json(self.root / "maps" / f"{identity}.json", record)
            self.map_records[identity] = record
            self.maps[identity] = scene
            return self.map_info(identity)

    def map_info(self, identity):
        if identity not in self.map_records:
            raise HTTPException(404, "Map not found")
        return MapInfo.model_validate(self.map_records[identity]).model_dump()

    def get_map(self, identity):
        with self.lock:
            self.map_info(identity)
            if identity not in self.maps:
                scene = MapScene(self.map_records[identity]["path"])
                if scene.signature != self.map_records[identity]["signature"]:
                    raise ValueError(
                        "Map changed since it was registered; load the map again"
                    )
                self.maps[identity] = scene
            return self.maps[identity]

    def inference_info(self):
        with self.lock:
            return InferenceInfo(
                status=self.inference_status,
                options=self.applied_options,
                device=self.inference_device,
                error=self.inference_error,
            ).model_dump()

    def _spawn_worker(self, options):
        self.inference_status = "loading"
        self.inference_error = None
        self.pending_options = dict(options)
        try:
            self.worker = self.worker_factory(self.root)
            self.worker.send({"op": "configure", "options": options})
        except Exception as error:
            if self.worker is not None:
                self.worker.close()
                self.worker = None
            self.pending_options = None
            self.inference_status = "failed"
            self.inference_error = f"Could not start model worker: {error}"

    def open(self):
        """Called by HTTP startup, before any map or inference request exists."""
        with self.lock:
            if self.worker is None and not self._closing:
                self._spawn_worker(self.applied_options)

    def configure(self, options):
        with self.lock:
            if self._closing:
                raise HTTPException(503, "The inference service is shutting down")
            if self.active_id or (
                self.inference_status == "loading" and self.worker is not None
            ):
                raise HTTPException(
                    409, "Inference or model loading is already in progress"
                )
            values = options.model_dump()
            if not load_text_labels(
                [s.strip() for s in values["labels"].split(",") if s.strip()]
            ):
                raise ValueError("At least one label is required")
            if self.worker is None or self.worker.process.poll() is not None:
                if self.worker is not None:
                    self.worker.close()
                    self.worker = None
                self._recovering = False
                self._spawn_worker(values)
            else:
                self.inference_status = "loading"
                self.inference_error = None
                self.pending_options = values
                try:
                    self.worker.send({"op": "configure", "options": values})
                except OSError:
                    self.worker.close()
                    self.worker = None
                    self._spawn_worker(values)
            return self.inference_info()

    def _new_job(self, map_id, options, readonly=False):
        identity = uuid.uuid4().hex
        return {
            "id": identity,
            "map_id": map_id,
            "status": "starting",
            "readonly": readonly,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(self.root / "jobs" / identity / "output"),
            "results_path": "",
            "options": options,
            "processed_frames": 0,
            "total_frames": 0,
            "elapsed_seconds": 0,
            "error": None,
        }

    def start(self, request):
        with self.lock:
            if self._closing:
                raise HTTPException(503, "The inference service is shutting down")
            if self.active_id is not None:
                raise HTTPException(409, "An inference job is already running")
            if (
                self.inference_status != "ready"
                or self.worker is None
                or self.worker.process.poll() is not None
            ):
                raise HTTPException(
                    503, "Inference models are not ready; check /api/inference"
                )
            if (
                request.options is not None
                and request.options.model_dump() != self.applied_options
            ):
                raise HTTPException(
                    409, "Parameters differ; apply them with PUT /api/inference first"
                )
            scene = self.get_map(request.map_id)
            scene.check_source()
            options = dict(self.applied_options)
            loader = XNavMapLoader(
                scene.loader.root,
                cameras=options["xnav_cameras"],
                camera_mode=options["xnav_camera_mode"],
                start_n=options["start_n"],
                skip_n=options["skip_n"],
                max_n=options["max_n"],
            )
            job = self._new_job(request.map_id, options)
            job["total_frames"] = len(loader)
            job["results_path"] = str(
                loader.output_path(job["output_dir"]) / "results.json"
            )
            self.jobs[job["id"]] = job
            self._save(job)
            self.active_id = job["id"]
            try:
                self.worker.send(
                    {
                        "op": "run",
                        "job_id": job["id"],
                        "input": self.map_records[job["map_id"]]["path"],
                        "output_dir": job["output_dir"],
                        "options": options,
                        "log_path": str(
                            self.root / "jobs" / job["id"] / "inference.log"
                        ),
                    }
                )
                self.processes[job["id"]] = self.worker.process
            except Exception as error:
                job.update(status="failed", error=f"Could not start inference: {error}")
                self.active_id = None
                self._save(job)
            return self.info(job["id"])

    def import_results(self, request):
        with self.lock:
            scene = self.get_map(request.map_id)
            scene.check_source()
            path = Path(request.results_path).expanduser().resolve()
            replay = XNavReplay(path, live=True)
            self._validate_replay(scene, replay)
            job = self._new_job(
                request.map_id,
                {
                    "xnav_camera_mode": replay.results["parameters"]["camera_mode"],
                    "xnav_dedup": replay.results["tracking"]["dedup"]["mode"],
                },
                readonly=True,
            )
            job.update(
                results_path=str(path),
                output_dir=str(path.parent),
                total_frames=len(replay.frames),
            )
            self.jobs[job["id"]] = job
            self.replays[job["id"]] = replay
            self._apply_replay(job, replay)
            self._save(job)
            return self.info(job["id"])

    @staticmethod
    def _validate_replay(scene, replay):
        if replay.results["source_signature"] != scene.signature:
            raise ValueError("Saved results do not match this map")
        scene.result_loader(replay.results["parameters"])

    def _apply_replay(self, job, replay):
        job.update(
            processed_frames=len(replay.frames),
            elapsed_seconds=replay.results.get("elapsed_seconds", 0),
        )
        status = replay.results.get("status", "complete")
        if job["readonly"]:
            job["total_frames"] = len(replay.frames)
            job["status"] = status
        elif job["status"] == "starting" and status == "running":
            job["status"] = "running"
        if replay.results.get("error") and not (
            job.get("stop_requested") and status == "interrupted"
        ):
            job["error"] = replay.results["error"]

    def _worker_events(self):
        if self.worker is None:
            return
        for message in self.worker.events():
            event = message["event"]
            if event == "configured":
                if self.pending_options is not None:
                    self.applied_options = self.pending_options
                self.pending_options = None
                self.inference_status = "ready"
                self.inference_device = message["device"]
                self.inference_error = None
                self._recovering = False
            elif event == "configure_failed":
                self.pending_options = None
                self.inference_status = "failed"
                self.inference_error = message["error"]
            elif event == "job_started" and message["job_id"] == self.active_id:
                job = self.jobs[self.active_id]
                if job["status"] == "starting":
                    job["status"] = "running"
                if job.get("stop_requested"):
                    self._interrupt(job)
                self._save(job)
            elif event == "job_finished" and message["job_id"] == self.active_id:
                self.finished[self.active_id] = message
            elif event == "control_error":
                self.inference_error = (
                    f"Worker control channel failed: {message['error']}"
                )
                self.worker.signal_group(signal.SIGTERM)

    def _complete_job(self, job, message):
        identity = job["id"]
        if identity in self._finalizations:
            return
        # job_finished acknowledges all output handles are closed. The worker is
        # idle now, even while this job continues exporting and compressing files.
        self.processes.pop(identity, None)
        self.stop_deadlines.pop(identity, None)
        replay = self.replays.get(identity)
        if job.get("stop_requested") or message["status"] == "interrupted":
            job["status"] = "interrupted"
        elif (
            message["status"] == "complete"
            and replay is not None
            and replay.results.get("status") == "complete"
            and not job["error"]
        ):
            if job["options"]["no_csv"]:
                job["status"] = "complete"
            else:
                cancel = threading.Event()
                future = self._finalizer.submit(
                    self._export_results,
                    Path(job["results_path"]).parent,
                    self.map_records[job["map_id"]]["path"],
                    self.root / "jobs" / identity / "3dbbox.zip",
                    cancel,
                )
                self._finalizations[identity] = (future, cancel)
                job["status"] = "running"
                self._save(job)
                return
        else:
            job["status"] = "failed"
            job["error"] = (
                job["error"]
                or message.get("error")
                or ("Worker finished without a complete committed result")
            )
        self._finish_job(job)

    def _finish_job(self, job):
        if self.active_id == job["id"]:
            self.active_id = None
        self._save(job)

    @staticmethod
    def _export_results(source, map_root, archive, cancel):
        if cancel.is_set():
            raise CancelledError()
        try:
            export_map_csv(source, map_root)
        except Exception as error:
            raise RuntimeError(f"Could not export map CSVs: {error}") from error
        try:
            create_map_archive(map_root, archive, cancel)
        except CancelledError:
            raise
        except Exception as error:
            raise RuntimeError(f"Could not archive map 3dbbox: {error}") from error

    def _poll_finalizations(self):
        for identity, (future, _) in list(self._finalizations.items()):
            if not future.done():
                continue
            job = self.jobs[identity]
            try:
                future.result()
            except CancelledError:
                job["status"] = "interrupted"
            except Exception as error:
                job.update(status="failed", error=str(error))
            else:
                job["status"] = "failed" if job["error"] else "complete"
            # A stop may arrive after the archive was atomically published but
            # before its future was collected. It still wins over completion.
            if job.get("stop_requested"):
                job["status"] = "interrupted"
            if job["status"] != "complete":
                with contextlib.suppress(OSError):
                    (self.root / "jobs" / identity / "3dbbox.zip").unlink(
                        missing_ok=True
                    )
            del self._finalizations[identity]
            self._finish_job(job)

    def _recover_worker(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        self.pending_options = None
        self.inference_status = "failed"
        if not self._closing and not self._recovering:
            self._recovering = True
            self._spawn_worker(self.applied_options)

    def poll(self):
        with self.lock:
            self._worker_events()
            for job in self.jobs.values():
                before = dict(job)
                identity = job["id"]
                path = Path(job["results_path"])
                try:
                    replay = self.replays.get(identity)
                    if replay is None and path.is_file():
                        scene = self.get_map(job["map_id"])
                        replay = XNavReplay(path, live=True)
                        self._validate_replay(scene, replay)
                        self.replays[identity] = replay
                    elif replay is not None:
                        replay.refresh()
                    if replay is not None:
                        self._apply_replay(job, replay)
                except (OSError, ValueError, RuntimeError, KeyError) as error:
                    job["error"] = f"Cannot read committed results: {error}"
                    if identity in self.processes:
                        self.stop(identity)
                    else:
                        job["status"] = "failed"
                if before != job:
                    self._save(job)
            recycle = False
            for identity, message in self.finished.items():
                self._complete_job(self.jobs[identity], message)
                recycle = recycle or not message.get("reusable", False)
            self.finished.clear()
            self._poll_finalizations()
            if self.worker is not None:
                if self.active_id is not None:
                    deadline = self.stop_deadlines.get(self.active_id)
                    if deadline and time.monotonic() >= deadline[0]:
                        sig = signal.SIGTERM if deadline[1] == 0 else signal.SIGKILL
                        self.worker.signal_group(sig)
                        self.stop_deadlines[self.active_id] = (
                            time.monotonic() + 5,
                            deadline[1] + 1,
                        )
                code = self.worker.process.poll()
                if code is not None:
                    self.inference_error = (
                        self.inference_error or f"Model worker exited with code {code}"
                    )
                    if self.active_id in self.processes:
                        job = self.jobs[self.active_id]
                        self._complete_job(
                            job,
                            {
                                "status": "failed",
                                "error": (
                                    f"Inference worker exited with code {code}.\n"
                                    f"{self.log(job['id'])[-4000:]}"
                                ),
                            },
                        )
                    recycle = True
            if recycle:
                self._recover_worker()

    def info(self, identity):
        with self.lock:
            if identity not in self.jobs:
                raise HTTPException(404, "Job not found")
            job = dict(self.jobs[identity])
            replay = self.replays.get(identity)
            job["tracking"] = (
                bool(replay.results["tracking"]["enabled"])
                if replay
                else bool(job["options"].get("track"))
            )
            job["device"] = (
                replay.results.get("device", "unknown")
                if replay
                else self.inference_device
            )
            return JobInfo.model_validate(job).model_dump()

    def stop(self, identity):
        with self.lock:
            job = self.info(identity)
            if job["readonly"]:
                raise HTTPException(409, "Imported replay is read-only")
            if identity in self._finalizations:
                self.jobs[identity].update(status="stopping", stop_requested=True)
                self._finalizations[identity][1].set()
                self._save(self.jobs[identity])
                return self.info(identity)
            if identity in self.processes and identity not in self.stop_deadlines:
                self.jobs[identity]["status"] = "stopping"
                self.jobs[identity]["stop_requested"] = True
                self.stop_deadlines[identity] = (
                    time.monotonic() + self.stop_timeout,
                    0,
                )
                # The worker may still be reading a queued run command. Wait for
                # job_started before delivering SIGINT to its inference frame.
                if job["status"] == "running":
                    self._interrupt(self.jobs[identity])
                self._save(self.jobs[identity])
            return self.info(identity)

    def archive_path(self, identity):
        with self.lock:
            job = self.info(identity)
            if job["readonly"]:
                raise HTTPException(404, "Imported replay has no job archive")
            if job["status"] != "complete":
                raise HTTPException(409, "Job has not completed successfully")
            if job["options"].get("no_csv"):
                raise HTTPException(409, "ZIP download is disabled for no_csv jobs")
            path = self.root / "jobs" / identity / "3dbbox.zip"
            if path.is_symlink() or not path.is_file():
                raise HTTPException(404, "Job 3dbbox archive not found")
            return path

    def _interrupt(self, job):
        if not job.get("interrupt_sent"):
            with contextlib.suppress(ProcessLookupError):
                self.processes[job["id"]].send_signal(signal.SIGINT)
            job["interrupt_sent"] = True

    def log(self, identity):
        self.info(identity)
        path = self.root / "jobs" / identity / "inference.log"
        if not path.exists():
            return ""
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 12000))
            return stream.read().decode("utf-8", errors="replace")

    def replay(self, identity):
        self.info(identity)
        if identity not in self.replays:
            raise HTTPException(404, "Waiting for the first committed result")
        return self.replays[identity]

    def frame(self, identity, index):
        with self.lock:
            replay = self.replay(identity)
            if not 0 <= index < len(replay.frames):
                raise HTTPException(404, "Frame has not been committed")
            frame = timestamps(replay.frames[index])
            key = processing_frame_id(frame)
            return {
                **frame,
                "index": index,
                "frame_id": key,
                "image_frame_id": frame.get("image_frame_id"),
                "lidar_keyframe_id": frame.get("lidar_keyframe_id", key),
            }

    def frame_page(self, identity, offset, limit):
        with self.lock:
            replay = self.replay(identity)
            frames = []
            for index in range(offset, min(offset + limit, len(replay.frames))):
                frame = replay.frames[index]
                key = processing_frame_id(frame)
                frames.append(
                    {
                        "index": index,
                        "frame_id": key,
                        "image_frame_id": frame.get("image_frame_id"),
                        "lidar_keyframe_id": frame.get("lidar_keyframe_id", key),
                        "time_ns": str(frame["time_ns"]),
                        "position": [row[3] for row in frame["T_map_base"][:3]],
                    }
                )
            return {"total": len(replay.frames), "offset": offset, "frames": frames}

    def box_page(self, identity, kind, offset, limit):
        """Project one page from the replay cache, never from in-progress CSVs."""
        with self.lock:
            replay = self.replay(identity)
            if kind == "owl_2dbbox":
                parameters = replay.results["parameters"]
                entries = (
                    (frame, observation, index)
                    for frame in replay.frames
                    for observation in frame["observations"]
                    for index in range(len(observation["boxes_2d"]))
                )
                project = partial(
                    _box2d_row,
                    image_size=parameters.get("image_size", -1),
                    sem_map={
                        label: index
                        for index, label in enumerate(parameters.get("labels", []))
                    },
                )
            else:
                entries = _box3d_entries(replay, kind)
                project = _box3d_row
            total, items = 0, []
            for entry in entries:
                if offset <= total < offset + limit:
                    items.append(project(*entry))
                total += 1
            return {
                "status": self.jobs[identity]["status"],
                "processed_frames": len(replay.frames),
                "total": total,
                "offset": offset,
                "limit": limit,
                "items": items,
            }

    def close(self):
        with self.lock:
            self._closing = True
            pending = set(self.processes) | set(self._finalizations)
        for identity in pending:
            self.stop(identity)
        while self.processes or self._finalizations:
            self.poll()
            time.sleep(0.05)
        self._finalizer.shutdown(wait=True)
        with self.lock:
            if self.worker is not None:
                self.worker.close()
                self.worker = None
