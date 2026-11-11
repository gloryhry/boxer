"""Durable orchestration of Boxer, voxel room segmentation and object binding."""

import contextlib
import copy
import json
import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import CancelledError, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException

from utils.potential_field_service import VoxelConnectivityJobRequest, _fingerprint
from utils.xnav_result_store import _atomic_json, _sync_directory
from utils.xnav_room_binding import preview_cloud
from utils.xnav_room_export import export_room_results, recover_room_export
from utils.xnav_service import (
    ACTIVE,
    REPO,
    MapExportInfo,
    RoomBindingRow,
    WorkflowInfo,
    _box3d_row,
)

LOG = logging.getLogger(__name__)
TERMINAL = {"complete", "failed", "interrupted"}


def _now():
    return datetime.now(timezone.utc).isoformat()


class MapWorkflows:
    def __init__(self, manager, dispatcher, rooms, output_dir):
        self.manager, self.dispatcher, self.rooms = manager, dispatcher, rooms
        self.root = Path(output_dir).expanduser().resolve() / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.records = {}
        self.dirty = set()
        self.process = None
        self.binding_id = None
        self.stop_deadline = None
        self.closing = False
        self.cloud_cache = OrderedDict()
        self.exports = {}
        self._exporter = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="xnav-room-export"
        )
        self._recover()

    def _path(self, identity, name):
        return self.root / identity / name

    def _save(self, record):
        try:
            _atomic_json(self._path(record["job_id"], "workflow.json"), record)
        except OSError:
            self.dirty.add(record["job_id"])
            raise
        self.dirty.discard(record["job_id"])
        self.records[record["job_id"]] = record

    def _recover(self):
        for path in self.root.glob("*/workflow.json"):
            if path.is_symlink() or path.parent.is_symlink():
                continue
            try:
                record = json.loads(path.read_bytes())
                identity = record["job_id"]
                if identity != path.parent.name:
                    raise ValueError("Workflow identity does not match its directory")
                self.manager.info(identity)
                flow = WorkflowInfo.model_validate(record["workflow"]).model_dump()
                record["workflow"] = flow
                was_active = flow["status"] in ACTIVE
                recovery_error = None
                if flow["map_export"] is not None and flow["status"] != "complete":
                    try:
                        recover_room_export(
                            self._path(identity, "map-export.json"),
                            record["map_root"],
                            identity,
                        )
                    except (OSError, ValueError, KeyError) as error:
                        recovery_error = str(error)
                        flow["map_export"].update(status="failed", error=str(error))
                        flow.update(
                            status="failed",
                            error=f"Room export recovery failed: {error}",
                        )
                        flow["finished_at"] = _now()
                        self._save(record)
                if was_active:
                    flow.update(
                        status="failed" if recovery_error else "interrupted",
                        finished_at=_now(),
                        error=recovery_error
                        or "Service restarted; workflow was not resumed",
                    )
                    for name in ("room_segmentation", "binding", "map_export"):
                        if flow[name] is not None and flow[name]["status"] in {
                            "pending",
                            "running",
                            "stopping",
                        }:
                            flow[name].update(status="interrupted", error=flow["error"])
                    self._path(identity, "room-bindings.staging.json").unlink(
                        missing_ok=True
                    )
                    if flow["binding"]["status"] != "complete":
                        self._path(identity, "room-bindings.json").unlink(
                            missing_ok=True
                        )
                    if flow["map_export"] is not None:
                        self._path(identity, "3dbbox.zip").unlink(missing_ok=True)
                        for temporary in self.root.joinpath(identity).glob(
                            ".3dbbox.zip.*.tmp"
                        ):
                            temporary.unlink(missing_ok=True)
                    self._save(record)
                else:
                    self.records[identity] = record
            except (HTTPException, OSError, ValueError, KeyError):
                LOG.exception("Could not recover workflow %s", path.parent.name)

    @property
    def active_id(self):
        with self.lock:
            return next(
                (
                    key
                    for key, record in self.records.items()
                    if record["workflow"]["status"] in ACTIVE
                ),
                None,
            )

    def info(self, identity):
        with self.lock:
            job = self.manager.info(identity)
            record = self.records.get(identity)
            job["workflow"] = copy.deepcopy(record["workflow"]) if record else None
            return job

    def list(self):
        with self.lock, self.manager.lock:
            return [self.info(key) for key in reversed(self.manager.jobs)]

    def start(self, request):
        from voxel_connectivity import load_voxel_connectivity_config

        with self.lock, self.dispatcher.lock, self.manager.lock:
            if self.closing:
                raise HTTPException(503, "Workflow service is shutting down")
            if self.active_id is not None:
                raise HTTPException(409, "A map workflow is already running")
            self.manager.prepare_start(request)
            room_request = VoxelConnectivityJobRequest(map_id=request.map_id)
            scene = self.dispatcher.prepare_start(self.rooms, room_request)
            # Fail invalid configuration before either child is dispatched.
            load_voxel_connectivity_config()
            identity = uuid.uuid4().hex
            record = dict(
                job_id=identity,
                map_id=request.map_id,
                source_signature=scene.signature,
                map_root=str(scene.loader.root),
                global_map_fingerprint=list(
                    _fingerprint(scene.loader.root / "global_map.pcd")
                ),
                workflow=WorkflowInfo(map_export=MapExportInfo()).model_dump(),
                stop_requested=False,
            )
            self._path(identity, "workflow.json").parent.mkdir()
            self._save(record)
            try:
                self.manager.start(request, identity=identity, defer_archive=True)
            except Exception:
                self.records.pop(identity, None)
                self._path(identity, "workflow.json").unlink(missing_ok=True)
                raise
            try:
                child = self.rooms.start(
                    room_request, scene.loader.root, workflow_artifacts=True
                )
            except Exception as error:
                record["workflow"]["room_segmentation"].update(
                    status="failed", error=f"Could not start room segmentation: {error}"
                )
            else:
                self._apply_room(record, child)
            record["workflow"]["status"] = "running"
            self._save(record)
            return self.info(identity)

    @staticmethod
    def _apply_room(record, child):
        record["workflow"]["room_segmentation"].update(
            job_id=child["id"],
            status=child["status"],
            error=child["error"],
            artifacts=child["artifacts"],
        )

    def _record(self, identity):
        self.manager.info(identity)
        if identity not in self.records:
            raise HTTPException(404, "Job has no associated room workflow")
        return self.records[identity]

    def _check_source(self, record):
        scene = self.manager.get_map(record["map_id"])
        scene.check_source()
        if (
            scene.signature != record["source_signature"]
            or list(_fingerprint(scene.loader.root / "global_map.pcd"))
            != record["global_map_fingerprint"]
        ):
            raise ValueError(
                "Map changed during the workflow; results were not published"
            )

    def _start_binding(self, record, boxer):
        identity = record["job_id"]
        stage = record["workflow"]["binding"]
        try:
            self._check_source(record)
            child_id = record["workflow"]["room_segmentation"]["job_id"]
            ply = self.rooms.artifact(child_id, "result_colored.ply")
            rooms = json.loads(self.rooms.artifact(child_id, "rooms.json").read_bytes())
            with self.manager.lock:
                replay = self.manager.replay(identity)
                if replay.results.get("status") != "complete":
                    raise ValueError("Boxer result is not a final committed snapshot")
                objects = [
                    _box3d_row(item)
                    for item in replay.results.get("objects", [])
                    if item.get("state") != "TENTATIVE"
                ]
            payload = dict(
                job_id=identity,
                map_id=record["map_id"],
                segmentation_job_id=child_id,
                source_signature=record["source_signature"],
                global_map_fingerprint=record["global_map_fingerprint"],
                tracking_enabled=boxer["tracking"],
                objects=objects,
                rooms=rooms,
                ply_path=str(ply),
            )
            source = self._path(identity, "binding-input.json")
            _atomic_json(source, payload)
            # Use the same CPU environment as room segmentation, including overrides.
            python = self.rooms._worker_command()[0]
            with self._path(identity, "binding.log").open("wb") as log:
                self.process = subprocess.Popen(
                    [
                        python,
                        "-u",
                        "-m",
                        "utils.xnav_room_binding",
                        "--input",
                        str(source),
                        "--output",
                        str(self._path(identity, "room-bindings.staging.json")),
                    ],
                    cwd=REPO,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={**os.environ, "BOXER_PF_OWNER_PID": str(os.getpid())},
                )
            self.binding_id = identity
            stage["status"] = "running"
        except Exception as error:
            stage.update(status="failed", error=f"Could not start binding: {error}")

    def _signal_binding(self, sig):
        if self.process is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, sig)

    def _poll_binding(self, record):
        if self.binding_id != record["job_id"] or self.process is None:
            return
        code = self.process.poll()
        if code is None:
            if (
                self.stop_deadline is not None
                and time.monotonic() >= self.stop_deadline
            ):
                self._signal_binding(signal.SIGKILL)
                self.stop_deadline = None
            return
        stage = record["workflow"]["binding"]
        staging = self._path(record["job_id"], "room-bindings.staging.json")
        if record["stop_requested"]:
            stage.update(status="interrupted", error="Stopped by request")
        elif code:
            stage.update(
                status="failed",
                error=f"Binding worker exited with status {code}. See binding.log.",
            )
        else:
            try:
                self._check_source(record)
                result = json.loads(staging.read_bytes())
                if (
                    result["job_id"] != record["job_id"]
                    or result["segmentation_job_id"]
                    != record["workflow"]["room_segmentation"]["job_id"]
                ):
                    raise ValueError("Binding output references another task")
                for item in result["items"]:
                    RoomBindingRow.model_validate(item)
                staging.replace(self._path(record["job_id"], "room-bindings.json"))
                _sync_directory(staging.parent)
                url = f"/api/jobs/{record['job_id']}/room-bindings.json"
                stage.update(
                    status="complete",
                    reason=result["summary"]["reason"],
                    artifacts={"room-bindings.json": url},
                )
            except Exception as error:
                stage.update(
                    status="failed", error=f"Could not publish binding: {error}"
                )
        staging.unlink(missing_ok=True)
        if stage["status"] != "complete":
            self._path(record["job_id"], "room-bindings.json").unlink(missing_ok=True)
        self.process = self.binding_id = self.stop_deadline = None

    def _start_export(self, record, boxer):
        identity = record["job_id"]
        stage = record["workflow"]["map_export"]
        try:
            child = record["workflow"]["room_segmentation"]["job_id"]
            sources = {
                name: self.rooms.artifact(child, name)
                for name in (
                    "result_colored.ply",
                    "rooms.json",
                    "summary.json",
                )
            }
            if (
                "room_polygons.json"
                in record["workflow"]["room_segmentation"]["artifacts"]
            ):
                sources["room_polygons.json"] = self.rooms.artifact(
                    child, "room_polygons.json"
                )
            bound = record["workflow"]["binding"]["status"] == "complete"
            if bound:
                sources["room-bindings.json"] = self.binding_path(identity)
            archive = None
            if (
                bound
                and boxer["status"] == "complete"
                and not boxer["options"].get("no_csv")
            ):
                archive = self._path(identity, "3dbbox.zip")
            snapshot = copy.deepcopy(record)
            provenance = dict(
                job_id=identity,
                map_id=record["map_id"],
                segmentation_job_id=child,
                algorithm="voxel_connectivity",
                source_signature=record["source_signature"],
                global_map_fingerprint=record["global_map_fingerprint"],
                exported_at=_now(),
            )
            cancel = threading.Event()
            future = self._exporter.submit(
                export_room_results,
                record["map_root"],
                sources,
                provenance,
                self._path(identity, "map-export.json"),
                cancel,
                lambda: self._check_source(snapshot),
                archive=archive,
                publish_lock=self.lock,
            )
            self.exports[identity] = future, cancel
            stage["status"] = "running"
        except Exception as error:
            stage.update(status="failed", error=f"Could not start map export: {error}")

    def _poll_export(self, record):
        identity = record["job_id"]
        pending = self.exports.get(identity)
        if pending is None or not pending[0].done():
            return
        future, _ = self.exports.pop(identity)
        stage = record["workflow"]["map_export"]
        try:
            future.result()
        except CancelledError:
            stage.update(status="interrupted", error="Map export cancelled")
        except Exception as error:
            stage.update(
                status="failed", error=f"Could not export map results: {error}"
            )
        else:
            if record["stop_requested"]:
                stage.update(status="interrupted", error="Stopped by request")
            else:
                stage["status"] = "complete"
        if stage["status"] != "complete":
            self._path(identity, "3dbbox.zip").unlink(missing_ok=True)

    def _settle(self, record, boxer, before):
        """Wait for computation, then publish map results, then finish the workflow."""
        flow = record["workflow"]
        room, binding, exported = (
            flow["room_segmentation"],
            flow["binding"],
            flow["map_export"],
        )
        if (
            boxer["status"] not in TERMINAL
            or room["status"] not in TERMINAL
            or binding["status"] not in TERMINAL | {"skipped"}
        ):
            return
        if exported is not None:
            if room["status"] != "complete":
                exported["status"] = "skipped"
            elif exported["status"] == "pending":
                # A binding snapshot must be durable before a background copy uses it.
                if (
                    before["workflow"]["binding"]["status"] == binding["status"]
                    and record["job_id"] not in self.dirty
                ):
                    self._start_export(record, boxer)
            if exported["status"] not in TERMINAL | {"skipped"}:
                return
        stages = [boxer, room, binding, *([exported] if exported is not None else [])]
        states = {stage["status"] for stage in stages}
        flow["status"] = (
            "complete"
            if states == {"complete"}
            else "failed"
            if "failed" in states
            else "interrupted"
        )
        flow["error"] = next(
            (stage["error"] for stage in reversed(stages) if stage.get("error")), None
        )

    def poll(self):
        with self.lock:
            for record in list(self.records.values()):
                flow = record["workflow"]
                if flow["status"] not in ACTIVE:
                    if record["job_id"] in self.dirty:
                        try:
                            self._save(record)
                        except OSError:
                            LOG.exception(
                                "Could not persist workflow %s", record["job_id"]
                            )
                    continue
                before = copy.deepcopy(record)
                try:
                    boxer = self.manager.info(record["job_id"])
                    child_id = flow["room_segmentation"]["job_id"]
                    if child_id is not None:
                        self._apply_room(record, self.rooms.info(child_id))
                    self._poll_binding(record)
                    self._poll_export(record)
                    room = flow["room_segmentation"]
                    binding = flow["binding"]
                    both_done = (
                        boxer["status"] in TERMINAL and room["status"] in TERMINAL
                    )
                    if record["stop_requested"]:
                        if binding["status"] == "pending":
                            binding.update(
                                status="interrupted", error="Stopped by request"
                            )
                        exported = flow["map_export"]
                        if exported is not None and exported["status"] == "pending":
                            exported.update(
                                status="interrupted", error="Stopped by request"
                            )
                        if (
                            both_done
                            and self.binding_id != record["job_id"]
                            and record["job_id"] not in self.exports
                        ):
                            flow.update(
                                status="interrupted", error="Stopped by request"
                            )
                    elif boxer["status"] == "complete" and room["status"] == "complete":
                        if binding["status"] == "pending":
                            self._start_binding(record, boxer)
                    elif boxer["status"] in {"failed", "interrupted"} or room[
                        "status"
                    ] in {"failed", "interrupted"}:
                        if binding["status"] == "pending":
                            binding.update(status="skipped", reason="upstream_failed")
                        flow["error"] = boxer["error"] or room["error"]
                    if not record["stop_requested"]:
                        self._settle(record, boxer, before)
                    if flow["status"] in TERMINAL:
                        flow["finished_at"] = _now()
                    if record != before or record["job_id"] in self.dirty:
                        self._save(record)
                except Exception as error:
                    # Keep active children owned and the monitor alive on I/O errors.
                    LOG.exception("Could not advance workflow %s", record["job_id"])
                    flow["error"] = str(error)
                    self.dirty.add(record["job_id"])
                    if (
                        flow["binding"]["status"] == "complete"
                        and before["workflow"]["binding"]["status"] != "complete"
                    ):
                        # Completion must be durable before the artifact is exposed.
                        with contextlib.suppress(OSError):
                            self._path(record["job_id"], "room-bindings.json").unlink(
                                missing_ok=True
                            )
                        flow["binding"].update(
                            status="failed", error=str(error), artifacts={}
                        )
                        if flow["map_export"] is None:
                            flow.update(status="failed", finished_at=_now())
                    exported = flow["map_export"]
                    previous_export = before["workflow"]["map_export"]
                    if (
                        exported is not None
                        and exported["status"] == "complete"
                        and previous_export["status"] != "complete"
                    ):
                        with contextlib.suppress(OSError):
                            self._path(record["job_id"], "3dbbox.zip").unlink(
                                missing_ok=True
                            )
                        exported.update(status="failed", error=str(error))
                        flow.update(status="failed", finished_at=_now())

    def stop(self, identity):
        with self.lock:
            if identity not in self.records:
                return self.manager.stop(identity)
            record = self.records[identity]
            if record["workflow"]["status"] not in ACTIVE:
                return self.info(identity)
            record["stop_requested"] = True
            record["workflow"]["status"] = "stopping"
            actions = [lambda: self._save(record), lambda: self.manager.stop(identity)]
            child_id = record["workflow"]["room_segmentation"]["job_id"]
            if child_id is not None:
                actions.append(lambda: self.rooms.stop(child_id))
            failures = []
            for action in actions:
                try:
                    action()
                except (OSError, HTTPException) as error:
                    failures.append(error)
            if self.binding_id == identity:
                record["workflow"]["binding"]["status"] = "stopping"
                self.stop_deadline = time.monotonic() + 5
                self._signal_binding(signal.SIGTERM)
            if identity in self.exports:
                record["workflow"]["map_export"]["status"] = "stopping"
                self.exports[identity][1].set()
            try:
                self._save(record)
            except OSError as error:
                failures.append(error)
            if failures:
                raise failures[0]
            return self.info(identity)

    def archive_path(self, identity):
        with self.lock:
            self.manager.info(identity)
            record = self.records.get(identity)
            if record is not None and record["workflow"]["map_export"] is not None:
                if record["workflow"]["status"] != "complete":
                    raise HTTPException(409, "Workflow has not completed successfully")
            elif self.manager.jobs[identity].get("defer_archive"):
                raise HTTPException(409, "Workflow archive is not available")
            return self.manager.archive_path(identity)

    def room_result(self, identity):
        with self.lock:
            record = self._record(identity)
            stage = record["workflow"]["room_segmentation"]
            if stage["status"] != "complete":
                raise HTTPException(
                    409, "Room segmentation has not completed successfully"
                )
            child = stage["job_id"]
            return dict(
                job_id=identity,
                map_id=record["map_id"],
                segmentation_job_id=child,
                rooms=json.loads(self.rooms.artifact(child, "rooms.json").read_bytes()),
                summary=json.loads(
                    self.rooms.artifact(child, "summary.json").read_bytes()
                ),
                artifacts=stage["artifacts"],
                algorithm="voxel_connectivity",
            )

    def room_cloud(self, identity, room_id):
        result = self.room_result(identity)
        if (
            room_id is not None
            and room_id >= 0
            and room_id not in {room["id"] for room in result["rooms"]}
        ):
            raise HTTPException(404, "Room not found")
        key = identity, room_id
        with self.lock:
            if key in self.cloud_cache:
                self.cloud_cache.move_to_end(key)
                return self.cloud_cache[key]
        path = self.rooms.artifact(result["segmentation_job_id"], "result_colored.ply")
        data = preview_cloud(path, room_id)
        with self.lock:
            self.cloud_cache[key] = data
            while sum(len(value) for value in self.cloud_cache.values()) > 32 * 1024**2:
                self.cloud_cache.popitem(last=False)
        return data

    def binding_path(self, identity):
        with self.lock:
            record = self._record(identity)
            if record["workflow"]["binding"]["status"] != "complete":
                raise HTTPException(409, "Room binding has not completed successfully")
            path = self._path(identity, "room-bindings.json")
            if path.is_symlink() or not path.is_file():
                raise HTTPException(404, "Room binding artifact not found")
            return path

    def binding_page(self, identity, offset, limit, room_id):
        result = json.loads(self.binding_path(identity).read_bytes())
        items = result["items"]
        if room_id is not None:
            owner = None if room_id == -1 else room_id
            items = [item for item in items if item["room_id"] == owner]
        return dict(
            job_id=identity,
            map_id=result["map_id"],
            segmentation_job_id=result["segmentation_job_id"],
            status="complete",
            summary=result["summary"],
            total=len(items),
            offset=offset,
            limit=limit,
            items=items[offset : offset + limit],
        )

    def close(self):
        with self.lock:
            self.closing = True
            for identity, record in self.records.items():
                if record["workflow"]["status"] in ACTIVE:
                    try:
                        self.stop(identity)
                    except (OSError, HTTPException):
                        LOG.exception("Could not persist workflow stop %s", identity)
            process = self.process
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._signal_binding(signal.SIGKILL)
                process.wait(timeout=5)
        self._exporter.shutdown(wait=True, cancel_futures=True)
        self.poll()
