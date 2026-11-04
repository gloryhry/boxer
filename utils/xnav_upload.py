"""Durable chunk uploads and single-owner ZIP validation/publication.

Only committed chunks are acknowledged. A published directory carries an owner
marker so a restart between rename, map registration and session commit can
finish publication with the same map identity.
"""

import copy
import errno
import hashlib
import json
import logging
import math
import os
import re
import shutil
import stat
import struct
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from utils.xnav_result_store import _atomic_file, _atomic_json, _sync_directory
from utils.xnav_service import MapInfo
from utils.xnav_web_scene import MapScene

LOG = logging.getLogger(__name__)
HASH = r"^[a-fA-F0-9]{64}$"
MARKER = ".boxer-upload.json"
BUFFER = 1024 * 1024


def utc(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def storage_error(error):
    code = 507 if error.errno in (errno.ENOSPC, errno.EDQUOT) else 400
    return HTTPException(code, str(error))


class UploadLimits(BaseModel):
    chunk_size: int = Field(default=8 * 2**20, ge=1, le=64 * 2**20)
    max_zip_bytes: int = Field(default=20 * 2**30, ge=1)
    max_expanded_bytes: int = Field(default=100 * 2**30, ge=1)
    max_files: int = Field(default=200_000, ge=1)
    ttl_seconds: int = Field(default=86400, ge=1)

    @classmethod
    def from_env(cls):
        return cls(
            **{
                name: int(os.environ[f"BOXER_UPLOAD_{name.upper()}"])
                for name in cls.model_fields
                if f"BOXER_UPLOAD_{name.upper()}" in os.environ
            }
        )


class UploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str = Field(min_length=5, max_length=240)
    size: int = Field(gt=0, strict=True)
    sha256: str = Field(pattern=HASH)

    @field_validator("filename")
    @classmethod
    def zip_filename(cls, value):
        if (
            not value.lower().endswith(".zip")
            or any(c in value for c in "/\\\x00")
            or any(ord(c) < 32 for c in value)
        ):
            raise ValueError("A ZIP basename is required")
        return value

    @field_validator("sha256")
    @classmethod
    def lowercase_hash(cls, value):
        return value.lower()


class UploadInfo(BaseModel):
    id: str
    filename: str
    size: int
    sha256: str
    chunk_size: int
    total_chunks: int
    received_chunks: list[int]
    received_bytes: int
    status: Literal["uploading", "processing", "complete", "failed", "cancelled"]
    stage: str
    progress: float
    created_at: str
    expires_at: str
    error: str | None = None
    error_code: int | None = None
    map: MapInfo | None = None


class UploadList(BaseModel):
    limits: UploadLimits
    uploads: list[UploadInfo]


class Interrupted(Exception):
    """A shutdown or cancellation must not be reported as invalid input."""


class RepairChunks(Exception):
    pass


class ChunkWriter:
    def __init__(self, owner, identity, index, digest, expected):
        self.owner, self.identity, self.index = owner, identity, index
        self.digest, self.expected = digest, expected
        self.size, self.hasher = 0, hashlib.sha256()
        self.lock = threading.RLock()
        self.path = owner.folder(identity) / f".{index}.{uuid.uuid4().hex}.tmp"
        self.stream = self.path.open("xb")

    def write(self, data):
        with self.lock:
            self.owner.check(self.identity, receiving=True)
            if self.size + len(data) > self.expected:
                raise HTTPException(413, "Chunk exceeds its expected length")
            self.stream.write(data)
            self.hasher.update(data)
            self.size += len(data)

    def finish(self):
        with self.lock:
            if self.size != self.expected:
                raise HTTPException(422, "Incorrect chunk length")
            if self.hasher.hexdigest() != self.digest:
                raise HTTPException(422, "Chunk SHA-256 mismatch")
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
            with self.owner.lock:
                self.owner.check(self.identity, receiving=True)
                record = copy.deepcopy(self.owner.sessions[self.identity])
                destination = self.owner.folder(self.identity) / f"{self.index}.part"
                os.replace(self.path, destination)
                _sync_directory(destination.parent)
                record["chunks"][str(self.index)] = {
                    "size": self.size,
                    "sha256": self.digest,
                }
                record.update(
                    last_activity=self.owner.clock(),
                    status="uploading",
                    stage="uploading",
                    error=None,
                    error_code=None,
                )
                self.owner.save(record)
                return self.owner.info(record)

    def abort(self):
        with self.lock:
            try:
                self.stream.close()
                self.path.unlink(missing_ok=True)
            finally:
                with self.owner.lock:
                    self.owner.writers.get(self.identity, set()).discard(self.index)
                    self.owner.condition.notify_all()


class UploadManager:
    def __init__(self, map_root, maps, *, limits=None, clock=time.time):
        self.map_root = Path(map_root).expanduser().resolve()
        self.root = self.map_root / ".boxer_uploads"
        self.maps = maps
        self.limits = limits or UploadLimits.from_env()
        self.clock = clock
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.sessions = {}
        self.writers = {}
        self.loaded = False
        self.stop = threading.Event()
        self.thread = None
        self.processing = None

    def folder(self, identity):
        return self.root / identity

    def target(self, record):
        # Bound encoded bytes as well as characters for Linux NAME_MAX.
        stem = re.sub(r"[^\w .-]", "_", record["filename"][:-4]).strip(" .")
        stem = stem.encode("utf-8")[:150].decode("utf-8", errors="ignore") or "map"
        return self.map_root / "uploads" / f"{stem}-{record['id']}"

    def save(self, record):
        _atomic_json(self.folder(record["id"]) / "session.json", record)
        self.sessions[record["id"]] = record

    def load(self):
        with self.lock:
            if self.loaded:
                return
            if self.root.is_symlink():
                raise HTTPException(400, "Upload staging directory cannot be a symlink")
            if self.root.exists():
                for folder in self.root.iterdir():
                    if folder.is_symlink() or not re.fullmatch(
                        r"[a-f0-9]{32}", folder.name
                    ):
                        continue
                    path = folder / "session.json"
                    try:
                        record = json.loads(path.read_text())
                        UploadRequest.model_validate(
                            {key: record[key] for key in ("filename", "size", "sha256")}
                        )
                        if record["id"] != folder.name:
                            raise ValueError("Session identity mismatch")
                        self.sessions[folder.name] = record
                        changed = False
                        if record["status"] not in ("complete", "cancelled"):
                            chunks = {}
                            for index, chunk in record["chunks"].items():
                                part = folder / f"{int(index)}.part"
                                if (
                                    part.is_file()
                                    and not part.is_symlink()
                                    and part.stat().st_size == chunk["size"]
                                ):
                                    chunks[index] = chunk
                            changed = record["chunks"] != chunks
                            record["chunks"] = chunks
                        for temp in folder.glob(".*.tmp"):
                            temp.unlink(missing_ok=True)
                        for part in folder.glob("*.part"):
                            if part.stem not in record["chunks"]:
                                part.unlink()
                        if record["status"] in ("complete", "cancelled"):
                            self.clean_data(folder.name)
                        if changed:
                            self.save(record)
                    except OSError:
                        # Reading a committed session must not require free disk
                        # space. Keep recovered state if cleanup/reconciliation
                        # cannot yet be saved; the next restart rechecks it.
                        LOG.warning(
                            "Upload %s recovery needs storage cleanup", folder.name
                        )
                    except (ValueError, KeyError, TypeError):
                        self.sessions.pop(folder.name, None)
                        LOG.exception("Cannot recover upload %s", folder.name)
            self.loaded = True

    def open(self):
        self.load()
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            self.stop.clear()
            self.thread = threading.Thread(
                target=self.run, name="map-uploads", daemon=True
            )
            self.thread.start()

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=10)

    def expired(self, record):
        return (
            record["status"] != "processing"
            and not self.writers.get(record["id"])
            and self.clock() >= record["last_activity"] + self.limits.ttl_seconds
            # A renamed map still needs registration even after a long outage.
            and not (record["status"] != "complete" and self.target(record).exists())
        )

    def require(self, identity):
        self.load()
        record = self.sessions.get(identity)
        if record is None or self.expired(record):
            raise HTTPException(404, "Upload not found or expired")
        return record

    def info(self, record):
        received = sum(part["size"] for part in record["chunks"].values())
        return UploadInfo(
            **{
                key: record[key]
                for key in (
                    "id",
                    "filename",
                    "size",
                    "sha256",
                    "chunk_size",
                    "total_chunks",
                    "status",
                    "stage",
                    "error",
                    "error_code",
                    "map",
                )
            },
            received_chunks=sorted(map(int, record["chunks"])),
            received_bytes=received,
            progress=received / record["size"]
            if record["status"] == "uploading"
            else record["progress"],
            created_at=utc(record["created"]),
            expires_at=utc(record["last_activity"] + self.limits.ttl_seconds),
        ).model_dump()

    def get(self, identity):
        with self.lock:
            return self.info(self.require(identity))

    def list(self):
        self.load()
        with self.lock:
            return {
                "limits": self.limits.model_dump(),
                "uploads": [
                    self.info(record)
                    for record in reversed(list(self.sessions.values()))
                    if not self.expired(record)
                ],
            }

    def space(self, required):
        if shutil.disk_usage(self.map_root).free < required:
            raise HTTPException(507, "Insufficient space for upload or extraction")

    def create(self, body):
        self.load()
        if body.size > self.limits.max_zip_bytes:
            raise HTTPException(413, "ZIP exceeds the configured size limit")
        if not self.map_root.is_dir():
            raise HTTPException(
                400, f"Map root must exist and be writable: {self.map_root}"
            )
        with self.lock:
            self.space(2 * body.size)
            self.root.mkdir(exist_ok=True)
            identity = uuid.uuid4().hex
            self.folder(identity).mkdir()
            now = self.clock()
            record = {
                **body.model_dump(),
                "id": identity,
                "created": now,
                "last_activity": now,
                "chunk_size": self.limits.chunk_size,
                "total_chunks": math.ceil(body.size / self.limits.chunk_size),
                "chunks": {},
                "status": "uploading",
                "stage": "uploading",
                "progress": 0,
                "error": None,
                "error_code": None,
                "map": None,
                "map_id": uuid.uuid4().hex,
            }
            self.save(record)
            _sync_directory(self.root)
            _sync_directory(self.map_root)
            return self.info(record)

    def check(self, identity, *, receiving=False):
        with self.lock:
            record = self.require(identity)
            if self.stop.is_set():
                raise Interrupted()
            if record["status"] == "cancelled":
                raise Interrupted()
            if receiving and record["status"] not in ("uploading", "failed"):
                raise HTTPException(409, "Upload is no longer receiving chunks")

    def begin_chunk(self, identity, index, digest, content_length=None):
        with self.lock:
            record = self.require(identity)
            self.check(identity, receiving=True)
            if not 0 <= index < record["total_chunks"]:
                raise HTTPException(422, "Chunk index out of range")
            if not re.fullmatch(HASH, digest):
                raise HTTPException(422, "A SHA-256 chunk checksum is required")
            digest = digest.lower()
            expected = min(
                record["chunk_size"], record["size"] - index * record["chunk_size"]
            )
            if content_length is not None and content_length != expected:
                raise HTTPException(422, "Incorrect Content-Length for chunk")
            old = record["chunks"].get(str(index))
            if old and old["sha256"] != digest:
                raise HTTPException(
                    409, "Chunk already committed with different content"
                )
            if index in self.writers.get(identity, set()):
                raise HTTPException(
                    409, "Chunk is being written; query status and retry"
                )
            self.space(expected)
            writer = ChunkWriter(self, identity, index, digest, expected)
            self.writers.setdefault(identity, set()).add(index)
            return writer

    def complete(self, identity):
        with self.condition:
            record = copy.deepcopy(self.require(identity))
            if record["status"] in ("complete", "processing"):
                return self.info(record)
            if record["status"] == "cancelled":
                raise HTTPException(409, "Upload was cancelled")
            if self.writers.get(identity):
                raise HTTPException(409, "Chunks are still being written")
            if (
                len(record["chunks"]) != record["total_chunks"]
                and not self.target(record).exists()
            ):
                raise HTTPException(409, "Upload has missing chunks")
            record.update(
                status="processing",
                stage="queued",
                progress=0,
                error=None,
                error_code=None,
                last_activity=self.clock(),
            )
            self.save(record)
            self.condition.notify_all()
            return self.info(record)

    def cancel(self, identity):
        with self.condition:
            record = copy.deepcopy(self.require(identity))
            if record["status"] == "complete" or self.target(record).exists():
                raise HTTPException(
                    409, "Map already published; uploaded maps cannot be deleted here"
                )
            record.update(
                status="cancelled",
                stage="cancelled",
                progress=0,
                last_activity=self.clock(),
            )
            self.save(record)
            if self.processing != identity and not self.writers.get(identity):
                self.clean_data(identity)
            self.condition.notify_all()
            return self.info(record)

    def clean_data(self, identity):
        for path in self.folder(identity).iterdir():
            if path.name == "session.json":
                continue
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)

    def cleanup(self):
        with self.lock:
            for identity, record in list(self.sessions.items()):
                if identity == self.processing or self.writers.get(identity):
                    continue
                try:
                    if self.expired(record):
                        shutil.rmtree(self.folder(identity))
                        self.sessions.pop(identity)
                    elif record["status"] in ("complete", "cancelled"):
                        self.clean_data(identity)
                except OSError:
                    LOG.warning("Cannot clean upload %s; will retry", identity)

    def run(self):
        while not self.stop.is_set():
            try:
                self.cleanup()
                with self.condition:
                    identity = next(
                        (
                            key
                            for key, value in self.sessions.items()
                            if value["status"] == "processing"
                        ),
                        None,
                    )
                    if identity is None:
                        self.condition.wait(timeout=30)
                        continue
                    self.processing = identity
                self.process(identity)
            except Exception:
                LOG.exception("Upload worker failed")
                self.stop.wait(1)
            finally:
                with self.lock:
                    self.processing = None

    def phase(self, identity, stage):
        with self.lock:
            self.check(identity)
            record = copy.deepcopy(self.sessions[identity])
            record.update(stage=stage, progress=0)
            self.save(record)

    def progress(self, identity, done, total):
        with self.lock:
            self.check(identity)
            self.sessions[identity]["progress"] = done / max(1, total)

    def merge(self, identity):
        record = self.sessions[identity]
        destination = self.folder(identity) / "archive.zip"
        self.space(record["size"])
        whole = hashlib.sha256()
        with _atomic_file(destination) as output:
            for index in range(record["total_chunks"]):
                self.check(identity)
                chunk = record["chunks"].get(str(index))
                path = self.folder(identity) / f"{index}.part"
                digest, size = hashlib.sha256(), 0
                if chunk and path.is_file():
                    with path.open("rb") as source:
                        while data := source.read(BUFFER):
                            self.check(identity)
                            output.write(data)
                            whole.update(data)
                            digest.update(data)
                            size += len(data)
                if (
                    not chunk
                    or size != chunk["size"]
                    or digest.hexdigest() != chunk["sha256"]
                ):
                    with self.lock:
                        updated = copy.deepcopy(self.sessions[identity])
                        updated["chunks"].pop(str(index), None)
                        updated.update(
                            status="uploading",
                            stage="uploading",
                            progress=0,
                            error=f"Chunk {index} is missing or damaged; retransmit it",
                            error_code=422,
                        )
                        self.save(updated)
                    path.unlink(missing_ok=True)
                    raise RepairChunks()
                self.progress(
                    identity,
                    min((index + 1) * record["chunk_size"], record["size"]),
                    record["size"],
                )
            if whole.hexdigest() != record["sha256"]:
                raise HTTPException(422, "ZIP SHA-256 mismatch")
        return destination

    def extract(self, identity, archive):
        stage = self.folder(identity) / "extracted"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        self.zip_entry_limit(archive)
        with zipfile.ZipFile(archive) as package:
            entries = package.infolist()
            if len(entries) > self.limits.max_files:
                raise HTTPException(413, "Too many ZIP entries")
            total = sum(entry.file_size for entry in entries)
            if total > self.limits.max_expanded_bytes:
                raise HTTPException(
                    413, "ZIP expanded size exceeds the configured limit"
                )
            self.space(total)
            seen, manifests, paths = set(), [], []
            files = set()
            for entry in entries:
                name = entry.orig_filename
                parts = name.rstrip("/").split("/")
                mode = entry.external_attr >> 16
                kind = stat.S_IFMT(mode)
                if (
                    not name
                    or "\x00" in name
                    or "\\" in name
                    or name.startswith("/")
                    or re.match(r"^[A-Za-z]:", name)
                    or any(part in ("", ".", "..") for part in parts)
                    or kind not in (0, stat.S_IFREG, stat.S_IFDIR)
                    or (kind == stat.S_IFDIR and not entry.is_dir())
                    or MARKER in parts
                ):
                    raise HTTPException(422, f"Unsafe ZIP entry: {name!r}")
                normalized = "/".join(parts)
                if normalized in seen:
                    raise HTTPException(422, f"Duplicate ZIP destination: {name}")
                seen.add(normalized)
                if not entry.is_dir():
                    files.add(normalized)
                if entry.flag_bits & 1 or entry.compress_type not in (
                    zipfile.ZIP_STORED,
                    zipfile.ZIP_DEFLATED,
                ):
                    raise HTTPException(422, "Encrypted or unsupported ZIP compression")
                if parts[-1] == "manifest.txt" and not entry.is_dir():
                    manifests.append(parts)
                paths.append((entry, stage.joinpath(*parts)))
            for name in seen:
                parts = name.split("/")
                if any(
                    "/".join(parts[:index]) in files for index in range(1, len(parts))
                ):
                    raise HTTPException(422, "ZIP file is also used as a directory")
            if len(manifests) != 1 or len(manifests[0]) not in (1, 2):
                raise HTTPException(
                    422,
                    "ZIP must contain exactly one map at root "
                    "or in one wrapper directory",
                )
            done = 0
            for entry, destination in paths:
                self.check(identity)
                if entry.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                entry_bytes = 0
                with package.open(entry) as source, destination.open("xb") as output:
                    while data := source.read(BUFFER):
                        self.check(identity)
                        done += len(data)
                        entry_bytes += len(data)
                        if done > self.limits.max_expanded_bytes:
                            raise HTTPException(
                                413, "Actual ZIP expanded size exceeds the limit"
                            )
                        output.write(data)
                        self.progress(identity, done, total)
                    if entry_bytes != entry.file_size:
                        raise HTTPException(422, "ZIP entry expanded length mismatch")
                    output.flush()
                    os.fsync(output.fileno())
            # Persist the entire directory tree before a durable rename.
            for folder, _, _ in os.walk(stage, topdown=False):
                self.check(identity)
                _sync_directory(folder)
            return stage.joinpath(*manifests[0][:-1])

    def zip_entry_limit(self, archive):
        """Count central-directory entries before ZipFile allocates ZipInfo objects.

        Parse the fixed EOCD/ZIP64 records, then walk headers without loading the
        directory in memory. ZipFile still performs the actual archive checks.
        """
        with archive.open("rb") as stream:
            size = archive.stat().st_size
            stream.seek(max(0, size - 65557))
            tail = stream.read()
            offset = tail.rfind(b"PK\x05\x06")
            if offset < 0 or offset + 22 > len(tail):
                raise HTTPException(422, "Invalid ZIP end record")
            end = struct.unpack_from("<4s4H2LH", tail, offset)
            if (
                offset + 22 + end[7] != len(tail)
                or end[1]
                or end[2]
                or end[3] != end[4]
            ):
                raise HTTPException(422, "Invalid or multi-volume ZIP")
            count, length, start = end[4:7]
            absolute = size - len(tail) + offset
            if count == 65535 or length == 0xFFFFFFFF or start == 0xFFFFFFFF:
                stream.seek(max(0, absolute - 20))
                locator = stream.read(20)
                if len(locator) != 20:
                    raise HTTPException(422, "Invalid ZIP64 locator")
                signature, disk, position, disks = struct.unpack("<4sLQL", locator)
                if signature != b"PK\x06\x07" or disk or disks != 1:
                    raise HTTPException(422, "Invalid ZIP64 locator")
                stream.seek(position)
                header = stream.read(56)
                if len(header) != 56:
                    raise HTTPException(422, "Invalid ZIP64 end record")
                values = struct.unpack("<4sQ2H2L4Q", header)
                if (
                    values[0] != b"PK\x06\x06"
                    or values[4]
                    or values[5]
                    or values[6] != values[7]
                ):
                    raise HTTPException(422, "Invalid ZIP64 end record")
                count, length, start = values[7:10]
            if count > self.limits.max_files:
                raise HTTPException(413, "Too many ZIP entries")
            if start + length > size:
                raise HTTPException(422, "Invalid ZIP central directory")
            stream.seek(start)
            boundary, actual = start + length, 0
            while stream.tell() < boundary:
                header = stream.read(46)
                if len(header) != 46 or header[:4] != b"PK\x01\x02":
                    raise HTTPException(422, "Invalid ZIP central directory entry")
                actual += 1
                if actual > self.limits.max_files:
                    raise HTTPException(413, "Too many ZIP entries")
                lengths = struct.unpack_from("<3H", header, 28)
                stream.seek(sum(lengths), os.SEEK_CUR)
            if stream.tell() != boundary or actual != count:
                raise HTTPException(422, "ZIP entry count mismatch")

    def process(self, identity):
        try:
            self.check(identity)
            record = self.sessions[identity]
            target = self.target(record)
            scene = None
            if not target.exists():
                self.phase(identity, "merging")
                archive = self.merge(identity)
                self.phase(identity, "extracting")
                source = self.extract(identity, archive)
                self.phase(identity, "validating")
                scene = self.validate(identity, source)
                _atomic_json(
                    source / MARKER, {"upload_id": identity, "sha256": record["sha256"]}
                )
                self.phase(identity, "publishing")
                with self.lock:
                    self.check(identity)
                    if target.parent.is_symlink():
                        raise HTTPException(
                            422, "Map upload destination cannot be a symlink"
                        )
                    target.parent.mkdir(exist_ok=True)
                    _sync_directory(self.map_root)
                    if target.exists():
                        raise HTTPException(409, "Map destination already exists")
                    source.rename(target)
                    _sync_directory(source.parent)
                    _sync_directory(target.parent)
            else:
                if target.is_symlink() or target.parent.is_symlink():
                    raise HTTPException(
                        409, "Map upload destination cannot be a symlink"
                    )
                marker = json.loads((target / MARKER).read_text())
                if marker != {"upload_id": identity, "sha256": record["sha256"]}:
                    raise HTTPException(
                        409, "Map destination is owned by another upload"
                    )
                self.phase(identity, "validating")
                scene = self.validate(identity, target)
            with self.lock:
                self.check(identity)
                self.phase(identity, "publishing")
                scene.loader.root = target.resolve()
                info = self.maps.register_map(scene, identity=record["map_id"])
                updated = copy.deepcopy(self.sessions[identity])
                updated.update(
                    status="complete",
                    stage="complete",
                    progress=1,
                    map=info,
                    error=None,
                    error_code=None,
                    last_activity=self.clock(),
                )
                self.save(updated)
            self.clean_data(identity)
        except (Interrupted, RepairChunks):
            pass
        except Exception as error:
            if isinstance(error, HTTPException):
                code, detail = error.status_code, str(error.detail)
            elif isinstance(error, OSError):
                failure = storage_error(error)
                code, detail = failure.status_code, str(failure.detail)
            else:
                code, detail = 422, str(error)
            LOG.warning("Upload %s failed: %s", identity, detail)
            with self.lock:
                record = copy.deepcopy(self.sessions[identity])
                if record["status"] not in ("complete", "cancelled"):
                    record.update(status="failed", error=detail, error_code=code)
                    try:
                        self.save(record)
                    except OSError:
                        # A full filesystem can prevent even the error descriptor
                        # from being committed. Still expose failure to live clients.
                        self.sessions[identity] = record
                        LOG.exception("Cannot persist upload failure %s", identity)
        finally:
            with self.lock:
                if self.sessions[identity][
                    "status"
                ] == "cancelled" and not self.writers.get(identity):
                    self.clean_data(identity)

    def validate(self, identity, root):
        try:
            return MapScene(
                root, progress=lambda done, total: self.progress(identity, done, total)
            )
        except OSError as error:
            if error.errno in (errno.ENOSPC, errno.EDQUOT):
                raise
            raise HTTPException(422, f"Invalid x-nav map: {error}") from error
