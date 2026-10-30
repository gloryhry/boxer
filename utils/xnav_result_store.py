"""Durable frame journal with a small atomic committed-prefix descriptor.

Running/failed results use schema 3 on disk; readers normalize them to the
schema-2 dictionary consumed by existing replay and cache code. Successful
export replaces the descriptor with a standalone schema-2 JSON document.
"""

import contextlib
import copy
import json
import os
import re
import uuid
from pathlib import Path


def _encode(value):
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def _atomic_file(path):
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("xb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        _sync_directory(path.parent)
    finally:
        with contextlib.suppress(OSError):
            temp.unlink(missing_ok=True)


def _atomic_json(path, value):
    with _atomic_file(path) as stream:
        stream.write(_encode(value) + b"\n")


def _entries(path, descriptor, offset=0):
    remaining = descriptor["committed_bytes"]
    count = 0
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            line = stream.readline(remaining + 1)
            if not line.endswith(b"\n") or len(line) > remaining:
                raise ValueError("Invalid or truncated committed x-nav journal")
            remaining -= len(line)
            try:
                entry = json.loads(line)
            except (ValueError, UnicodeError) as error:
                raise ValueError("Invalid committed x-nav journal entry") from error
            if not isinstance(entry, dict) or not isinstance(entry.get("frame"), dict):
                raise ValueError("Invalid committed x-nav frame")
            count += 1
            yield entry
    if count != descriptor["committed_frames"]:
        raise ValueError("Invalid committed x-nav frame count")


def _bundle_paths(path, descriptor):
    run_id = descriptor.get("run_id", "")
    if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("Invalid x-nav journal run ID")
    for name in ("committed_frames", "committed_bytes"):
        value = descriptor.get(name)
        if type(value) is not int or value < 0:
            raise ValueError("Invalid committed x-nav journal boundary")
    paths = []
    for key, suffix in (("metadata", "metadata.json"), ("journal", "frames.jsonl")):
        expected = f"{path.stem}.{run_id}.{suffix}"
        if descriptor.get(key) != expected:
            raise ValueError("Invalid x-nav journal side-file name")
        paths.append(path.parent / expected)
    if descriptor.get("status") not in ("running", "failed", "interrupted"):
        raise ValueError("Invalid x-nav journal status")
    return paths


def _read_bundle(path, descriptor):
    metadata_path, journal_path = _bundle_paths(path, descriptor)
    result = json.loads(metadata_path.read_bytes())
    if not isinstance(result, dict) or result.get("schema_version") != 2:
        raise ValueError("Invalid x-nav journal metadata")
    result["frames"] = []
    for entry in _entries(journal_path, descriptor):
        result["frames"].append(entry["frame"])
        result["views"].update(entry["views"])
        result["id_aliases"] = entry["id_aliases"]
        result["elapsed_seconds"] = entry["elapsed_seconds"]
    result["status"] = descriptor["status"]
    if "error" in descriptor:
        result["error"] = descriptor["error"]
    return result


def read_result_file(path):
    """Read a stable committed prefix, retrying a concurrent final replacement."""
    path = Path(path)
    for _ in range(3):
        document = json.loads(path.read_bytes())
        if not isinstance(document, dict):
            raise ValueError("Invalid x-nav result document")
        if document.get("schema_version") != 3:
            return document
        try:
            return _read_bundle(path, document)
        except FileNotFoundError:
            # Final export can remove side files after we opened the old head.
            if json.loads(path.read_bytes()) == document:
                raise
    raise RuntimeError("x-nav result changed repeatedly while opening")


class ResultFollower:
    """Follow committed frames, reading each journal entry only once.

    The final standalone document is read once when it replaces the journal.
    An unchanged file requires only a stat, including after completion.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.results = None
        self._stamp = None
        self._run_id = None
        self._offset = 0

    def poll(self):
        """Return whether a new committed snapshot or status was loaded."""
        for _ in range(3):
            with self.path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
                if stamp == self._stamp:
                    return False
                document = json.load(stream)
            if not isinstance(document, dict):
                raise ValueError("Invalid x-nav result document")
            if document.get("schema_version") != 3:
                self.results = document
                self._run_id = None
                self._offset = 0
            else:
                metadata_path, journal_path = _bundle_paths(self.path, document)
                same_run = document["run_id"] == self._run_id
                offset = self._offset if same_run else 0
                count = len(self.results["frames"]) if same_run else 0
                remaining = {
                    "committed_bytes": document["committed_bytes"] - offset,
                    "committed_frames": document["committed_frames"] - count,
                }
                if min(remaining.values()) < 0:
                    raise ValueError("Committed x-nav journal moved backwards")
                try:
                    result = (
                        dict(self.results)
                        if same_run
                        else json.loads(metadata_path.read_bytes())
                    )
                    if result.get("schema_version") != 2:
                        raise ValueError("Invalid x-nav journal metadata")
                    entries = list(_entries(journal_path, remaining, offset))
                except FileNotFoundError:
                    # Export may have replaced the head and removed side files.
                    continue
                frames = result["frames"] if same_run else []
                views = dict(result.get("views", {}))
                for entry in entries:
                    frames.append(entry["frame"])
                    views.update(entry["views"])
                    result["id_aliases"] = entry["id_aliases"]
                    result["elapsed_seconds"] = entry["elapsed_seconds"]
                result.update(frames=frames, views=views, status=document["status"])
                if "error" in document:
                    result["error"] = document["error"]
                self.results = result
                self._run_id = document["run_id"]
                self._offset = document["committed_bytes"]
            self._stamp = stamp
            return True
        raise RuntimeError("x-nav result changed repeatedly while following")


class ResultStore:
    """Single writer; concurrent readers see only published complete frames."""

    def __init__(self, path, metadata):
        self.path = Path(path)
        self.run_id = uuid.uuid4().hex
        prefix = f"{self.path.stem}.{self.run_id}"
        self.metadata_path = self.path.parent / f"{prefix}.metadata.json"
        self.journal_path = self.path.parent / f"{prefix}.frames.jsonl"
        self.committed_frames = 0
        self.committed_bytes = 0
        self.views = copy.deepcopy(metadata.get("views", {}))
        self.finished = False
        self.failed = False
        self.stream = None
        _atomic_json(self.metadata_path, metadata)
        self.metadata_bytes = self.metadata_path.stat().st_size
        self.stream = self.journal_path.open("x+b")
        try:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            _atomic_json(self.path, self._descriptor())
        except BaseException as error:
            try:
                self.close()
            except Exception as close_error:
                error.add_note(f"Could not close result journal: {close_error}")
            raise

    def _descriptor(self, **updates):
        return {
            "schema_version": 3,
            "run_id": self.run_id,
            "metadata": self.metadata_path.name,
            "journal": self.journal_path.name,
            "committed_frames": self.committed_frames,
            "committed_bytes": self.committed_bytes,
            "status": "running",
            **updates,
        }

    @property
    def result_bytes(self):
        return self.metadata_bytes + self.committed_bytes + self.path.stat().st_size

    def append(self, frame, views, id_aliases, elapsed_seconds):
        if self.finished or self.failed or self.stream is None:
            raise RuntimeError("Cannot append to a closed x-nav result store")
        changed_views = {k: v for k, v in views.items() if self.views.get(k) != v}
        line = (
            _encode(
                {
                    "frame": frame,
                    "views": changed_views,
                    "id_aliases": id_aliases,
                    "elapsed_seconds": elapsed_seconds,
                }
            )
            + b"\n"
        )
        # A previous failed publication may have left an uncommitted tail.
        self.stream.seek(self.committed_bytes)
        self.stream.truncate()
        self.stream.write(line)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        head = self._descriptor(
            committed_frames=self.committed_frames + 1,
            committed_bytes=self.committed_bytes + len(line),
        )
        try:
            _atomic_json(self.path, head)
        except BaseException:
            # A directory fsync can fail after rename has already published it.
            with contextlib.suppress(OSError, ValueError):
                if json.loads(self.path.read_bytes()) == head:
                    self._accept(head, changed_views)
            raise
        self._accept(head, changed_views)

    def _accept(self, head, views):
        self.committed_frames = head["committed_frames"]
        self.committed_bytes = head["committed_bytes"]
        self.views.update(copy.deepcopy(views))

    def finish(self, metadata):
        if self.finished or self.failed or self.stream is None:
            raise RuntimeError("Cannot finish a closed x-nav result store")
        with _atomic_file(self.path) as output:
            output.write(b"{")
            for key, value in metadata.items():
                if key == "frames":
                    continue
                output.write(_encode(key) + b":" + _encode(value) + b",")
            output.write(b'"frames":[')
            for i, entry in enumerate(_entries(self.journal_path, self._descriptor())):
                if i:
                    output.write(b",")
                output.write(_encode(entry["frame"]))
            output.write(b"]}\n")
        self.finished = True
        self.close()
        # A complete results.json is standalone. Cleanup is best effort; failure
        # cannot turn a successfully exported result back into a running journal.
        for path in (self.metadata_path, self.journal_path):
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)

    def fail(self, error):
        if self.finished:
            return
        self.failed = True
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        _atomic_json(
            self.path,
            self._descriptor(status=status, error=f"{type(error).__name__}: {error}"),
        )

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
