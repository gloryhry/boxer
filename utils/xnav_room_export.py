"""Publish one workflow's room bundle into a map, with recoverable replacement."""

import json
import os
import re
import shutil
from contextlib import nullcontext
from pathlib import Path

from utils.xnav_archive import BUFFER_SIZE, _check_cancelled, create_map_archive
from utils.xnav_result_store import _atomic_json, _sync_directory

ROOM_DIRECTORY = "room_segmentation"
ROOM_RELATIVE_PATH = f"3dbbox/{ROOM_DIRECTORY}"


def _paths(map_root, identity):
    if re.fullmatch(r"[a-f0-9]{32}", identity) is None:
        raise ValueError("Invalid room export identity")
    parent = Path(map_root).resolve() / "3dbbox"
    if parent.is_symlink():
        raise ValueError("Map 3dbbox directory cannot be a symlink")
    current = parent / ROOM_DIRECTORY
    staging = parent / f".boxer-rooms-{identity}.staging"
    backup = parent / f".boxer-rooms-{identity}.backup"
    for path in (current, staging, backup):
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError(f"Invalid room export directory: {path}")
    return parent, current, staging, backup


def _owner(directory):
    try:
        return json.loads((directory / "source.json").read_bytes())["job_id"]
    except (OSError, ValueError, KeyError):
        return None


def recover_room_export(journal, map_root, identity):
    """Restore a displaced old bundle or retain an already published new bundle."""
    journal = Path(journal)
    if not journal.is_file():
        return
    if journal.is_symlink():
        raise ValueError("Room export journal cannot be a symlink")
    record = json.loads(journal.read_bytes())
    if (
        record["job_id"] != identity
        or Path(record["map_root"]).resolve() != Path(map_root).resolve()
    ):
        raise ValueError("Room export journal references another map or workflow")
    parent, current, staging, backup = _paths(map_root, identity)
    if backup.exists():
        if not current.exists():
            backup.rename(current)
            _sync_directory(parent)
        elif _owner(current) is not None:
            # The new bundle (or a later workflow's bundle) is already published.
            shutil.rmtree(backup)
        else:
            raise ValueError(
                "Cannot replace an unexpected room directory during recovery"
            )
    if staging.exists():
        shutil.rmtree(staging)
    if parent.is_dir():
        _sync_directory(parent)


def _copy(source, destination, cancel):
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Invalid room artifact: {source}")
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        while True:
            _check_cancelled(cancel)
            data = incoming.read(BUFFER_SIZE)
            if not data:
                break
            outgoing.write(data)
        outgoing.flush()
        os.fsync(outgoing.fileno())


def export_room_results(
    map_root,
    sources,
    provenance,
    journal,
    cancel,
    check_source,
    *,
    archive=None,
    publish_lock=None,
):
    """Stage raw artifacts, replace the whole room directory, then archive once."""
    identity = provenance["job_id"]
    parent, current, staging, backup = _paths(map_root, identity)
    archive = Path(archive) if archive is not None else None
    if archive is not None and (archive.exists() or archive.is_symlink()):
        raise FileExistsError(f"Workflow archive already exists: {archive}")
    check_source()
    _check_cancelled(cancel)
    parent.mkdir(exist_ok=True)
    state = dict(
        job_id=identity, map_root=str(Path(map_root).resolve()), phase="staging"
    )
    _atomic_json(Path(journal), state)
    try:
        staging.mkdir()
        for name, source in sources.items():
            if name not in {
                "room_polygons.json",
                "room-bindings.json",
            }:
                raise ValueError(f"Unsupported room artifact: {name}")
            _copy(source, staging / name, cancel)
        _atomic_json(
            staging / "source.json",
            dict(
                provenance,
                files=sorted([*sources, "source.json"]),
                schema_version=2,
            ),
        )
        _sync_directory(staging)
        check_source()
        with publish_lock if publish_lock is not None else nullcontext():
            _check_cancelled(cancel)
            state["phase"] = "replacing"
            _atomic_json(Path(journal), state)
            if current.exists():
                current.rename(backup)
                _sync_directory(parent)
            staging.rename(current)
            _sync_directory(parent)
            state["phase"] = "published"
            _atomic_json(Path(journal), state)
        if backup.exists():
            shutil.rmtree(backup)
            _sync_directory(parent)
        _check_cancelled(cancel)
        check_source()
        if archive is not None:
            create_map_archive(map_root, archive, cancel, publish_lock=publish_lock)
            check_source()
        with publish_lock if publish_lock is not None else nullcontext():
            _check_cancelled(cancel)
            state["phase"] = "complete"
            _atomic_json(Path(journal), state)
        return ROOM_RELATIVE_PATH
    except BaseException:
        if archive is not None:
            archive.unlink(missing_ok=True)
        recover_room_export(journal, map_root, identity)
        raise
