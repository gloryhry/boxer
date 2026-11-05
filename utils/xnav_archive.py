"""Stream a map's completed 3dbbox directory into an atomic job archive."""

import os
import re
import stat
import zipfile
from concurrent.futures import CancelledError
from pathlib import Path

from utils.xnav_result_store import _atomic_file

BUFFER_SIZE = 1024 * 1024
ATOMIC_TEMP = re.compile(r"^\..+\.[a-f0-9]{32}\.tmp$")


def _check_cancelled(cancel):
    if cancel.is_set():
        raise CancelledError("Job archive cancelled")


def _internal(name):
    return (
        name == ".dino_features.lock"
        or name.startswith(".boxer-")
        or ATOMIC_TEMP.fullmatch(name) is not None
    )


def _add_directory(package, directory, member, cancel):
    _check_cancelled(cancel)
    package.writestr(f"{member}/", b"")
    with os.scandir(directory) as entries:
        for entry in sorted(entries, key=lambda entry: entry.name):
            _check_cancelled(cancel)
            if _internal(entry.name):
                continue
            name = f"{member}/{entry.name}"
            mode = entry.stat(follow_symlinks=False).st_mode
            if stat.S_ISDIR(mode):
                child = os.open(
                    entry.name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory,
                )
                try:
                    _add_directory(package, child, name, cancel)
                finally:
                    os.close(child)
            elif stat.S_ISREG(mode):
                # Open relative to the inspected directory and recheck the inode.
                # NONBLOCK also prevents a replaced FIFO from hanging finalization.
                descriptor = os.open(
                    entry.name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=directory,
                )
                with os.fdopen(descriptor, "rb") as source:
                    if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                        raise ValueError(f"Not a regular archive file: {name}")
                    with package.open(name, "w", force_zip64=True) as target:
                        while True:
                            _check_cancelled(cancel)
                            data = source.read(BUFFER_SIZE)
                            if not data:
                                break
                            target.write(data)
            else:
                raise ValueError(f"Symlink or special archive entry: {name}")


def create_map_archive(map_root, destination, cancel):
    """Snapshot business files, without following links or publishing partial ZIPs."""
    _check_cancelled(cancel)
    directory = os.open(
        Path(map_root) / "3dbbox", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        with _atomic_file(Path(destination)) as output:
            with zipfile.ZipFile(
                output,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
                allowZip64=True,
            ) as package:
                _add_directory(package, directory, "3dbbox", cancel)
            _check_cancelled(cancel)
    finally:
        os.close(directory)
