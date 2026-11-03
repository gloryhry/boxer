"""Publish completed server CSVs alongside the source x-nav map."""

import os
import shutil
import tempfile
from pathlib import Path

CSV_FILES = (
    "boxer_3dbbs_tracked.csv",
    "boxer_3dbbs_tracks.csv",
    "boxer_3dbbs.csv",
    "owl_2dbbs.csv",
)


def export_map_csv(source, map_root):
    source = Path(source)
    destination = Path(map_root) / "3dbbox"
    for name in CSV_FILES:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Missing inference CSV: {source / name}")
        if (destination / name).is_dir():
            raise IsADirectoryError(str(destination / name))
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".boxer-", dir=destination) as temporary:
        staging = Path(temporary)
        for name in CSV_FILES:
            shutil.copyfile(source / name, staging / name)
            with (staging / name).open("rb") as stream:
                os.fsync(stream.fileno())
        for name in CSV_FILES:
            os.replace(staging / name, destination / name)
