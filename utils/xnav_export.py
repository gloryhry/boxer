"""Publish inference CSVs and their effective vocabulary inside the x-nav map."""

import csv
import io
import os
import shutil
import tempfile
from pathlib import Path

from utils.xnav_result_store import _atomic_file

MAP_VOCABULARY_PATH = "3dbbox/labels.csv"

CSV_FILES = (
    "boxer_3dbbs_tracked.csv",
    "boxer_3dbbs_tracks.csv",
    "boxer_3dbbs.csv",
    "owl_2dbbs.csv",
)


def save_map_vocabulary(map_root, labels):
    """Publish the full ordered label list, including labels with no detections."""
    if not labels or any(
        not isinstance(label, str) or not label.strip() for label in labels
    ):
        raise ValueError("Map vocabulary must contain nonempty text labels")
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(("sem_id", "label"))
    writer.writerows(enumerate(labels))
    destination = Path(map_root) / MAP_VOCABULARY_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _atomic_file(destination) as stream:
        stream.write(text.getvalue().encode("utf-8"))


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
