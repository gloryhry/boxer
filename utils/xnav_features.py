"""Original-image descriptors and a durable, map-wide CSV keyed by image/camera."""

import csv
import fcntl
import io
import json
import operator
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from loaders.xnav_loader import XNavMapLoader, frame_identity
from utils.tw.pose import rotmat_to_quat
from utils.tw.xnav_camera import PREPROCESS_VERSION
from utils.xnav_result_store import _atomic_file, _sync_directory

FEATURE_CSV_PATH = "3dbbox/dino_features.csv"
FEATURE_VECTOR_FIELDS = ("cls_token", "patch_mean_pooling")
FEATURE_POSE_FIELDS = (
    "base_link_x",
    "base_link_y",
    "base_link_z",
    "base_link_qx",
    "base_link_qy",
    "base_link_qz",
    "base_link_qw",
)
_LEGACY_CSV_FIELDS = ("相机帧序号", "相机序号", *FEATURE_VECTOR_FIELDS)
FEATURE_CSV_FIELDS = (*_LEGACY_CSV_FIELDS, *FEATURE_POSE_FIELDS)


def _csv_line(fields):
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerow(fields)
    return stream.getvalue().encode("utf-8")


def _index(value):
    try:
        index = int(value) if isinstance(value, str) else operator.index(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid DINOv3 frame/camera index: {value!r}") from error
    if isinstance(value, bool) or index < 0:
        raise ValueError(f"Invalid DINOv3 frame/camera index: {value!r}")
    return index


def _vector_json(value, name, *, shape):
    vector = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if tuple(vector.shape) != shape or not torch.isfinite(vector).all():
        raise ValueError(f"Invalid DINOv3 {name}: expected finite {list(shape)} vector")
    return json.dumps(
        vector.reshape(-1).tolist(), allow_nan=False, separators=(",", ":")
    )


def _pose_values(value):
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Invalid DINOv3 base_link pose: expected seven numbers"
        ) from error
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Invalid DINOv3 base_link pose: expected seven finite numbers")
    norm = np.linalg.norm(pose[3:])
    if not np.isfinite(norm) or abs(norm - 1) > 1e-3:
        raise ValueError("Invalid DINOv3 base_link pose: expected unit quaternion")
    return [*pose[:3].tolist(), *(pose[3:] / norm).tolist()]


def _base_link_pose(matrix):
    """Convert T_map_base in float64, without camera or virtual-view extrinsics."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-9)
    ):
        raise ValueError("Invalid DINOv3 T_map_base: expected finite 4x4 rigid pose")
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6
    ) or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-6):
        raise ValueError("Invalid DINOv3 T_map_base rotation")
    qw, qx, qy, qz = rotmat_to_quat(rotation)
    return _pose_values([*matrix[:3, 3], qx, qy, qz, qw])


def cleanup_legacy_features(map_root):
    """Delete only legacy PT entries directly inside the map's feature directory."""
    folder = Path(map_root) / "3dbbox/dino_features"
    if folder.is_symlink():
        raise ValueError(f"Legacy DINOv3 feature directory is a symlink: {folder}")
    if not folder.is_dir():
        return 0
    removed = 0
    for path in folder.glob("*.pt"):
        if path.is_file() or path.is_symlink():
            path.unlink()
            removed += 1
    _sync_directory(folder)
    if not any(folder.iterdir()):
        folder.rmdir()
        _sync_directory(folder.parent)
    return removed


def image_feature_payload(
    frame, source, native, patches, cls_token, model_name, checkpoint
):
    """Reduce on the model device; only small CPU arrays cross the output queue."""
    path = FEATURE_CSV_PATH
    view = native["view_geometry"]
    patch_mean = patches.detach().float().mean(dim=(-2, -1))
    data = {
        "cls_token": cls_token.detach().float().cpu().numpy().copy(),
        "patch_mean_pooling": patch_mean.cpu().numpy().copy(),
        "source_image": f"images/{source['file']}",
        **frame_identity(frame),
        "camera_id": source["camera_id"],
        "image_stamp_ns": source["image_stamp_ns"],
        **dict(zip(FEATURE_POSE_FIELDS, _base_link_pose(frame["T_map_base"]))),
        "model_name": model_name,
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "preprocessing": {
            "camera_mode": "native",
            "version": PREPROCESS_VERSION,
            "input_hw": list(native["img0"].shape[-2:]),
            "resized_size": list(view.resized_size),
            "padding": list(view.padding),
            "pooling": "mean_all_patches",
            "l2_normalized": False,
        },
    }
    return {"path": path, "data": data}


class DinoFeatureCsvStore:
    """Merge one frame atomically, retaining unrelated rows and concurrent updates."""

    def __init__(self, map_root):
        self.map_root = Path(map_root)
        self.path = self.map_root / FEATURE_CSV_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_name(".dino_features.lock")
        self._thread_lock = threading.Lock()
        self._rows = {}
        self._version = None
        with self._locked():
            self._refresh()
            if self._version is None:
                self._publish(self._rows)
            cleanup_legacy_features(self.map_root)

    @contextmanager
    def _locked(self):
        with self._thread_lock, self.lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _fingerprint(self):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return stat.st_ino, stat.st_mtime_ns, stat.st_size

    def _refresh(self):
        version = self._fingerprint()
        if version == self._version:
            return
        rows = {}
        legacy = False
        if version is not None:
            with self.path.open(encoding="utf-8", newline="") as stream:
                reader = csv.reader(stream, strict=True)
                header = next(reader, None)
                legacy = header is None or header == list(_LEGACY_CSV_FIELDS)
                if not legacy and header != list(FEATURE_CSV_FIELDS):
                    raise ValueError(f"Invalid DINOv3 CSV header: {self.path}")
                loader = None
                for fields in reader:
                    if len(fields) != len(header):
                        raise ValueError(f"Invalid DINOv3 CSV row: {self.path}")
                    key = _index(fields[0]), _index(fields[1])
                    if key in rows:
                        raise ValueError(f"Duplicate DINOv3 CSV image/camera: {key}")
                    vectors = [
                        _vector_json(json.loads(value), name, shape=(384,))
                        for name, value in zip(FEATURE_VECTOR_FIELDS, fields[2:4])
                    ]
                    if legacy:
                        if loader is None:
                            # Migration uses all source frames and cameras.
                            loader = XNavMapLoader(self.map_root)
                        source = loader.image_index.get(key)
                        if source is None or source["status"] != "saved":
                            raise ValueError(
                                f"Cannot upgrade DINOv3 CSV image/camera: {key}"
                            )
                        pose = _base_link_pose(loader.poses[key[0]])
                    else:
                        pose = _pose_values(fields[4:])
                    rows[key] = _csv_line([*key, *vectors, *pose])
        if legacy:
            # Publish only after every legacy row has been validated and enriched.
            self._publish(rows)
        else:
            self._rows, self._version = rows, version

    def _publish(self, rows):
        with _atomic_file(self.path) as stream:
            stream.write(_csv_line(FEATURE_CSV_FIELDS))
            stream.writelines(rows[key] for key in sorted(rows))
        self._rows, self._version = rows, self._fingerprint()

    def upsert(self, payloads):
        """Sync the CSV before the caller commits the associated result frame."""
        updates = {}
        for payload in payloads:
            data = payload["data"]
            frame_id = data.get("image_frame_id")
            if frame_id is None:
                frame_id = data["frame_id"]
            key = _index(frame_id), _index(data["camera_id"])
            vectors = [
                _vector_json(data[name], name, shape=(1, 384))
                for name in FEATURE_VECTOR_FIELDS
            ]
            pose = _pose_values([data[name] for name in FEATURE_POSE_FIELDS])
            updates[key] = _csv_line([*key, *vectors, *pose])
        if updates:
            with self._locked():
                self._refresh()
                self._publish({**self._rows, **updates})
