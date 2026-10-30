"""Read-only adapter for OMNI_KEYFRAME_MAP V6 snapshots."""

import csv
import re
from collections import Counter, OrderedDict
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import torch

from loaders.base_loader import BaseLoader
from loaders.xnav_camera import NativeView, XNavCamera, make_views
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW, quat_to_rotmat


def is_xnav_map(path):
    root = Path(path).expanduser()
    return (root / "manifest.txt").is_file() or (
        (root / "images/index.csv").is_file() and (root / "poses_base.txt").is_file()
    )


def _numbers(value, count):
    if isinstance(value, str):
        value = value.split(",")
    arr = np.asarray(value, dtype=np.float64)
    if arr.size != count or not np.isfinite(arr).all():
        raise ValueError(f"Expected {count} finite numbers")
    return arr.reshape(-1)


def _rotation(value, normalize=False):
    R = _numbers(value, 9).reshape(3, 3)
    error = np.linalg.norm(R.T @ R - np.eye(3))
    determinant = np.linalg.det(R)
    if error > 1e-3 or determinant <= 0:
        raise ValueError("Invalid calibration rotation")
    # Match camera_calibration_io.cpp's threshold for rounded map parameters.
    if normalize and (error > 1e-5 or abs(determinant - 1) > 1e-5):
        u, _, vh = np.linalg.svd(R)
        R = u @ vh
    return R


def load_calibration(root):
    import yaml

    root = Path(root)
    doc = yaml.safe_load((root / "camera_calibration.yaml").read_text())
    if not isinstance(doc, dict) or doc.get("schema_version") not in (1, 2):
        raise ValueError("Expected camera calibration schema_version 1 or 2")
    if (
        doc.get("image_geometry") != "original_distorted"
        or doc.get("rotation_storage") != "row_major"
        or doc.get("translation_unit") != "metre"
    ):
        raise ValueError("Unsupported calibration geometry/storage/unit")
    if doc["schema_version"] == 1:
        if doc.get("extrinsic_convention") != "p_camera = Rcl * p_lidar + Pcl":
            raise ValueError("Invalid V1 extrinsic convention")
        config = yaml.safe_load((root / "config/map_config.yaml").read_text())
        base = config["input_transform"]
        if not base.get("base_frame") or "enable" not in base:
            raise ValueError("V1 calibration requires map input_transform")
        enabled = str(base["enable"]).lower()
        if enabled not in ("true", "false", "0", "1"):
            raise ValueError("Invalid input_transform enable")
        Rbl, tbl = np.eye(3), np.zeros(3)
        if enabled in ("true", "1"):
            Rbl = _rotation(base["lidar_to_base_r"], normalize=True)
            tbl = _numbers(base["lidar_to_base_t"], 3)
        for c in doc["cameras"]:
            Rbc = Rbl @ np.linalg.inv(_rotation(c.pop("Rcl")))
            tbc = tbl - Rbc @ _numbers(c.pop("Pcl"), 3)
            c["Rbc"], c["Pbc"] = Rbc.reshape(-1).tolist(), tbc.tolist()
        doc.update(
            schema_version=2,
            reference_frame=base["base_frame"],
            extrinsic_convention="p_base = Rbc * p_camera + Pbc",
        )
    if doc.get("extrinsic_convention") != "p_base = Rbc * p_camera + Pbc":
        raise ValueError("Invalid V2 extrinsic convention")
    if doc.get("reference_frame") != "base_link":
        config_path = root / "config/map_config.yaml"
        config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
        # x-nav permits a configured name for its canonical base frame.
        if doc.get("reference_frame") != config.get("input_transform", {}).get(
            "base_frame"
        ):
            raise ValueError(
                "Calibration reference frame does not match map base frame"
            )
    cameras, extrinsics = {}, {}
    if not isinstance(doc.get("cameras"), list) or not doc["cameras"]:
        raise ValueError("Calibration contains no cameras")
    for c in doc["cameras"]:
        camera = XNavCamera(c)
        if (
            camera.camera_id < 0
            or camera.camera_id in cameras
            or "Rcl" in c
            or "Pcl" in c
        ):
            raise ValueError(
                "Invalid/duplicate camera id or mixed extrinsic conventions"
            )
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = _rotation(c["Rbc"]), _numbers(c["Pbc"], 3)
        cameras[camera.camera_id], extrinsics[camera.camera_id] = camera, T
    return doc, cameras, extrinsics


def read_pcd(path):
    """Read x-nav's binary XYZRGB or PointType PCD; honor field layout."""
    path = Path(path)
    header = {}
    with path.open("rb") as f:
        for _ in range(64):
            line = f.readline(4096)
            if not line or len(line) >= 4096:
                raise ValueError(f"Invalid PCD header: {path}")
            words = line.decode("ascii").strip().split()
            if not words or words[0].startswith("#"):
                continue
            if words[0] in header:
                raise ValueError(f"Duplicate PCD header: {path}")
            header[words[0]] = words[1:]
            if words[0] == "DATA":
                break
        else:
            raise ValueError(f"Oversized PCD header: {path}")
        fields = header.get("FIELDS", [])
        expected = [
            "x",
            "y",
            "z",
            "intensity",
            "normal_x",
            "normal_y",
            "normal_z",
            "curvature",
        ]
        if (
            fields not in (["x", "y", "z", "rgb"], expected)
            or header.get("DATA") != ["binary"]
            or header.get("SIZE") != ["4"] * len(fields)
            or header.get("TYPE") != ["F"] * len(fields)
            or header.get("COUNT") != ["1"] * len(fields)
            or header.get("VERSION") not in ([".7"], ["0.7"])
        ):
            raise ValueError(f"Unsupported x-nav binary PCD schema: {path}")
        count = int(header["POINTS"][0])
        width, height = int(header["WIDTH"][0]), int(header["HEIGHT"][0])
        if count < 0 or width < 0 or height < 1 or width * height != count:
            raise ValueError(f"Invalid PCD count: {path}")
        offset = f.tell()
        if path.stat().st_size - offset != count * len(fields) * 4:
            raise ValueError(f"Truncated or oversized PCD payload: {path}")
        data = np.fromfile(f, dtype=np.dtype([(k, "<f4") for k in fields]), count=count)
    xyz = np.column_stack([data[k] for k in ("x", "y", "z")])
    valid = np.isfinite(xyz).all(axis=1)
    colors = None
    if "rgb" in fields:
        bits = data["rgb"].copy().view("<u4")
        colors = np.column_stack(
            [(bits >> 16) & 255, (bits >> 8) & 255, bits & 255]
        ).astype(np.uint8)[valid]
    return xyz[valid], colors


class XNavMapLoader:
    """Iterate keyframes, each containing zero or more Boxer-compatible views.

    start_n is one-based, as in run_boxer; skip_n/max_n count keyframes,
    independently of how many cameras or virtual views each frame contains.
    """

    def __init__(
        self,
        root,
        cameras="all",
        start_n=1,
        skip_n=1,
        max_n=None,
        size=960,
        max_points=10000,
        fov=100.0,
        camera_mode="native",
    ):
        if not isinstance(size, int) or size < 2:
            raise ValueError("Image size must be an integer >= 2")
        if camera_mode not in ("native", "virtual_pinhole"):
            raise ValueError(f"Unsupported x-nav camera mode: {camera_mode}")
        self.camera_mode = camera_mode
        self.root = Path(root).expanduser().resolve()
        self.manifest = {}
        for line in (self.root / "manifest.txt").read_text().splitlines():
            k, v = line.split("=", 1)
            if k in self.manifest:
                raise ValueError(f"Duplicate manifest field {k}")
            self.manifest[k] = v
        m = self.manifest
        if (
            m.get("format") != "OMNI_KEYFRAME_MAP"
            or m.get("schema_version") != "6"
            or m.get("snapshot_complete") != "1"
            or m.get("pose_frame") != "map"
            or m.get("keyframe_cloud_frame") != "base_link"
        ):
            raise ValueError(
                "Expected complete V6 x-nav map in map/base_link coordinates"
            )
        for field in ("map_id", "snapshot_id"):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", m.get(field, "")):
                raise ValueError(f"Invalid {field}")
        self.calibration, self.cameras, self.T_base_camera = load_calibration(self.root)
        self.camera_ids = (
            sorted(self.cameras)
            if cameras == "all"
            else sorted({int(c) for c in cameras.split(",")})
        )
        if not self.camera_ids or not set(self.camera_ids) <= self.cameras.keys():
            raise ValueError(f"Requested cameras not in calibration: {cameras}")
        self.poses, self.pose_times_ns = [], []
        for line in (self.root / "poses_base.txt").read_text().splitlines():
            fields = line.split()
            if len(fields) != 8:
                raise ValueError("Pose rows must be timestamp tx ty tz qx qy qz qw")
            t = int(Decimal(fields[0]) * 1_000_000_000)
            values = np.asarray(fields[1:], dtype=np.float64)
            q = values[3:]
            if not np.isfinite(values).all() or abs(np.linalg.norm(q) - 1) > 1e-3:
                raise ValueError("Invalid pose quaternion/translation")
            q = q / np.linalg.norm(q)
            T = np.eye(4)
            T[:3, :3] = quat_to_rotmat(q[3], q[0], q[1], q[2])
            T[:3, 3] = values[:3]
            self.poses.append(T)
            self.pose_times_ns.append(t)
        self.count = int(m["keyframe_count"])
        if (
            self.count != len(self.poses)
            or self.count < 1
            or np.any(np.diff(self.pose_times_ns) <= 0)
        ):
            raise ValueError("Pose count/timeline does not match manifest")
        self.image_index = {}
        with (self.root / "images/index.csv").open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != [
                "keyframe_id",
                "camera_id",
                "image_stamp_ns",
                "file",
                "source",
                "status",
            ]:
                raise ValueError("Invalid image index header")
            for row in reader:
                key, cid, stamp = (
                    int(row[k]) for k in ["keyframe_id", "camera_id", "image_stamp_ns"]
                )
                if (
                    not 0 <= key < self.count
                    or cid not in self.cameras
                    or stamp < 0
                    or (key, cid) in self.image_index
                ):
                    raise ValueError("Invalid or duplicate image index row")
                if row["status"] == "saved":
                    if (
                        row["file"] != f"{key}_cam{cid}.jpg"
                        or stamp == 0
                        or row["source"] not in ("encoded", "compressed")
                    ):
                        raise ValueError("Invalid saved image metadata")
                    if not (self.root / "images" / row["file"]).is_file():
                        raise ValueError(f"Missing saved image: {row['file']}")
                elif (
                    row["status"] not in ("no_image", "disabled", "legacy")
                    or row["file"]
                    or row["source"] != "none"
                ):
                    raise ValueError("Invalid absent image metadata")
                row.update(keyframe_id=key, camera_id=cid, image_stamp_ns=stamp)
                self.image_index[key, cid] = row
        if set(self.image_index) != {
            (k, c) for k in range(self.count) for c in self.cameras
        }:
            raise ValueError("Image index must describe every keyframe/camera pair")
        if (
            start_n < 1
            or skip_n < 1
            or (max_n is not None and max_n < 1)
            or max_points < 1
        ):
            raise ValueError("Invalid keyframe range or point limit")
        self.keyframe_ids = list(range(self.count))[start_n - 1 :: skip_n]
        if max_n is not None:
            self.keyframe_ids = self.keyframe_ids[:max_n]
        if not self.keyframe_ids:
            raise ValueError("Selected keyframe range is empty")
        self.size, self.max_points, self.fov = size, max_points, fov
        self._views = {}
        self._point_cache = OrderedDict()

    def __len__(self):
        return len(self.keyframe_ids)

    def output_path(self, output_dir):
        return (
            Path(output_dir).expanduser()
            / self.manifest["map_id"]
            / self.manifest["snapshot_id"]
            / self.camera_mode
        )

    def result_file(self, output_dir):
        """Old flat outputs are readable only through the explicit virtual mode."""
        directory = self.output_path(output_dir)
        if self.camera_mode == "virtual_pinhole" and not directory.exists():
            legacy = directory.parent / "results.json"
            if legacy.is_file():
                return legacy
        return directory / "results.json"

    def views(self, camera_id):
        key = (self.camera_mode, camera_id, self.size, self.fov)
        if key not in self._views:
            self._views[key] = (
                [NativeView(self.cameras[camera_id], self.size)]
                if self.camera_mode == "native"
                else make_views(self.cameras[camera_id], self.size, self.fov)
            )
            if not self._views[key]:
                raise ValueError(f"Camera {camera_id} has no valid virtual view")
        return self._views[key]

    def read_image(self, keyframe_id, camera_id):
        row = self.image_index[keyframe_id, camera_id]
        if row["status"] != "saved":
            return None
        img = cv2.imread(str(self.root / "images" / row["file"]), cv2.IMREAD_COLOR)
        c = self.cameras[camera_id]
        if img is None or img.shape != (c.height, c.width, 3):
            raise ValueError(
                f"Corrupt image or calibration size mismatch: {row['file']}"
            )
        return img

    def world_points(self, keyframe_id):
        if keyframe_id in self._point_cache:
            return self._point_cache[keyframe_id]
        result = None
        for folder in ("rgb", "surf"):
            path = self.root / folder / f"{keyframe_id}.pcd"
            if not path.exists():
                continue
            xyz, colors = read_pcd(path)
            T = self.poses[keyframe_id]
            result = ((xyz @ T[:3, :3].T + T[:3, 3]).astype(np.float32), colors)
            if len(xyz):
                break
        if result is None:
            raise ValueError(f"No rgb/surf cloud for keyframe {keyframe_id}")
        self._point_cache[keyframe_id] = result
        while len(self._point_cache) > 2:
            self._point_cache.popitem(last=False)
        return result

    def validate(self):
        counts = Counter()
        # Verify usable geometry before the entry point constructs any model.
        view_counts = {cid: len(self.views(cid)) for cid in self.camera_ids}
        for key in range(self.count):
            for cid in self.cameras:
                row = self.image_index[key, cid]
                counts[row["status"]] += 1
                self.read_image(key, cid)
            # Check the chosen geometry plus any present fallback file.
            self.world_points(key)
            surf = self.root / "surf" / f"{key}.pcd"
            if surf.exists():
                read_pcd(surf)
        return {
            "map_id": self.manifest["map_id"],
            "snapshot_id": self.manifest["snapshot_id"],
            "keyframes": self.count,
            "selected_keyframes": len(self),
            "images": dict(counts),
            "camera_ids": self.camera_ids,
            "camera_mode": self.camera_mode,
            "observations": sum(
                view_counts[cid]
                for key in self.keyframe_ids
                for cid in self.camera_ids
                if self.image_index[key, cid]["status"] == "saved"
            ),
        }

    def load_view(self, keyframe_id, camera_id, view_id, image=None, with_points=True):
        view = next((v for v in self.views(camera_id) if v.view_id == view_id), None)
        if view is None:
            raise ValueError(
                f"Camera {camera_id} has no view {view_id} in {self.camera_mode} mode"
            )
        if image is None:
            image = self.read_image(keyframe_id, camera_id)
        if image is None:
            return None
        T = self.poses[keyframe_id] @ self.T_base_camera[camera_id]
        T = T.copy()
        T[:3, :3] = T[:3, :3] @ view.R_camera_view
        sdp = np.empty((0, 3), dtype=np.float32)
        if with_points:
            world, _ = self.world_points(keyframe_id)
            local = (world - T[:3, 3]) @ T[:3, :3]
            z = local[:, 2]
            uv, valid = view.project(local)
            valid &= ((uv >= 0) & (uv < self.size - 1)).all(axis=1)
            positive = np.flatnonzero(valid)
            uv = uv[positive]
            pix = np.rint(uv).astype(np.int64)
            ok = view.valid_mask[pix[:, 1], pix[:, 0]]
            positive, pix = positive[ok], pix[ok]
            pixels = pix[:, 1] * self.size + pix[:, 0]
            order = np.lexsort((z[positive], pixels))
            _, first = np.unique(pixels[order], return_index=True)
            ids = positive[order[first]]
            if len(ids) > self.max_points:
                ids = ids[np.linspace(0, len(ids) - 1, self.max_points, dtype=int)]
            sdp = world[ids]
        rectified = view.remap(image)
        rgb = cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB)
        return {
            "img0": BaseLoader.img_to_tensor(rgb),
            "cam0": view.cam.clone(),
            "T_world_rig0": PoseTW.from_matrix(torch.from_numpy(T).float()),
            "sdp_w": torch.from_numpy(sdp.copy()),
            "time_ns0": self.image_index[keyframe_id, camera_id]["image_stamp_ns"],
            "bb2d0": torch.empty((0, 4)),
            "obbs": ObbTW(torch.empty((0, 165))),
            "keyframe_id": keyframe_id,
            "camera_id": camera_id,
            "view_id": view_id,
            "rectification": view,
            "view_geometry": view,
            "valid_mask": view.valid_mask,
            "camera_mode": self.camera_mode,
        }

    def iter_keyframes(self, with_points=True):
        for key in self.keyframe_ids:
            views = []
            for cid in self.camera_ids:
                image = self.read_image(key, cid)
                if image is not None:
                    for view in self.views(cid):
                        views.append(
                            self.load_view(key, cid, view.view_id, image, with_points)
                        )
            yield {
                "keyframe_id": key,
                "time_ns": self.pose_times_ns[key],
                "T_map_base": self.poses[key],
                "views": views,
                "sources": [
                    self.image_index[key, cid].copy() for cid in self.camera_ids
                ],
            }
