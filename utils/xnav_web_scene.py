"""CPU-only scene and image preparation; no OpenGL or model ownership."""

import threading
from collections import OrderedDict

import cv2
import numpy as np

from loaders.xnav_loader import XNavMapLoader, processing_frame_id, read_pcd
from utils.image import draw_bb3s, render_bb2
from utils.xnav_pipeline import records_to_obbs, source_signature


def cloud_bytes(points, colors=None, limit=500000):
    points = np.asarray(points).reshape(-1, 3)
    colors = np.broadcast_to(
        (0.5, 0.55, 0.6) if colors is None else colors, points.shape
    )
    valid = np.isfinite(points).all(axis=1)
    points, colors = points[valid], colors[valid]
    if len(points) > limit:
        ids = np.linspace(0, len(points) - 1, limit, dtype=int)
        points, colors = points[ids], colors[ids]
    data = np.column_stack([points, colors]).astype("<f4")
    return data.tobytes(), points


def render_observation(loader, frame, observation, tracked=True, tentative=False):
    """Return the same two BGR panels used by desktop live_viz."""
    datum = loader.load_view(
        processing_frame_id(frame),
        observation["camera_id"],
        observation["view_id"],
        with_points=False,
    )
    if datum is None:
        raise ValueError("Image observation is no longer available")
    image = (datum["img0"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    boxes2 = np.asarray(observation["boxes_2d"], dtype=np.float32).reshape(-1, 4)
    top = render_bb2(image.copy(), boxes2, texts=observation["labels_2d"])
    boxes = records_to_obbs(
        frame["tracks" if tracked else "detections"],
        confirmed_only=tracked and not tentative,
    )
    names = [
        f"#{int(o.inst_id.item())} {name}" if tracked else name
        for o, name in zip(boxes, boxes.text_string())
    ]
    bottom = draw_bb3s(
        image.copy(),
        datum["T_world_rig0"],
        datum["cam0"],
        boxes,
        texts=names,
    )
    return top, bottom


class MapScene:
    def __init__(self, root):
        self.loader = XNavMapLoader(root)
        self.validation = self.loader.validate()
        self.signature = source_signature(self.loader)
        self.lock = threading.RLock()
        self.cloud = None
        self.metadata = None
        self.loaders = OrderedDict()
        self.assets = OrderedDict()
        self.asset_bytes = 0

    def check_source(self):
        if source_signature(self.loader) != self.signature:
            raise ValueError("Map changed after loading; load the map again")

    def scene(self):
        with self.lock:
            if self.metadata is None:
                path = self.loader.root / "global_map.pcd"
                if path.exists():
                    points, _ = read_pcd(path)
                else:
                    points, _ = self.loader.world_points(self.loader.frame_ids[0])
                self.cloud, points = cloud_bytes(points)
                center = np.median(points, axis=0) if len(points) else np.zeros(3)
                radius = (
                    max(5.0, float(np.linalg.norm(points - center, axis=1).max()))
                    if len(points)
                    else 5.0
                )
                self.metadata = {
                    "center": center.tolist(),
                    "radius": radius,
                    "point_count": len(points),
                    "format": "float32-le xyzrgb",
                    "up_axis": "+Z",
                    "unit": "metre",
                }
            return self.metadata

    def result_loader(self, parameters):
        with self.lock:
            return self._result_loader(parameters)

    def _result_loader(self, parameters):
        key = (
            parameters["camera_mode"],
            parameters["image_size"],
            parameters["fov_degrees"],
        )
        if key not in self.loaders:
            loader = XNavMapLoader(
                self.loader.root, camera_mode=key[0], size=key[1], fov=key[2]
            )
            self.loaders[key] = loader
            if len(self.loaders) > 4:
                self.loaders.popitem(last=False)
        return self.loaders[key]

    def cached(self, key, make):
        """Bound binary/image cache to 32 MiB, regardless of replay length."""
        with self.lock:
            if key in self.assets:
                self.assets.move_to_end(key)
                return self.assets[key]
            data = make()
            self.assets[key] = data
            self.asset_bytes += len(data)
            while self.asset_bytes > 32 * 1024 * 1024 and self.assets:
                _, old = self.assets.popitem(last=False)
                self.asset_bytes -= len(old)
            return data
