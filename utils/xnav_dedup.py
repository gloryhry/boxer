"""Conservative static-map identity checks shared by association and merging.

Geometry proposes nearby pairs. Detector evidence can veto a merge, or recover
thin objects whose 3D overlap is unreliable. Image groups, rather than stereo
views, are the unit of evidence. No point clouds or inference models are owned
here; saved detector observations suffice for cache-only replay.
"""

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch

from utils.tw.obb import iou_mc7


def rectangle_iou(a, b):
    """IoU of one x1,x2,y1,y2 rectangle against an array of rectangles."""
    b = np.asarray(b).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(a[1], b[:, 1]) - np.maximum(a[0], b[:, 0]))
    intersection *= np.maximum(0, np.minimum(a[3], b[:, 3]) - np.maximum(a[2], b[:, 2]))
    area = max(0, a[1] - a[0]) * max(0, a[3] - a[2])
    other = np.maximum(0, b[:, 1] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 2])
    return intersection / np.maximum(area + other - intersection, 1e-8)


@dataclass
class _EvidenceView:
    frame_id: int
    cam: object
    pose: object
    position: np.ndarray
    boxes: np.ndarray
    labels: tuple[str, ...]
    world_to_camera: np.ndarray


class ConservativeDeduplicator:
    """Pair compatibility with bounded, geometry-keyed projection caches."""

    max_distance = 0.30
    distance_scale = 0.35
    max_size_ratio = 1.5
    semantic_threshold = 0.7
    merge_iou = 0.30
    projection_iou = 0.70
    detection_iou = 0.50
    separate_iou = 0.20
    min_baseline = 0.30
    cache_size = 512

    def __init__(self, semantic_similarity, device="cpu"):
        self.semantic_similarity = semantic_similarity
        self.device = device
        self.views = []
        self._view_keys = set()
        self._projections = OrderedDict()
        self._evidence_cache = OrderedDict()
        self._label_indices = {}

    @classmethod
    def metadata(cls):
        return {
            "mode": "conservative",
            "version": 1,
            "merge_iou_threshold": cls.merge_iou,
            "semantic_threshold": cls.semantic_threshold,
            "max_center_distance": cls.max_distance,
            "center_distance_scale": cls.distance_scale,
            "max_size_ratio": cls.max_size_ratio,
            "projection_iou_threshold": cls.projection_iou,
            "detection_iou_threshold": cls.detection_iou,
            "separate_detection_iou": cls.separate_iou,
            "min_image_groups": 2,
            "min_online_track_support": 2,
            "min_baseline": cls.min_baseline,
        }

    def reset(self):
        self.views.clear()
        self._view_keys.clear()
        self._projections.clear()
        self._evidence_cache.clear()
        self._label_indices.clear()

    def remember(self, observations, frame_idx):
        """Copy small immutable detector/camera records, never dense map points."""
        for observation in observations:
            if observation.boxes_2d is None or observation.labels_2d is None:
                continue
            boxes = torch.as_tensor(observation.boxes_2d).detach().float().cpu()
            boxes = boxes.numpy().reshape(-1, 4).copy()
            labels = tuple(s.lower() for s in observation.labels_2d)
            if len(boxes) != len(labels):
                raise ValueError("ViewObservation boxes_2d/labels_2d lengths differ")
            valid = np.isfinite(boxes).all(axis=1)
            valid &= (boxes[:, 1] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 2])
            boxes, labels = boxes[valid], tuple(s for s, ok in zip(labels, valid) if ok)
            if not len(boxes):
                continue
            cam = observation.cam.detach().float().cpu().reshape(-1).clone()
            pose = observation.T_world_rig.detach().float().cpu().reshape(-1).clone()
            frame_id = (
                frame_idx if observation.frame_id is None else observation.frame_id
            )
            key = (frame_id, cam._data.numpy().tobytes(), pose._data.numpy().tobytes())
            if key in self._view_keys:
                continue
            self._view_keys.add(key)
            self.views.append(
                _EvidenceView(
                    frame_id,
                    cam,
                    pose,
                    pose.t.numpy().copy(),
                    boxes,
                    labels,
                    (pose @ cam.T_camera_rig.inverse()).inverse().matrix.numpy()[:3],
                )
            )

    @staticmethod
    def _key(box):
        return (
            torch.cat(
                (box.bb3_object.reshape(-1), box.T_world_object._data.reshape(-1))
            )
            .detach()
            .float()
            .cpu()
            .numpy()
            .tobytes()
        )

    @staticmethod
    def _label(box):
        return box.reshape(1, -1).text_string()[0].lower()

    def geometry_compatible(self, a, b):
        sizes = torch.stack((a.bb3_diagonal.reshape(3), b.bb3_diagonal.reshape(3)))
        sizes = sizes.detach().float().cpu().numpy()
        centers = (
            torch.stack((a.T_world_object.t.reshape(3), b.T_world_object.t.reshape(3)))
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        if not np.isfinite(sizes).all() or not np.isfinite(centers).all():
            return False
        if (sizes <= 0).any():
            return False
        radius = min(
            self.max_distance,
            self.distance_scale * np.linalg.norm(sizes[:, :2], axis=1).min(),
        )
        if np.linalg.norm(centers[0] - centers[1]) > radius + 1e-6:
            return False
        # Swapping horizontal dimensions is the same rectangular box at yaw+90.
        # Sorting also avoids choosing an unstable yaw for near-square objects.
        aligned = np.column_stack((np.sort(sizes[:, :2], axis=1), sizes[:, 2]))
        if (
            aligned.max(axis=0) / aligned.min(axis=0) > self.max_size_ratio + 1e-6
        ).any():
            return False
        return (
            self.semantic_similarity(self._label(a), self._label(b))
            >= self.semantic_threshold
        )

    @classmethod
    def candidate_pairs(cls, boxes, changed=None):
        """3D grid neighbours within the maximum admissible center distance."""
        cells = defaultdict(list)
        centers = [b.T_world_object.t.reshape(3).detach().cpu().numpy() for b in boxes]
        for i, center in enumerate(centers):
            if np.isfinite(center).all():
                cells[tuple(np.floor(center / cls.max_distance).astype(int))].append(i)
        pairs = []
        for cell, indices in cells.items():
            for offset in product((-1, 0, 1), repeat=3):
                neighbours = cells.get(tuple(c + d for c, d in zip(cell, offset)), ())
                for i in indices:
                    for j in neighbours:
                        if i >= j or (
                            changed is not None
                            and i not in changed
                            and j not in changed
                        ):
                            continue
                        if (
                            np.linalg.norm(centers[i] - centers[j])
                            <= cls.max_distance + 1e-6
                        ):
                            pairs.append((i, j))
        return sorted(pairs)

    def _project(self, box):
        """Batch all new views; an unchanged box only projects appended history."""
        if not self.views:
            # Optional detector metadata may be absent on non-map callers;
            # there is no evidence to cache in that case.
            return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=bool)
        label = self._label(box)
        key = self._key(box), label
        previous = self._projections.pop(key, None)
        count = 0 if previous is None else len(previous[0])
        if count == len(self.views):
            if previous is None:
                previous = np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=bool)
            self._projections[key] = previous
            return previous
        rectangles = np.zeros((len(self.views), 4), dtype=np.float32)
        valid = np.zeros(len(self.views), dtype=bool)
        if previous is not None:
            rectangles[:count], valid[:count] = previous
        groups = defaultdict(list)
        compatible = self._compatible_indices(label)
        center = box.bb3_center_world.reshape(3).detach().float().cpu().numpy()
        radius = float(box.bb3_diagonal.detach().float().cpu().norm()) / 2
        for i in range(count, len(self.views)):
            # Exact exclusions: no compatible detector label, or the entire
            # bounding sphere lies behind the camera's front hemisphere.
            view = self.views[i]
            depth = view.world_to_camera[2, :3] @ center + view.world_to_camera[2, 3]
            if not len(compatible[i]) or depth + radius < -1e-5:
                continue
            groups[self.views[i].cam.shape[-1]].append(i)
        for indices in groups.values():
            cameras = torch.stack([self.views[i].cam for i in indices]).to(self.device)
            poses = torch.stack([self.views[i].pose for i in indices]).to(self.device)
            projected, ok = (
                box.detach()
                .float()
                .to(self.device)
                .reshape(1, -1)
                .get_pseudo_bb2(
                    cameras, poses, num_samples_per_edge=32, valid_ratio=0.16667
                )
            )
            rectangles[indices] = projected.reshape(-1, 4).cpu().numpy()
            valid[indices] = ok.reshape(-1).cpu().numpy()
        result = rectangles, valid
        self._projections[key] = result
        if len(self._projections) > self.cache_size:
            self._projections.popitem(last=False)
        return result

    def _compatible_indices(self, label):
        indices = self._label_indices.setdefault(label, [])
        for view in self.views[len(indices) :]:
            indices.append(
                np.array(
                    [
                        i
                        for i, name in enumerate(view.labels)
                        if self.semantic_similarity(label, name)
                        >= self.semantic_threshold
                    ],
                    dtype=int,
                )
            )
        return indices

    def _evidence(self, a, b):
        """Return (veto, projection score), including all available negative views."""
        label_a, label_b = self._label(a), self._label(b)
        identity = tuple(sorted(((self._key(a), label_a), (self._key(b), label_b))))
        key = identity, len(self.views)
        if key in self._evidence_cache:
            self._evidence_cache.move_to_end(key)
            return self._evidence_cache[key]
        pa, va = self._project(a)
        pb, vb = self._project(b)
        ia, ib = self._compatible_indices(label_a), self._compatible_indices(label_b)
        separate_frames, positive = set(), []
        for i in np.flatnonzero(va & vb):
            view = self.views[i]
            if not len(ia[i]) or not len(ib[i]):
                continue
            ma, mb = rectangle_iou(pa[i], view.boxes), rectangle_iou(pb[i], view.boxes)
            aa, bb = (
                ia[i][ma[ia[i]] >= self.detection_iou],
                ib[i][mb[ib[i]] >= self.detection_iou],
            )
            if not len(aa) or not len(bb):
                continue
            first, second = aa[np.argmax(ma[aa])], bb[np.argmax(mb[bb])]
            if (
                first != second
                and rectangle_iou(view.boxes[first], view.boxes[second : second + 1])[0]
                < self.separate_iou
            ):
                separate_frames.add(view.frame_id)
            if len(separate_frames) >= 2:
                result = True, 0.0
                break
            overlap = float(rectangle_iou(pa[i], pb[i : i + 1])[0])
            if overlap >= self.projection_iou and np.intersect1d(aa, bb).size:
                positive.append((view.frame_id, view.position, overlap))
        else:
            baseline = False
            for i, (frame_id, position, _) in enumerate(positive):
                if any(
                    frame_id != other_id
                    and np.linalg.norm(position - other_pos) >= self.min_baseline - 1e-6
                    for other_id, other_pos, _ in positive[i + 1 :]
                ):
                    baseline = True
                    break
            result = (
                False,
                float(np.median([p[2] for p in positive])) if baseline else 0.0,
            )
        self._evidence_cache[key] = result
        if len(self._evidence_cache) > self.cache_size:
            self._evidence_cache.popitem(last=False)
        return result

    def score(self, a, b, iou_threshold=None, *, overlap=None, allow_projection=True):
        """None means incompatible; a finite score ranks admissible pairs."""
        if not self.geometry_compatible(a, b):
            return None
        veto, projection_score = self._evidence(a, b)
        if veto:
            return None
        if overlap is None:
            overlap = float(
                iou_mc7(
                    a.detach().float().cpu().reshape(1, -1),
                    b.detach().float().cpu().reshape(1, -1),
                    samp_per_dim=8,
                ).item()
            )
        threshold = self.merge_iou if iou_threshold is None else iou_threshold
        if np.isfinite(overlap) and overlap >= threshold:
            return float(overlap)
        if allow_projection and projection_score:
            return projection_score
        return None

    def conflicts(self, a, b):
        """Independent-object evidence also protects otherwise valid IoU matches."""
        return self._evidence(a, b)[0]

    def redundant_observation_score(self, a, b):
        """Only multi-group detector corroboration can absorb a same-frame extra."""
        if not self.geometry_compatible(a, b):
            return None
        veto, score = self._evidence(a, b)
        return score if score and not veto else None
