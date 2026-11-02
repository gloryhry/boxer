"""Float32 CUDA visibility with the CPU track objects as canonical state."""

import contextlib

import numpy as np
import torch

from utils.tw.xnav_camera import fixed_camera_model


def mask_visible_boxes_cpu(bounds, visible, mask, cache):
    """The original CPU rectangle clipping and integral-mask predicate."""
    mask = mask.bool().cpu()
    key = (mask.data_ptr(), tuple(mask.shape), tuple(mask.stride()))
    pixels = mask.numpy()
    cached = cache.get(key)
    # Tensor wrappers around the same NumPy view are recreated each frame.
    # Compare contents as well, so mutable external masks cannot go stale.
    if cached is None or not np.array_equal(cached[0], pixels):
        snapshot = pixels.copy()
        integral = np.pad(
            snapshot.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0))
        )
        if len(cache) >= 16:
            cache.pop(next(iter(cache)))
        cache[key] = snapshot, integral
    else:
        integral = cached[1]
    indices = visible.nonzero(as_tuple=True)[0]
    if not len(indices):
        return visible
    rectangles = bounds.reshape(-1, 4).cpu().numpy()[indices.numpy()]
    if not np.isfinite(rectangles).all():
        # Retain Python int()'s exception for non-finite visible bounds.
        for value in rectangles.flat:
            int(value)
    # astype truncates towards zero just like Python int(), including -0.x.
    rectangles = rectangles.astype(np.int64)
    xa = np.maximum(0, rectangles[:, 0])
    xb = np.minimum(pixels.shape[1], rectangles[:, 1] + 1)
    ya = np.maximum(0, rectangles[:, 2])
    yb = np.minimum(pixels.shape[0], rectangles[:, 3] + 1)
    nonempty = (xb > xa) & (yb > ya)
    xa, xb, ya, yb = (a[nonempty] for a in (xa, xb, ya, yb))
    hits = np.zeros(len(indices), dtype=bool)
    hits[nonempty] = (
        integral[yb, xb] - integral[ya, xb] - integral[yb, xa] + integral[ya, xa]
    ) > 0
    visible[indices] = torch.from_numpy(hits)
    return visible


def visibility_cpu(obbs, observations, min_obs_points, cache):
    """Canonical CPU predicates, shared by serial execution and boundary repair."""
    count = len(obbs)
    projected = torch.zeros(count, dtype=torch.bool)
    supported = torch.zeros(count, dtype=torch.bool)
    if not count:
        return projected, supported
    for view in observations:
        bb2, ok = obbs.get_pseudo_bb2(
            view.cam.unsqueeze(0),
            view.T_world_rig.unsqueeze(0),
            num_samples_per_edge=32 if view.cam.is_xnav else 4,
            valid_ratio=0.16667,
        )
        visible = ok.reshape(-1).cpu()
        if view.valid_mask is not None:
            visible = mask_visible_boxes_cpu(bb2, visible, view.valid_mask, cache)
        projected |= visible
        if view.observed_points is None:
            supported |= visible
        elif len(view.observed_points):
            points = view.observed_points.float().cpu()
            indices = visible.nonzero(as_tuple=True)[0]
            for chunk in indices.split(256):
                if not len(chunk):
                    continue
                inside = obbs[chunk].batch_points_inside_bb3(
                    points[None].expand(len(chunk), -1, -1)
                )
                supported[chunk] |= inside.sum(-1).cpu() >= min_obs_points
    return projected, supported


class XNavGeometry:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self._masks = {}
        self._cameras = {}
        self._reference_masks = {}
        self.repaired_boxes = 0

    def _camera(self, camera):
        data = camera._data.detach().cpu()
        key = data.numpy().tobytes()
        if key not in self._cameras:
            model = (
                int(data.reshape(-1, data.shape[-1])[0, 22]) if camera.is_xnav else None
            )
            self._cameras[key] = camera.to(self.device), model
        return self._cameras[key]

    def _mask(self, bounds, visible, mask):
        cpu = mask.bool().cpu().numpy()
        key = (mask.data_ptr(), tuple(mask.shape), tuple(mask.stride()))
        cached = self._masks.get(key)
        if cached is None or not np.array_equal(cached[0], cpu):
            snapshot = cpu.copy()
            integral = np.pad(
                snapshot.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0))
            )
            if len(self._masks) >= 16:
                self._masks.pop(next(iter(self._masks)))
            self._masks[key] = snapshot, torch.from_numpy(integral).to(self.device)
        integral = self._masks[key][1]
        height, width = cpu.shape
        rectangles = bounds.reshape(-1, 4).trunc().long()
        xa = rectangles[:, 0].clamp_min(0)
        xb = (rectangles[:, 1] + 1).clamp_max(width)
        ya = rectangles[:, 2].clamp_min(0)
        yb = (rectangles[:, 3] + 1).clamp_max(height)
        nonempty = (xb > xa) & (yb > ya)
        xa, xb = xa.clamp_max(width), xb.clamp_min(0)
        ya, yb = ya.clamp_max(height), yb.clamp_min(0)
        hits = integral[yb, xb] - integral[ya, xb] - integral[yb, xa] + integral[ya, xa]
        return visible & nonempty & (hits > 0)

    @torch.inference_mode()
    def visibility(self, boxes, observations, min_obs_points):
        reference = boxes.cpu()
        count = len(reference)
        if not count:
            return torch.zeros(0, dtype=torch.bool), torch.zeros(0, dtype=torch.bool)
        box_scale = float(torch.nan_to_num(reference.T_world_object.t.abs()).max())
        boxes = boxes.to(device=self.device, dtype=torch.float32)
        projected = torch.zeros(count, device=self.device, dtype=torch.bool)
        supported = torch.zeros_like(projected)
        possible_support = torch.zeros_like(projected)
        with torch.autocast(self.device.type, enabled=False):
            for view in observations:
                camera, model = self._camera(view.cam)
                context = (
                    fixed_camera_model(model)
                    if model is not None
                    else contextlib.nullcontext()
                )
                with context:
                    bounds, valid = boxes.get_pseudo_bb2(
                        camera.unsqueeze(0),
                        view.T_world_rig.to(self.device).unsqueeze(0),
                        num_samples_per_edge=32 if camera.is_xnav else 4,
                        valid_ratio=0.16667,
                    )
                visible = valid.reshape(-1)
                if view.valid_mask is not None:
                    visible = self._mask(bounds, visible, view.valid_mask)
                projected |= visible
                if view.observed_points is None:
                    supported |= visible
                    possible_support |= visible
                elif len(view.observed_points):
                    points = view.observed_points.to(
                        device=self.device, dtype=torch.float32
                    )
                    point_scale = float(
                        torch.nan_to_num(view.observed_points.abs()).max()
                    )
                    # A conservative roundoff band around each box face. Only
                    # decisions that could cross the support threshold require
                    # CPU repair; comfortably supported boxes stay on CUDA.
                    tolerance = (
                        64
                        * torch.finfo(torch.float32).eps
                        * (point_scale + box_scale + 1)
                    )
                    indices = visible.nonzero(as_tuple=True)[0]
                    for chunk in indices.split(256):
                        if len(chunk):
                            subset = boxes[chunk]
                            local = subset.T_world_object.inverse() * points[
                                None
                            ].expand(len(chunk), -1, -1)
                            minimum = subset.bb3_min_object.unsqueeze(1)
                            maximum = subset.bb3_max_object.unsqueeze(1)
                            certain = (
                                (local > minimum + tolerance)
                                & (local < maximum - tolerance)
                            ).all(-1)
                            possible = (
                                (local > minimum - tolerance)
                                & (local < maximum + tolerance)
                            ).all(-1)
                            supported[chunk] |= certain.sum(-1) >= min_obs_points
                            possible_support[chunk] |= (
                                possible.sum(-1) >= min_obs_points
                            )
        decisions = torch.stack((projected, supported, possible_support)).cpu()
        ambiguous = (~decisions[1] & decisions[2]).nonzero(as_tuple=True)[0]
        if len(ambiguous):
            repaired = visibility_cpu(
                reference[ambiguous],
                observations,
                min_obs_points,
                self._reference_masks,
            )
            decisions[0, ambiguous], decisions[1, ambiguous] = repaired
            self.repaired_boxes += len(ambiguous)
        return decisions[0], decisions[1]
