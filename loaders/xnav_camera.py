"""NumPy projections for the six camera models exported by x-nav.

The equations and domain checks follow x-nav's rpg_vikit implementations.
Exported intrinsics describe the original image; vio_scale is not reapplied.
All rotations below map virtual-camera coordinates into the physical camera.
"""

from dataclasses import dataclass

import cv2
import numpy as np

MODELS = {
    "Pinhole",
    "PolynomialCamera",
    "EquidistantCamera",
    "OmniRadtanCamera",
    "EucmCamera",
    "TripleSphereCamera",
}


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


class XNavCamera:
    def __init__(self, entry):
        self.entry = entry
        self.model = entry["model"]
        if self.model not in MODELS:
            raise ValueError(f"Unsupported x-nav camera model: {self.model}")
        self.camera_id = int(entry["cam_id"])
        self.width, self.height = int(entry["width"]), int(entry["height"])
        self.p = dict(entry["intrinsics"])
        self.d = dict(entry.get("distortion") or {})
        required = ["fx", "fy", "cx", "cy"]
        required += {
            "PolynomialCamera": ["skew"],
            "OmniRadtanCamera": ["xi"],
            "EucmCamera": ["alpha", "beta"],
            "TripleSphereCamera": ["xi", "lambda", "alpha", "b", "c"],
        }.get(self.model, [])
        if any(k not in self.p for k in required):
            raise ValueError(f"Camera {self.camera_id}: missing {required}")
        dist_keys = {
            "Pinhole": ["k1", "k2", "p1", "p2"],
            "OmniRadtanCamera": ["k1", "k2", "p1", "p2"],
            "EquidistantCamera": ["k1", "k2", "k3", "k4"],
            "PolynomialCamera": [f"k{i}" for i in range(2, 8)],
        }.get(self.model, [])
        if any(k not in self.d for k in dist_keys):
            raise ValueError(f"Camera {self.camera_id}: missing distortion {dist_keys}")
        if (
            min(self.width, self.height) <= 0
            or min(self.p["fx"], self.p["fy"]) <= 0
            or not np.isfinite(list(self.p.values()) + list(self.d.values())).all()
        ):
            raise ValueError("Invalid camera dimensions or intrinsics")
        if self.model == "EucmCamera":
            if not 0 <= self.p["alpha"] <= 1 or self.p["beta"] <= 0:
                raise ValueError("Invalid EUCM alpha/beta")
        if self.model == "OmniRadtanCamera" and self.p["xi"] < 0:
            raise ValueError("OmniRadtan xi must be nonnegative")
        if self.model == "TripleSphereCamera":
            det = self.p["fx"] * self.p["fy"] - self.p["b"] * self.p["c"]
            if abs(1 - self.p["alpha"]) <= 1e-12 or abs(det) <= 1e-12:
                raise ValueError("Singular TripleSphere calibration")
        self.max_theta = np.pi - 1e-6
        if self.model == "EquidistantCamera":
            coefs = [1] + [(2 * i + 1) * self.d[f"k{i}"] for i in range(1, 5)]
            roots = np.polynomial.polynomial.polyroots(coefs)
            roots = [
                np.sqrt(r.real)
                for r in roots
                if abs(r.imag) < 1e-10 and 0 < r.real < self.max_theta**2
            ]
            if roots:
                first = min(roots)
                margin = max(1e-10, first * 1e-10)
                self.max_theta = first - margin
                for _ in range(32):
                    derivative = 1 + sum(
                        (2 * i + 1) * self.d[f"k{i}"] * self.max_theta ** (2 * i)
                        for i in range(1, 5)
                    )
                    if self.max_theta <= 0 or derivative > 1e-9:
                        break
                    margin *= 2
                    self.max_theta = first - margin
                if self.max_theta <= 0 or derivative <= 1e-9:
                    raise ValueError("Equidistant radial mapping is not monotonic")
            corners = np.array(
                [
                    [0, 0],
                    [self.width - 1, 0],
                    [0, self.height - 1],
                    [self.width - 1, self.height - 1],
                ]
            )
            radius = np.linalg.norm(self._normalized(corners), axis=-1).max()
            if radius > self._radial(self.max_theta) + 1e-12:
                raise ValueError("Equidistant image corners exceed invertible domain")

    def _normalized(self, pixels):
        p = self.p
        x = (pixels[..., 0] - p["cx"]) / p["fx"]
        y = (pixels[..., 1] - p["cy"]) / p["fy"]
        if self.model == "PolynomialCamera" and abs(self.d["k2"]) > 1e-7:
            x -= y * p["skew"] / p["fx"]
        if self.model == "TripleSphereCamera":
            du, dv = pixels[..., 0] - p["cx"], pixels[..., 1] - p["cy"]
            det = p["fx"] * p["fy"] - p["b"] * p["c"]
            x, y = (
                (p["fy"] * du - p["b"] * dv) / det,
                (p["fx"] * dv - p["c"] * du) / det,
            )
        return np.stack([x, y], axis=-1)

    def _radial(self, theta):
        if self.model == "PolynomialCamera":
            return theta + sum(self.d[f"k{i}"] * theta**i for i in range(2, 8))
        return theta + sum(self.d[f"k{i}"] * theta ** (2 * i + 1) for i in range(1, 5))

    def _distort(self, xy):
        x, y = xy[..., 0], xy[..., 1]
        r2 = x * x + y * y
        if self.model == "Pinhole":
            if abs(self.d["k1"]) <= 1e-7:
                return xy.copy()
            r2 = np.minimum(r2, 1.0)  # x-nav's PinholeCamera convention
        k1, k2, p1, p2 = (self.d[k] for k in ["k1", "k2", "p1", "p2"])
        radial = 1 + k1 * r2 + k2 * r2 * r2
        return np.stack(
            [
                x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
            ],
            axis=-1,
        )

    def project(self, points, image_bounds=True):
        """Return pixels and mask; NaN pixels are outside the camera domain."""
        points = np.asarray(points, dtype=np.float64)
        x, y, z = np.moveaxis(points, -1, 0)
        norm = np.linalg.norm(points, axis=-1)
        valid = np.isfinite(points).all(axis=-1) & (norm > 1e-12)
        p = self.p
        with np.errstate(all="ignore"):
            if self.model in ("Pinhole", "PolynomialCamera", "EquidistantCamera"):
                distorted_poly = (
                    self.model == "PolynomialCamera" and abs(self.d["k2"]) > 1e-7
                )
                if distorted_poly or self.model == "EquidistantCamera":
                    theta = np.arccos(np.clip(z / norm, -1, 1))
                    radius = np.hypot(x, y)
                    scale = self._radial(theta) / np.maximum(radius, 1e-30)
                    nx, ny = x * scale, y * scale
                    valid &= (theta <= self.max_theta) & ((radius > 1e-12) | (z > 0))
                else:
                    nx, ny = x / z, y / z
                    valid &= z > 1e-12
                if self.model == "Pinhole":
                    nx, ny = np.moveaxis(self._distort(np.stack([nx, ny], -1)), -1, 0)
            elif self.model == "EucmCamera":
                den = (
                    p["alpha"] * np.sqrt(p["beta"] * (x * x + y * y) + z * z)
                    + (1 - p["alpha"]) * z
                )
                nx, ny = x / den, y / den
                valid &= den > 1e-12
            elif self.model == "OmniRadtanCamera":
                xi = p["xi"]
                den = z + xi * norm
                valid &= (np.abs(den) > 1e-12) & (
                    z > -min(xi, 1 / xi if xi else 0) * norm
                )
                nx, ny = np.moveaxis(
                    self._distort(np.stack([x / den, y / den], -1)), -1, 0
                )
            else:
                x, y, z = x / norm, y / norm, z / norm
                z1 = z + p["xi"]
                z2 = z1 + p["lambda"] * np.sqrt(x * x + y * y + z1 * z1)
                den = z2 + p["alpha"] / (1 - p["alpha"]) * np.sqrt(
                    x * x + y * y + z2 * z2
                )
                nx, ny = x / den, y / den
                valid &= den > 1e-12
            skew = (
                p["skew"] if self.model == "PolynomialCamera" and distorted_poly else 0
            )
            u = p["fx"] * nx + (p.get("b", 0) + skew) * ny + p["cx"]
            v = p.get("c", 0) * nx + p["fy"] * ny + p["cy"]
        pixels = np.stack([u, v], -1)
        valid &= np.isfinite(pixels).all(axis=-1)
        if image_bounds:
            valid &= (u >= 0) & (u < self.width - 1) & (v >= 0) & (v < self.height - 1)
        return np.where(valid[..., None], pixels, np.nan), valid

    def unproject(self, pixels):
        """Return unit bearings on the invertible branch and their validity."""
        pixels = np.asarray(pixels, dtype=np.float64)
        n = self._normalized(pixels)
        x, y = np.moveaxis(n, -1, 0)
        p = self.p
        valid = np.isfinite(n).all(axis=-1)
        with np.errstate(all="ignore"):
            if self.model == "OmniRadtanCamera" or (
                self.model == "Pinhole" and abs(self.d["k1"]) > 1e-7
            ):
                # Solve the actual forward distortion, including Pinhole's clamp.
                target, estimate = n.copy(), n.copy()
                for _ in range(30):
                    residual = self._distort(estimate) - target
                    eps = 1e-6
                    jx = (
                        self._distort(estimate + [eps, 0])
                        - self._distort(estimate - [eps, 0])
                    ) / (2 * eps)
                    jy = (
                        self._distort(estimate + [0, eps])
                        - self._distort(estimate - [0, eps])
                    ) / (2 * eps)
                    det = jx[..., 0] * jy[..., 1] - jy[..., 0] * jx[..., 1]
                    dx = (
                        jy[..., 1] * residual[..., 0] - jy[..., 0] * residual[..., 1]
                    ) / det
                    dy = (
                        -jx[..., 1] * residual[..., 0] + jx[..., 0] * residual[..., 1]
                    ) / det
                    estimate -= np.clip(np.stack([dx, dy], -1), -0.5, 0.5)
                valid &= (
                    np.linalg.norm(self._distort(estimate) - target, axis=-1) < 1e-7
                )
                x, y = np.moveaxis(estimate, -1, 0)
            r2 = x * x + y * y
            if self.model in ("PolynomialCamera", "EquidistantCamera") and not (
                self.model == "PolynomialCamera" and abs(self.d["k2"]) <= 1e-7
            ):
                radius = np.sqrt(r2)
                upper = (
                    self.max_theta
                    if self.model == "EquidistantCamera"
                    else np.pi / 2 - 1e-6
                )
                lo, hi = np.zeros_like(radius), np.full_like(radius, upper)
                valid &= radius <= self._radial(upper)
                for _ in range(45):
                    mid = (lo + hi) / 2
                    below = self._radial(mid) < radius
                    lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
                theta = (lo + hi) / 2
                scale = np.sin(theta) / np.maximum(radius, 1e-30)
                ray = np.stack([x * scale, y * scale, np.cos(theta)], -1)
            elif self.model == "EucmCamera":
                a, b = p["alpha"], p["beta"]
                disc = 1 - (2 * a - 1) * b * r2
                valid &= disc >= 0
                z = (1 - a * a * b * r2) / (a * np.sqrt(np.maximum(0, disc)) + 1 - a)
                ray = _unit(np.stack([x, y, z], -1))
            elif self.model == "TripleSphereCamera":
                k = p["alpha"] / (1 - p["alpha"])
                disc = 1 + (1 - k * k) * r2
                valid &= disc >= -1e-12
                s = (k + np.sqrt(np.maximum(0, disc))) / (1 + r2)
                valid &= s > 1e-12
                qx, qy, qz = s * x, s * y, s - k
                for offset in (p["lambda"], p["xi"]):
                    disc = 1 - offset * offset * (qx * qx + qy * qy)
                    valid &= disc >= -1e-12
                    s = offset * qz + np.sqrt(np.maximum(0, disc))
                    valid &= s > 1e-12
                    qx, qy, qz = s * qx, s * qy, s * qz - offset
                ray = _unit(np.stack([qx, qy, qz], -1))
            elif self.model == "OmniRadtanCamera":
                xi = p["xi"]
                disc = 1 + (1 - xi * xi) * r2
                valid &= disc >= -1e-10
                den = xi + np.sqrt(np.maximum(0, disc))
                valid &= np.abs(den) > 1e-12
                ray = _unit(np.stack([x, y, 1 - xi * (r2 + 1) / den], -1))
            else:
                ray = _unit(np.stack([x, y, np.ones_like(x)], -1))
        reproj, forward_ok = self.project(ray, image_bounds=False)
        valid &= forward_ok & (np.linalg.norm(reproj - pixels, axis=-1) < 1e-4)
        return np.where(valid[..., None], ray, np.nan), valid


# Columns are virtual-camera x/right, y/down, z/forward in the source camera.
VIEW_ROTATIONS = {
    "front": np.eye(3),
    "back": np.diag([-1.0, 1.0, -1.0]),
    "left": np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    "right": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
    "up": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    "down": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
}


@dataclass
class RectifiedView:
    view_id: str
    R_camera_view: np.ndarray
    K: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray
    valid_mask: np.ndarray
    cam: object = None

    def remap(self, image):
        return cv2.remap(
            image,
            self.map_x,
            self.map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def project(self, points):
        z = points[:, 2]
        with np.errstate(all="ignore"):
            uv = points[:, :2] / z[:, None] * self.K[[0, 1], [0, 1]] + self.K[:2, 2]
        return uv, (z > 1e-3) & np.isfinite(uv).all(-1)

    def metadata(self):
        return dict(
            camera_model=self.cam.type_string(),
            camera_tensor=self.cam._data.tolist(),
            K=self.K.tolist(),
            size=self.valid_mask.shape[0],
            R_camera_view=self.R_camera_view.tolist(),
            valid_pixel_fraction=float(self.valid_mask.mean()),
        )

    def box_is_valid(self, boxes, min_fraction=0.5):
        """Boxer-order xyxy boxes: center valid and >=50% valid image support."""
        out = []
        h, w = self.valid_mask.shape
        for x1, x2, y1, y2 in np.asarray(boxes):
            xa, xb = max(0, int(np.floor(x1))), min(w, int(np.ceil(x2)))
            ya, yb = max(0, int(np.floor(y1))), min(h, int(np.ceil(y2)))
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            out.append(
                xb > xa
                and yb > ya
                and 0 <= cx < w
                and 0 <= cy < h
                and self.valid_mask[cy, cx]
                and self.valid_mask[ya:yb, xa:xb].mean() >= min_fraction
            )
        return np.asarray(out, dtype=bool)


class NativeView(RectifiedView):
    """An original distorted image, resized/padded without lens remapping."""

    def __init__(self, camera, size, *, unproject=None):
        from utils.tw.camera import CameraTW

        ratio = size / max(camera.width, camera.height)
        self.resized_size = (
            max(1, round(camera.width * ratio)),
            max(1, round(camera.height * ratio)),
        )
        self.padding = (
            (size - self.resized_size[0]) // 2,
            (size - self.resized_size[1]) // 2,
        )
        self.source = camera
        self._unproject = unproject or camera.unproject
        sx, sy = (
            self.resized_size[0] / camera.width,
            self.resized_size[1] / camera.height,
        )
        self.affine = np.array(
            [sx, sy, (sx - 1) / 2 + self.padding[0], (sy - 1) / 2 + self.padding[1]]
        )
        cam = (
            CameraTW.from_xnav(camera.entry)
            .scale((sx, sy))
            .crop((-self.padding[0], -self.padding[1]), (size, size))
        )
        yy, xx = np.mgrid[:size, :size]
        pixels = (np.stack((xx, yy), -1) - self.affine[2:]) / self.affine[:2]
        bearings, valid = self._unproject(pixels)
        valid &= (bearings[..., 2] > 0) & (
            (pixels >= 0) & (pixels <= [camera.width - 1, camera.height - 1])
        ).all(-1)
        if not valid.any():
            raise ValueError(
                f"Camera {camera.camera_id} has no invertible front image region"
            )
        super().__init__("native", np.eye(3), None, None, None, valid, cam)

    def remap(self, image):
        size = self.valid_mask.shape[0]
        out = np.zeros((size, size, 3), dtype=image.dtype)
        w, h = self.resized_size
        x, y = self.padding
        out[y : y + h, x : x + w] = cv2.resize(
            image, (w, h), interpolation=cv2.INTER_LINEAR
        )
        out[~self.valid_mask] = 0
        return out

    def project(self, points):
        pixels, valid = self.source.project(points)
        valid &= points[:, 2] > 1e-3
        indices = np.flatnonzero(valid)
        bearings, invertible = self._unproject(pixels[indices])
        valid[indices] &= invertible & (
            np.linalg.norm(bearings - _unit(points[indices]), axis=-1) < 1e-5
        )
        return pixels * self.affine[:2] + self.affine[2:], valid

    def metadata(self):
        return dict(
            camera_model=self.source.model,
            camera_tensor=self.cam._data.tolist(),
            source_calibration=self.source.entry,
            size=self.valid_mask.shape[0],
            source_to_image=self.affine.tolist(),
            resized_size=self.resized_size,
            padding=self.padding,
            hemisphere="front",
            R_camera_view=np.eye(3).tolist(),
            valid_pixel_fraction=float(self.valid_mask.mean()),
        )


def make_views(camera, size=960, fov=100.0, *, unproject=None):
    from loaders.base_loader import BaseLoader

    if size < 2 or not 0 < fov < 180:
        raise ValueError("Invalid virtual image size/FOV")
    f = (size - 1) / (2 * np.tan(np.deg2rad(fov / 2)))
    K = np.array([[f, 0, (size - 1) / 2], [0, f, (size - 1) / 2], [0, 0, 1.0]])
    yy, xx = np.mgrid[:size, :size]
    rays = np.stack([(xx - K[0, 2]) / f, (yy - K[1, 2]) / f, np.ones_like(xx)], -1)
    views = []
    unproject = unproject or camera.unproject
    for name, rotation in VIEW_ROTATIONS.items():
        source_rays = rays @ rotation.T
        pixels, valid = camera.project(source_rays)
        # Reject projections on another branch (e.g. EUCM's rear fold).
        bearings, invertible = unproject(np.nan_to_num(pixels))
        valid &= invertible & (
            np.sum(bearings * _unit(source_rays), axis=-1) > 1 - 1e-7
        )
        if not valid.any():
            continue
        maps = np.where(valid[..., None], pixels, -1).astype(np.float32)
        views.append(
            RectifiedView(
                name,
                rotation.copy(),
                K.copy(),
                maps[..., 0],
                maps[..., 1],
                valid,
                BaseLoader.pinhole_from_K(size, size, f, f, K[0, 2], K[1, 2]),
            )
        )
    return views
