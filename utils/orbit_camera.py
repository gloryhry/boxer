"""Display-independent Z-up orbit/follow math for the x-nav viewer."""

import math

import numpy as np


class OrbitCamera:
    def __init__(self, center, radius, follow=False):
        self.center = np.asarray(center, dtype=float)
        self.radius = max(float(radius), 1.0)
        self.target = self.center.copy()
        self.distance = self.radius * 2.8
        self.azimuth = 45.0
        self.elevation = 30.0
        self.follow = follow
        self.offset = np.zeros(3)
        self.pose = None
        self.message = ""

    def set_pose(self, matrix, snap=False):
        pose = np.asarray(matrix, dtype=float) if matrix is not None else None
        self.pose = (
            pose[:3, 3].copy()
            if pose is not None and pose.shape == (4, 4) and np.isfinite(pose).all()
            else None
        )
        self.message = "" if self.pose is not None else "No camera pose in this frame"
        if snap and self.follow and self.pose is not None:
            self.target = self.pose + self.offset

    def update(self, seconds):
        if self.follow and self.pose is not None:
            alpha = 1 - math.exp(-max(seconds, 0) / 0.12)
            self.target += (self.pose + self.offset - self.target) * alpha

    def focus(self):
        self.offset[:] = 0
        if self.pose is not None:
            self.target = self.pose.copy()

    def overview(self, aspect=1.0):
        self.follow = False
        self.offset[:] = 0
        self.target = self.center.copy()
        self.distance = self.radius * 2.8 / min(1.0, max(aspect, 0.1))
        self.azimuth, self.elevation = 45.0, 30.0

    def rotate(self, dx, dy, height):
        scale = 180.0 / max(height, 1)
        self.azimuth -= dx * scale
        self.elevation = float(np.clip(self.elevation + dy * scale, -89, 89))

    def pan(self, dx, dy, height):
        az, el = np.radians([self.azimuth, self.elevation])
        right = np.array([-np.sin(az), np.cos(az), 0])
        up = np.array([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)])
        scale = 2 * self.distance * math.tan(math.radians(22.5)) / max(height, 1)
        delta = (-right * dx + up * dy) * scale
        self.target += delta
        if self.follow:
            self.offset += delta

    def zoom(self, ticks):
        self.distance = float(
            np.clip(
                self.distance * math.exp(-float(np.clip(ticks, -100, 100)) * 0.1),
                0.05,
                max(100.0, self.radius * 20),
            )
        )

    @property
    def clipping(self):
        return max(0.005, self.distance / 10000), max(
            1000.0, self.distance + self.radius * 4
        )
