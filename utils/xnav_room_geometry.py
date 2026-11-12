"""Validated room prisms and exact intersections with fully oriented boxes."""

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.affinity import translate
from shapely.geometry import MultiPoint, Polygon, shape
from shapely.geometry.base import BaseGeometry

RESULT_SCHEMA_VERSION = 2
BINDING_METHOD = "obb_room_volume_v1"
RELATIVE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class RoomPrism:
    id: int
    name: str
    geometry: BaseGeometry
    z_bounds_m: tuple[float, float]

    def record(self):
        from shapely.geometry import mapping

        return dict(
            id=self.id,
            name=self.name,
            geometry=mapping(self.geometry),
            z_bounds_m=list(self.z_bounds_m),
            area_m2=float(self.geometry.area),
        )


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Room coordinates must be finite numbers")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Room coordinates must be finite numbers")
    return value


def load_room_prisms(path):
    """Read the worker's geometry without repairing invalid topology."""
    document = json.loads(Path(path).read_bytes())
    if not isinstance(document, dict) or any(
        document.get(key) != value
        for key, value in (
            ("schema_version", 1),
            ("coordinate_frame", "scene_xy"),
            ("units", "m"),
            ("volume_type", "vertical_prism"),
        )
    ):
        raise ValueError("Unsupported room polygon representation")
    if type(document["schema_version"]) is not int:
        raise ValueError("Invalid room polygon schema version")
    if _number(document.get("resolution_m")) <= 0:
        raise ValueError("Room resolution must be positive")
    if not isinstance(document.get("rooms"), list):
        raise ValueError("Room polygons must contain a room array")
    rooms, identities = [], set()
    for item in document["rooms"]:
        if not isinstance(item, dict):
            raise ValueError("Invalid room record")
        identity = item.get("id")
        if type(identity) is not int or identity < 0 or identity in identities:
            raise ValueError("Room IDs must be unique nonnegative integers")
        identities.add(identity)
        geometry = item.get("geometry")
        if not isinstance(geometry, dict) or geometry.get("type") not in {
            "Polygon",
            "MultiPolygon",
        }:
            raise ValueError("Rooms require Polygon or MultiPolygon geometry")
        parts = geometry.get("coordinates")
        if geometry["type"] == "Polygon":
            parts = [parts]
        if not isinstance(parts, list) or not parts:
            raise ValueError("Room geometry cannot be empty")
        for part in parts:
            if not isinstance(part, list) or not part:
                raise ValueError("Room polygons require an exterior ring")
            for ring in part:
                if not isinstance(ring, list) or len(ring) < 4:
                    raise ValueError("Room rings require at least four coordinates")
                for point in ring:
                    if not isinstance(point, list) or len(point) != 2:
                        raise ValueError("Room footprints require XY coordinates")
                    for value in point:
                        _number(value)
                if ring[0] != ring[-1]:
                    raise ValueError("Room polygon rings must be closed")
        footprint = shape(geometry)
        if (
            footprint.is_empty
            or not footprint.is_valid
            or not math.isfinite(footprint.area)
            or footprint.area <= 0
        ):
            raise ValueError(f"Invalid polygon topology for room {identity}")
        bounds = item.get("z_bounds_m")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError("Rooms require two Z bounds")
        bottom, top = map(_number, bounds)
        if top < bottom or not math.isfinite(top - bottom):
            raise ValueError("Room Z bounds must be ordered")
        name = item.get("name", f"房间 {identity + 1:02d}")
        if not isinstance(name, str) or not name:
            raise ValueError("Room names must be nonempty strings")
        rooms.append(RoomPrism(identity, name, footprint, (bottom, top)))
    return document, rooms


def box_geometry(box):
    """Return double-precision box geometry, or None for an invalid object."""
    try:
        center = np.array(
            [box[f"{axis}_world_object"] for axis in ("tx", "ty", "tz")],
            dtype=np.float64,
        )
        size = np.array([box[f"scale_{axis}"] for axis in "xyz"], dtype=np.float64)
        quaternion = np.array(
            [box[f"{axis}_world_object"] for axis in ("qw", "qx", "qy", "qz")],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (
        center.shape != (3,)
        or size.shape != (3,)
        or quaternion.shape != (4,)
        or not np.isfinite(center).all()
        or not np.isfinite(size).all()
        or not np.isfinite(quaternion).all()
        or (size <= 0).any()
    ):
        return None
    # Scaling first avoids overflow when normalizing finite quaternions.
    scale = abs(quaternion).max()
    if scale < 1e-12:
        return None
    quaternion /= scale
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    volume = math.prod(float(value) for value in size)
    if not math.isfinite(volume) or volume <= 0:
        return None
    half = size / 2
    extent = abs(rotation) @ half
    with np.errstate(over="ignore", invalid="ignore"):
        if not (
            np.isfinite(extent).all()
            and np.isfinite(center - extent).all()
            and np.isfinite(center + extent).all()
        ):
            return None
    corners = np.array(list(itertools.product((-1.0, 1.0), repeat=3))) * half
    corners = corners @ rotation.T
    faces = []
    for axis in range(3):
        j, k = [other for other in range(3) if other != axis]
        for sign in (-1.0, 1.0):
            normal = sign * rotation[:, axis]
            origin = sign * half[axis] * rotation[:, axis]
            ring = [
                origin + a * half[j] * rotation[:, j] + b * half[k] * rotation[:, k]
                for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            ]
            faces.append((normal, float(normal @ origin), ring))
    edges = [
        (corners[i], corners[j])
        for i, j in itertools.combinations(range(8), 2)
        if (i ^ j).bit_count() == 1
    ]
    return dict(
        center=center,
        extent=extent,
        volume=volume,
        faces=faces,
        edges=edges,
    )


def _clip_height(points, level, above):
    result = []
    for a, b in zip(points, points[1:] + points[:1]):
        inside_a = a[2] >= level if above else a[2] <= level
        inside_b = b[2] >= level if above else b[2] <= level
        if inside_a:
            result.append(a)
        if inside_a != inside_b:
            result.append(a + (b - a) * ((level - a[2]) / (b[2] - a[2])))
    return result


def intersection_volume(geometry, room):
    """Integrate oriented face heights over the prism's XY footprint.

    The divergence of (0, 0, z) is one. Vertical room walls contribute zero;
    every other clipped face contributes its signed projected area times the
    height at that area's centroid. This preserves holes and arbitrary tilt.
    All coordinates are translated by the object center before calculation.
    """
    center, extent = geometry["center"], geometry["extent"]
    low = max(room.z_bounds_m[0] - center[2], -extent[2])
    high = min(room.z_bounds_m[1] - center[2], extent[2])
    if high <= low:
        return 0.0
    footprint = translate(room.geometry, xoff=-center[0], yoff=-center[1])
    contributions = []
    for normal, offset, ring in geometry["faces"]:
        if normal[2] == 0:
            continue
        clipped = _clip_height(_clip_height(ring, low, True), high, False)
        if len(clipped) < 3:
            continue
        projected = Polygon(np.asarray(clipped)[:, :2])
        if projected.area == 0:
            continue
        part = projected.intersection(footprint)
        if part.area == 0:
            continue
        centroid = part.centroid
        height = (offset - normal[0] * centroid.x - normal[1] * centroid.y) / normal[2]
        contributions.append(float(np.sign(normal[2]) * part.area * height))
    # Only actual cuts need caps; existing horizontal box faces already count.
    for level, sign in ((low, -1.0), (high, 1.0)):
        if not -extent[2] < level < extent[2]:
            continue
        points = []
        for a, b in geometry["edges"]:
            if min(a[2], b[2]) <= level <= max(a[2], b[2]):
                if a[2] == b[2]:
                    points.extend((a[:2], b[:2]))
                else:
                    points.append((a + (b - a) * ((level - a[2]) / (b[2] - a[2])))[:2])
        if len(points) >= 3:
            cap = MultiPoint(points).convex_hull.intersection(footprint)
            contributions.append(float(sign * level * cap.area))
    volume = math.fsum(contributions)
    tolerance = RELATIVE_TOLERANCE * geometry["volume"]
    if not math.isfinite(volume) or not -tolerance <= volume <= (
        geometry["volume"] + tolerance
    ):
        raise ValueError("Room intersection volume is outside the object volume")
    return min(geometry["volume"], max(0.0, volume))
