"""CPU-only, full-cloud voting of final oriented boxes into room instances."""

import argparse
import json
from collections import Counter
from pathlib import Path

from utils.xnav_result_store import _atomic_json

MIN_SUPPORT_POINTS = 10
MIN_VOTE_RATIO = 0.6
BOX_TOLERANCE_M = 1e-5
CHUNK_SIZE = 131_072


def read_vertices(path):
    from plyfile import PlyData

    vertices = PlyData.read(str(path), mmap="r")["vertex"].data
    if not {"x", "y", "z", "label"}.issubset(vertices.dtype.names or ()):
        raise ValueError("Room PLY must contain x, y, z and label")
    return vertices


def _box_geometry(box):
    import numpy as np

    center = np.array([box[f"{axis}_world_object"] for axis in ("tx", "ty", "tz")])
    size = np.array([box[f"scale_{axis}"] for axis in "xyz"])
    q = np.array([box[f"{axis}_world_object"] for axis in ("qw", "qx", "qy", "qz")])
    length = np.linalg.norm(q)
    if (
        not np.isfinite(center).all()
        or not np.isfinite(size).all()
        or not np.isfinite(q).all()
        or length < 1e-12
        or (size <= 0).any()
    ):
        return None
    w, x, y, z = q / length
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    half = size / 2 + BOX_TOLERANCE_M
    extent = abs(rotation) @ half
    return center, half, rotation, center - extent, center + extent


def bind_objects(payload):
    """Use every finite PLY point, including label=-1 in each box's denominator."""
    import numpy as np

    rooms = {room["id"]: room for room in payload["rooms"]}
    tracking = payload["tracking_enabled"]
    objects = payload["objects"] if tracking else []
    geometries = [_box_geometry(box) for box in objects]
    counts = [0 for _ in objects]
    votes = [Counter() for _ in objects]
    if objects:
        vertices = read_vertices(payload["ply_path"])
        if not np.issubdtype(vertices["label"].dtype, np.integer):
            raise ValueError("Room labels must be integers")
        for start in range(0, len(vertices), CHUNK_SIZE):
            block = vertices[start : start + CHUNK_SIZE]
            points = np.column_stack([block[axis] for axis in "xyz"]).astype(float)
            labels = block["label"]
            finite = np.isfinite(points).all(axis=1)
            points, labels = points[finite], labels[finite]
            if any(
                int(label) not in rooms for label in np.unique(labels) if label >= 0
            ):
                raise ValueError("Room PLY references a missing room")
            if (labels < -1).any():
                raise ValueError("Room labels must be -1 or nonnegative")
            for index, geometry in enumerate(geometries):
                if geometry is None:
                    continue
                center, half, rotation, low, high = geometry
                candidates = ((points >= low) & (points <= high)).all(axis=1)
                local = (points[candidates] - center) @ rotation
                inside = (abs(local) <= half).all(axis=1)
                owners = labels[candidates][inside]
                counts[index] += len(owners)
                identities, numbers = np.unique(owners[owners >= 0], return_counts=True)
                votes[index].update(dict(zip(identities.tolist(), numbers.tolist())))
    items = []
    for box, geometry, count, owners in zip(objects, geometries, counts, votes):
        ordered = sorted(owners.items(), key=lambda pair: (-pair[1], pair[0]))
        winner, support = ordered[0] if ordered else (None, 0)
        ratio = support / count if count else 0
        reason = (
            "invalid_bbox"
            if geometry is None
            else "no_rooms"
            if not rooms
            else "no_points"
            if not count
            else "no_labeled_points"
            if not ordered
            else "insufficient_support"
            if support < MIN_SUPPORT_POINTS
            else "ambiguous_room"
            if ratio < MIN_VOTE_RATIO
            else None
        )
        room_id = winner if reason is None else None
        items.append(
            dict(
                box,
                room_id=room_id,
                room_name=rooms[room_id]["name"] if room_id is not None else None,
                point_count=count,
                labeled_point_count=sum(owners.values()),
                support_points=support,
                vote_ratio=ratio,
                reason=reason,
                votes=[
                    dict(room_id=identity, point_count=number)
                    for identity, number in ordered
                ],
            )
        )
    assigned = sum(item["room_id"] is not None for item in items)
    return dict(
        schema_version=1,
        job_id=payload["job_id"],
        map_id=payload["map_id"],
        segmentation_job_id=payload["segmentation_job_id"],
        algorithm="voxel_connectivity",
        source_signature=payload["source_signature"],
        global_map_fingerprint=payload["global_map_fingerprint"],
        coordinate_frame="map",
        unit="metre",
        up_axis="+Z",
        parameters=dict(
            min_support_points=MIN_SUPPORT_POINTS,
            min_vote_ratio=MIN_VOTE_RATIO,
            box_tolerance_m=BOX_TOLERANCE_M,
        ),
        summary=dict(
            object_count=len(items),
            assigned_count=assigned,
            unassigned_count=len(items) - assigned,
            reason="tracking_disabled"
            if not tracking
            else "no_tracked_objects"
            if not items
            else None,
        ),
        items=items,
    )


def preview_cloud(path, room_id=None, *, limit=500_000):
    """Sample XYZ and RGB together from the labeled artifact, never the map preview."""
    import numpy as np

    vertices = read_vertices(path)
    keep = np.ones(len(vertices), dtype=bool)
    for axis in "xyz":
        keep &= np.isfinite(vertices[axis])
    if room_id is not None:
        keep &= vertices["label"] == room_id
    indices = np.flatnonzero(keep)
    if len(indices) > limit:
        indices = indices[np.linspace(0, len(indices) - 1, limit, dtype=int)]
    selected = vertices[indices]
    data = np.empty((len(selected), 6), dtype="<f4")
    for column, name in enumerate(("x", "y", "z", "red", "green", "blue")):
        data[:, column] = selected[name]
    data[:, 3:] /= 255
    return data.tobytes()


def main():
    from potential_field import _watch_owner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _watch_owner()
    result = bind_objects(json.loads(args.input.read_bytes()))
    _atomic_json(args.output, result)
    print(
        f"Bound {result['summary']['assigned_count']}/{len(result['items'])} objects",
        flush=True,
    )


if __name__ == "__main__":
    main()
