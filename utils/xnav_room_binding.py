"""Bind final oriented objects to room prisms using exact intersection volumes."""

import argparse
import json
from pathlib import Path

from utils.xnav_result_store import _atomic_json
from utils.xnav_room_geometry import (
    BINDING_METHOD,
    RELATIVE_TOLERANCE,
    RESULT_SCHEMA_VERSION,
    box_geometry,
    intersection_volume,
    load_room_prisms,
)

MIN_OVERLAP_RATIO = 0.6


def bind_objects(payload):
    from shapely import STRtree
    from shapely.geometry import box as rectangle

    _, rooms = load_room_prisms(payload["room_polygons_path"])
    index = STRtree([room.geometry for room in rooms])
    names = {room.id: room.name for room in rooms}
    tracking = payload["tracking_enabled"]
    objects = payload["objects"] if tracking else []
    items = []
    for item in objects:
        geometry = box_geometry(item)
        overlaps = []
        if geometry is not None:
            center, extent = geometry["center"], geometry["extent"]
            candidates = index.query(
                rectangle(
                    center[0] - extent[0],
                    center[1] - extent[1],
                    center[0] + extent[0],
                    center[1] + extent[1],
                )
            )
            for candidate in candidates:
                room = rooms[int(candidate)]
                volume = intersection_volume(geometry, room)
                ratio = volume / geometry["volume"]
                if ratio > RELATIVE_TOLERANCE:
                    overlaps.append(
                        dict(
                            room_id=room.id,
                            intersection_volume_m3=volume,
                            overlap_ratio=ratio,
                        )
                    )
        overlaps.sort(
            key=lambda value: (-value["intersection_volume_m3"], value["room_id"])
        )
        winner = overlaps[0] if overlaps else None
        ratio = winner["overlap_ratio"] if winner else 0.0
        tied = len(overlaps) > 1 and (
            abs(ratio - overlaps[1]["overlap_ratio"]) <= RELATIVE_TOLERANCE
        )
        reason = (
            "invalid_bbox"
            if geometry is None
            else "no_rooms"
            if not rooms
            else "no_overlap"
            if not overlaps
            else "ambiguous_overlap"
            if tied
            else "insufficient_overlap"
            if ratio + RELATIVE_TOLERANCE < MIN_OVERLAP_RATIO
            else None
        )
        room_id = winner["room_id"] if reason is None else None
        items.append(
            dict(
                item,
                room_id=room_id,
                room_name=names.get(room_id),
                object_volume_m3=geometry["volume"] if geometry else None,
                intersection_volume_m3=winner["intersection_volume_m3"]
                if winner
                else 0.0,
                overlap_ratio=ratio,
                overlaps=overlaps,
                reason=reason,
            )
        )
    assigned = sum(item["room_id"] is not None for item in items)
    return dict(
        schema_version=RESULT_SCHEMA_VERSION,
        binding_method=BINDING_METHOD,
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
            min_overlap_ratio=MIN_OVERLAP_RATIO,
            relative_tolerance=RELATIVE_TOLERANCE,
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
