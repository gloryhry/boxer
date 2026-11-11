"""Export room footprints on the source point cloud's XY plane."""

import numpy as np


MAX_HOLE_AREA_M2 = 0.25


def _fill_small_holes(footprint, labels, room_id, resolution):
    from shapely import Polygon, contains_xy, union_all

    parts = [footprint] if footprint.geom_type == "Polygon" else footprint.geoms
    filled = []
    changed = False
    for part in parts:
        kept_holes = []
        for ring in part.interiors:
            hole = Polygon(ring)
            if hole.area * resolution ** 2 > MAX_HOLE_AREA_M2 + 1e-12:
                kept_holes.append(ring)
                continue

            left, bottom, right, top = map(int, hole.bounds)
            window = labels[bottom:top, left:right]
            y, x = np.nonzero((window >= 0) & (window != room_id))
            if contains_xy(hole, left + x + 0.5, bottom + y + 0.5).any():
                kept_holes.append(ring)

        if len(kept_holes) < len(part.interiors):
            filled.append(Polygon(part.exterior, kept_holes))
            changed = True
        else:
            filled.append(part)

    if not changed:
        return footprint
    # A filled hole can absorb an island of the same room.
    return union_all(filled) if len(filled) > 1 else filled[0]


def room_polygons(label_map, transform, room_ids):
    """Convert labeled grid cells to closed polygon rings in metres."""
    labels = np.asarray(label_map)
    if labels.shape != (transform["height"], transform["width"]) or transform["row_order"] != "y_up":
        raise ValueError("房间轮廓需要与坐标变换一致的 Y 向上标签图")
    resolution = float(transform["resolution"])
    origin = np.asarray(transform["origin"], dtype=float)
    result = dict(schema_version=1, coordinate_frame="scene_xy", units="m", resolution_m=resolution, rooms=[])
    room_ids = np.asarray(list(room_ids), dtype=np.int64)
    if not len(room_ids):
        return result

    from shapely import box, union_all
    from shapely.geometry.polygon import orient

    # Merge horizontal runs on the integer grid. This keeps shared cell
    # edges exact and uses one rectangle per run instead of one per cell.
    padded = np.pad(labels, ((0, 0), (1, 1)), constant_values=-1)
    rows, columns = np.nonzero(padded[:, 1:] != padded[:, :-1])
    same_row = rows[:-1] == rows[1:]
    y, left, right = rows[:-1][same_row], columns[:-1][same_row], columns[1:][same_row]
    owners = labels[y, left]
    kept = (owners >= 0) & np.isin(owners, room_ids)
    y, left, right, owners = y[kept], left[kept], right[kept], owners[kept]
    if not len(owners):
        return result
    order = np.argsort(owners, kind="stable")
    y, left, right, owners = y[order], left[order], right[order], owners[order]
    cuts = np.r_[0, np.flatnonzero(owners[1:] != owners[:-1]) + 1, len(owners)]

    for start, stop in zip(cuts[:-1], cuts[1:]):
        group = slice(start, stop)
        room_id = int(owners[start])
        footprint = union_all(box(left[group], y[group], right[group], y[group] + 1)).simplify(0)
        footprint = _fill_small_holes(footprint, labels, room_id, resolution)
        parts = [footprint] if footprint.geom_type == "Polygon" else footprint.geoms
        coordinates = []
        for part in parts:
            part = orient(part, sign=1.)
            rings = (part.exterior, *part.interiors)
            coordinates.append([(np.asarray(ring.coords) * resolution + origin).tolist() for ring in rings])
        result["rooms"].append(dict(id=room_id, area_m2=float(footprint.area * resolution ** 2),
                                    geometry=dict(type=footprint.geom_type,
                                                  coordinates=coordinates[0] if len(coordinates) == 1 else coordinates)))
    return result
