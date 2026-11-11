"""Lossless image/detection ownership, shared by CSV export and HTTP replay.

OBB tensors deliberately contain no timestamps or provenance. Frame journals
store each association once; only final objects contain the expanded history.
"""

import csv
import io
import json
from collections import defaultdict
from pathlib import Path

from utils.xnav_result_store import _atomic_file

PROVENANCE_VERSION = 1
BOX3D_COLUMNS = (
    "time_ns,tx_world_object,ty_world_object,tz_world_object,qw_world_object,"
    "qx_world_object,qy_world_object,qz_world_object,scale_x,scale_y,scale_z,"
    "name,instance,sem_id,prob"
).split(",")
BOX2D_COLUMNS = (
    "time_ns,frame_id,sensor,device,img_width,img_height,x1,y1,x2,y2,name,"
    "instance,sem_id,prob,detection_id,object_id,camera_id,view_id,image_file,"
    "pose_time_ns"
).split(",")
SOURCE_COLUMNS = [
    "object_id",
    "source_detection_id",
    "image_file",
    "frame_id",
    "pose_time_ns",
]
ASSOCIATION_COLUMNS = [*BOX2D_COLUMNS, "bbox3d_id", "track_id"]


def frame_id(frame):
    return frame["frame_id"] if "frame_id" in frame else frame["keyframe_id"]


def detection_id(frame, camera_id, view_id, index):
    return f"f{frame_id(frame)}:c{camera_id}:v{view_id}:d{index}"


def image_file(frame, camera_id):
    source = next(s for s in frame["sources"] if s["camera_id"] == camera_id)
    if source.get("status") != "saved" or not source.get("file"):
        raise ValueError("Detection has no saved source image")
    return f"images/{source['file']}"


def resolve_alias(identity, aliases):
    if identity is None or int(identity) < 0:
        return None
    identity = int(identity)
    visited = set()
    while identity in aliases or str(identity) in aliases:
        if identity in visited:
            raise ValueError("Cyclic track aliases")
        visited.add(identity)
        identity = int(aliases.get(identity, aliases.get(str(identity))))
    return identity


def source_order(row):
    return (
        int(row["time_ns"]),
        int(row["frame_id"]),
        int(row["camera_id"]),
        row["view_id"],
        int(row["detection_id"].rsplit(":d", 1)[1]),
    )


def box2d_row(frame, observation, index, *, image_size, sem_map, object_id=None):
    x1, x2, y1, y2 = observation["boxes_2d"][index]
    labels, scores = observation["labels_2d"], observation["scores_2d"]
    name = labels[index] if index < len(labels) else "unknown"
    row = {
        "time_ns": str(observation["image_stamp_ns"]),
        "frame_id": frame_id(frame),
        "sensor": f"cam{observation['camera_id']}/{observation['view_id']}",
        "device": "x-nav",
        "img_width": image_size,
        "img_height": image_size,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "name": name,
        "instance": -1,
        "sem_id": sem_map.get(name, -1),
        "prob": scores[index] if index < len(scores) else 0.0,
    }
    if "detection_ids" in observation:
        # The embedded rows and API use exactly the precision of owl_2dbbs.csv.
        for key in ("x1", "y1", "x2", "y2"):
            row[key] = float(f"{row[key]:.2f}")
        row["prob"] = float(f"{row['prob']:.6f}")
        row.update(
            detection_id=observation["detection_ids"][index],
            object_id=object_id,
            camera_id=observation["camera_id"],
            view_id=observation["view_id"],
            image_file=observation["image_file"],
            pose_time_ns=str(frame["time_ns"]),
        )
    return row


def box3d_row(record, time_ns=0, *, aliases=None, confirmed_ids=None):
    tx, ty, tz = record["center"]
    qw, qx, qy, qz = record["quaternion_wxyz"]
    sx, sy, sz = record["size"]
    row = {
        "time_ns": str(record.get("time_ns", time_ns)),
        "tx_world_object": tx,
        "ty_world_object": ty,
        "tz_world_object": tz,
        "qw_world_object": qw,
        "qx_world_object": qx,
        "qy_world_object": qy,
        "qz_world_object": qz,
        "scale_x": sx,
        "scale_y": sy,
        "scale_z": sz,
        "name": record["label"],
        "instance": record["track_id"],
        "sem_id": record["sem_id"],
        "prob": record["prob"],
    }
    if "source_detection_id" in record:
        identity = resolve_alias(
            record.get("matched_track_id", record["track_id"]), aliases or {}
        )
        if confirmed_ids is not None and identity not in confirmed_ids:
            identity = None
        row.update({key: record[key] for key in SOURCE_COLUMNS if key != "object_id"})
        row["object_id"] = identity
        if "bbox3d_id" in record:
            row["bbox3d_id"] = record["bbox3d_id"]
        if "observations_2d" in record:
            row["observations_2d"] = record["observations_2d"]
    return row


def attach_source(record, source, *, frame=None):
    record.update(
        time_ns=source["time_ns"],
        source_detection_id=source["detection_id"],
        image_file=source["image_file"],
        frame_id=frame_id(frame) if frame is not None else source["frame_id"],
        pose_time_ns=str(frame["time_ns"])
        if frame is not None
        else source["pose_time_ns"],
    )


class ProvenanceIndex:
    """Incremental, linear-size index of actual 2D -> fused -> track decisions."""

    def __init__(self, image_size, sem_map):
        self.image_size, self.sem_map = image_size, sem_map
        self.rows = {}
        self.owners = {}
        self.summaries = {}
        self.frame_count = 0

    def add_frame(self, frame, aliases, *, enrich=False):
        current_rows = {}
        for observation in frame["observations"]:
            for i in range(len(observation["boxes_2d"])):
                row = box2d_row(
                    frame,
                    observation,
                    i,
                    image_size=self.image_size,
                    sem_map=self.sem_map,
                )
                identity = row["detection_id"]
                if identity in self.rows:
                    raise ValueError(f"Duplicate detection ID: {identity}")
                self.rows[identity] = current_rows[identity] = row
        # Merge small first/latest summaries rather than copying whole histories
        # into every track snapshot. Alias keys can be strings after JSON replay.
        merged = {}
        for track_id, (first, sequence, latest) in self.summaries.items():
            target = resolve_alias(track_id, aliases)
            if target in merged:
                old_first, old_sequence, old_latest = merged[target]
                first = min((first, old_first), key=source_order)
                if old_sequence > sequence:
                    sequence, latest = old_sequence, old_latest
                elif old_sequence == sequence:
                    latest = min((latest, old_latest), key=source_order)
            merged[target] = first, sequence, latest
        self.summaries = merged
        updates = defaultdict(list)
        for record in frame["detections"]:
            sources = record["source_detection_ids"]
            if not sources or any(s not in current_rows for s in sources):
                raise ValueError("Fused box has missing 2D provenance")
            source = min((current_rows[s] for s in sources), key=source_order)
            if enrich:
                attach_source(record, source, frame=frame)
            owner = record.get("matched_track_id")
            for identity in sources:
                if identity in self.owners:
                    raise ValueError(
                        f"Detection belongs to multiple fused boxes: {identity}"
                    )
                self.owners[identity] = record["bbox3d_id"], owner
            target = resolve_alias(owner, aliases)
            if target is not None:
                updates[target].append(source)
        for target, sources in updates.items():
            latest = min(sources, key=source_order)
            first = latest
            if target in self.summaries:
                first = min((first, self.summaries[target][0]), key=source_order)
            self.summaries[target] = first, self.frame_count, latest
        if enrich:
            for record in frame["tracks"]:
                target = resolve_alias(record["track_id"], aliases)
                if target not in self.summaries:
                    raise ValueError("Track has no associated image detection")
                attach_source(record, self.summaries[target][2], frame=frame)
        self.frame_count += 1

    def object_id(self, identity, aliases, confirmed_ids):
        _, owner = self.owners.get(identity, (None, None))
        target = resolve_alias(owner, aliases)
        return target if target in confirmed_ids else None

    def finalize(self, objects, aliases):
        confirmed_ids = {r["track_id"] for r in objects}
        groups = defaultdict(list)
        for identity, (bbox3d_id, _) in self.owners.items():
            target = self.object_id(identity, aliases, confirmed_ids)
            if target is not None:
                groups[target].append(
                    dict(self.rows[identity], object_id=target, bbox3d_id=bbox3d_id)
                )
        result = []
        for record in objects:
            observations = sorted(groups[record["track_id"]], key=source_order)
            if not observations:
                raise ValueError("Confirmed object has no source detections")
            record = dict(record, observations_2d=observations)
            attach_source(record, observations[0])
            result.append(record)
        return result


def _csv_row(row):
    row = dict(row)
    if "tx_world_object" in row:
        # Retain the existing OBB CSV's label normalization.
        row["name"] = row["name"].replace(",", " ").lower()
    if "x1" in row:
        for key in ("x1", "y1", "x2", "y2"):
            row[key] = f"{row[key]:.2f}"
        row["prob"] = f"{row['prob']:.6f}"
    if "observations_2d" in row:
        row["observations_2d"] = json.dumps(
            row["observations_2d"],
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    return row


class ProvenanceCsvWriter:
    """One CSV implementation for serial execution and the output process."""

    def __init__(self, output, prefix, image_size, sem_map):
        self.output, self.prefix = Path(output), prefix
        self.image_size, self.sem_map = image_size, sem_map
        self.columns = {
            "owl_2dbbs.csv": BOX2D_COLUMNS,
            f"{prefix}_3dbbs.csv": [*BOX3D_COLUMNS, "bbox3d_id", *SOURCE_COLUMNS],
            f"{prefix}_3dbbs_tracks.csv": [*BOX3D_COLUMNS, *SOURCE_COLUMNS],
            f"{prefix}_3dbbs_tracked.csv": [
                *BOX3D_COLUMNS,
                *SOURCE_COLUMNS,
                "observations_2d",
            ],
            f"{prefix}_3dbbs_observations.csv": ASSOCIATION_COLUMNS,
        }
        self.streams, self.writers = {}, {}
        try:
            for name, columns in self.columns.items():
                stream = (self.output / name).open("w", newline="")
                self.streams[name] = stream
                self.writers[name] = csv.DictWriter(
                    stream, columns, lineterminator="\n"
                )
                self.writers[name].writeheader()
        except BaseException:
            self.close()
            raise

    def _write(self, name, row):
        self.writers[name].writerow(_csv_row(row))

    def append(self, frame, aliases):
        confirmed = {
            r["track_id"] for r in frame["tracks"] if r["state"] != "TENTATIVE"
        }
        owners = {
            identity: record
            for record in frame["detections"]
            for identity in record["source_detection_ids"]
        }
        for observation in frame["observations"]:
            for i, identity in enumerate(observation["detection_ids"]):
                owner = owners.get(identity)
                target = (
                    resolve_alias(owner.get("matched_track_id"), aliases)
                    if owner
                    else None
                )
                row = box2d_row(
                    frame,
                    observation,
                    i,
                    image_size=self.image_size,
                    sem_map=self.sem_map,
                    object_id=target if target in confirmed else None,
                )
                self._write("owl_2dbbs.csv", row)
                if owner:
                    self._write(
                        f"{self.prefix}_3dbbs_observations.csv",
                        dict(
                            row,
                            bbox3d_id=owner["bbox3d_id"],
                            track_id=owner.get("matched_track_id"),
                        ),
                    )
        for key, suffix in (("detections", "3dbbs"), ("tracks", "3dbbs_tracks")):
            for record in frame[key]:
                if key == "tracks" and record["state"] == "TENTATIVE":
                    continue
                self._write(
                    f"{self.prefix}_{suffix}.csv",
                    box3d_row(record, aliases=aliases, confirmed_ids=confirmed),
                )
        for stream in self.streams.values():
            stream.flush()

    def finish(self, objects, aliases, map_root):
        images = {o["image_file"] for r in objects for o in r["observations_2d"]}
        for name in images:
            if not (Path(map_root) / name).is_file():
                raise FileNotFoundError(f"Missing associated source image: {name}")
        for record in objects:
            self._write(
                f"{self.prefix}_3dbbs_tracked.csv", box3d_row(record, aliases=aliases)
            )
        self.close()
        detection_objects = {
            o["detection_id"]: r["track_id"]
            for r in objects
            for o in r["observations_2d"]
        }
        confirmed = {r["track_id"] for r in objects}
        # Tentative histories and absorbed IDs become canonical only at finish.
        for name, columns in self.columns.items():
            if name == f"{self.prefix}_3dbbs_tracked.csv":
                continue
            path = self.output / name
            with path.open(newline="") as incoming, _atomic_file(path) as binary:
                outgoing = io.TextIOWrapper(binary, encoding="utf-8", newline="")
                try:
                    writer = csv.DictWriter(outgoing, columns, lineterminator="\n")
                    writer.writeheader()
                    for row in csv.DictReader(incoming):
                        if name == f"{self.prefix}_3dbbs_tracks.csv":
                            target = resolve_alias(row["instance"], aliases)
                            row["object_id"] = target if target in confirmed else None
                        else:
                            identity = row.get(
                                "detection_id", row.get("source_detection_id")
                            )
                            row["object_id"] = detection_objects.get(identity)
                        writer.writerow(row)
                    outgoing.flush()
                finally:
                    outgoing.detach()

    def close(self):
        for stream in self.streams.values():
            stream.close()
