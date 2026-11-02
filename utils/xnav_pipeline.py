"""Keyframe-wise Boxer inference, persistent identities and snapshot replay."""

import contextlib
import copy
import hashlib
import io
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from loaders.xnav_loader import XNavMapLoader, frame_identity, processing_frame_id
from utils.exception_compat import add_exception_note
from utils.file_io import ObbCsvWriter2, save_bb2d_csv
from utils.fuse_3d_boxes import BoundingBox3DFuser
from utils.image import draw_bb3s, put_text
from utils.taxonomy import load_text_labels
from utils.track_3d_boxes import BoundingBox3DTracker, ViewObservation
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW
from utils.tw.tensor_utils import pad_string, string2tensor
from utils.tw.xnav_camera import PREPROCESS_VERSION
from utils.xnav_execution import InferredKeyframe, XNavExecution, XNavOutput
from utils.xnav_profile import XNavProfiler
from utils.xnav_result_store import ResultFollower, ResultStore, read_result_file


def empty_obbs():
    return ObbTW(torch.empty((0, 165)))


def _stack(boxes):
    return torch.stack(boxes) if boxes else empty_obbs()


def box_record(obb, track=None):
    record = {
        "center": obb.T_world_object.t.reshape(3).tolist(),
        "quaternion_wxyz": obb.T_world_object.q.reshape(4).tolist(),
        "size": obb.bb3_diagonal.reshape(3).tolist(),
        "label": obb.reshape(1, -1).text_string()[0],
        "sem_id": int(obb.sem_id.item()),
        "prob": float(obb.prob.item()),
        "track_id": int(obb.inst_id.item()) if track is None else track.track_id,
    }
    if track is not None:
        record.update(
            state=track.state.name,
            support_count=track.support_count,
            first_seen_frame=track.first_seen_frame,
            last_seen_frame=track.last_seen_frame,
            missed_count=track.missed_count,
            visible=track.last_visible,
        )
    return record


def box_records(boxes, tracks=None):
    """Extract OBB fields in batches while retaining scalar record semantics."""
    if not len(boxes):
        return []
    if not isinstance(boxes, ObbTW):
        boxes = torch.stack([box.reshape(-1) for box in boxes])
    poses = boxes.T_world_object
    centers = poses.t.reshape(-1, 3).tolist()
    quaternions = poses.q.reshape(-1, 4).tolist()
    sizes = boxes.bb3_diagonal.reshape(-1, 3).tolist()
    # Match text_string's int32 conversion, invalid-codepoint filtering and
    # rstrip, without extracting 128 separate Torch scalars for every label.
    labels = [
        "".join(chr(c) for c in row if 0 <= c <= 0x10FFFF).rstrip()
        for row in boxes.text.int().tolist()
    ]
    sem_ids = boxes.sem_id.reshape(-1).tolist()
    probabilities = boxes.prob.reshape(-1).tolist()
    identities = boxes.inst_id.reshape(-1).tolist()
    records = []
    for i in range(len(boxes)):
        track = tracks[i] if tracks is not None else None
        record = {
            "center": centers[i],
            "quaternion_wxyz": quaternions[i],
            "size": sizes[i],
            "label": labels[i],
            "sem_id": int(sem_ids[i]),
            "prob": float(probabilities[i]),
            "track_id": int(identities[i]) if track is None else track.track_id,
        }
        if track is not None:
            record.update(
                state=track.state.name,
                support_count=track.support_count,
                first_seen_frame=track.first_seen_frame,
                last_seen_frame=track.last_seen_frame,
                missed_count=track.missed_count,
                visible=track.last_visible,
            )
        records.append(record)
    return records


def records_to_obbs(records, confirmed_only=False):
    boxes = []
    for record in records:
        if confirmed_only and record.get("state") == "TENTATIVE":
            continue
        size = torch.tensor(record["size"], dtype=torch.float32)
        bb = torch.stack(
            [
                -size[0] / 2,
                size[0] / 2,
                -size[1] / 2,
                size[1] / 2,
                -size[2] / 2,
                size[2] / 2,
            ]
        )
        pose = PoseTW.from_qt(
            torch.tensor(record["quaternion_wxyz"]), torch.tensor(record["center"])
        )
        boxes.append(
            ObbTW.from_lmc(
                bb3_object=bb,
                T_world_object=pose,
                prob=torch.tensor([record["prob"]]),
                sem_id=torch.tensor([record["sem_id"]]),
                inst_id=torch.tensor([record["track_id"]]),
                text=string2tensor(
                    pad_string(record["label"], max_len=128, silent=True)
                ),
            ).reshape(-1)
        )
    return _stack(boxes)


def save_results(path, results):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(results, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    temp.replace(path)


def source_signature(loader):
    digest = hashlib.sha256()
    for name in (
        "manifest.txt",
        "poses_base.txt",
        "poses_image_base.txt",
        "camera_calibration.yaml",
        "images/index.csv",
        "content_hashes.txt",
        "config/map_config.yaml",
    ):
        path = loader.root / name
        if path.is_file():
            digest.update(name.encode() + path.read_bytes())
    for folder, pattern in (("images", "*.jpg"), ("rgb", "*.pcd"), ("surf", "*.pcd")):
        for path in sorted((loader.root / folder).glob(pattern)):
            stat = path.stat()
            digest.update(
                f"{path.relative_to(loader.root)}:{stat.st_size}:{stat.st_mtime_ns}".encode()
            )
    return digest.hexdigest()


def fuse_keyframe(detections, embeddings=None, force_cpu=False):
    if len(detections) < 2:
        return detections.clone()
    fuser = BoundingBox3DFuser(
        iou_threshold=0.3,
        min_detections=1,
        semantic_threshold=0.7,
        conf_threshold=0,
        force_cpu=force_cpu,
    )
    if embeddings is None:
        labels = detections.text_string()
        names = sorted(set(labels))
        embeddings = torch.eye(len(names))[[names.index(label) for label in labels]]
    # The original fuser prints a performance report for every call.
    with contextlib.redirect_stdout(io.StringIO()):
        instances = fuser.fuse(detections, semantic_embeddings=embeddings)
    fused = []
    for inst in instances:
        # Resolve ties deterministically and keep the semantic id consistent
        # with the selected text (the legacy fuser chooses them separately).
        members = detections[inst.detection_indices]
        names = members.text_string()
        counts = Counter(names)
        scores = members.prob.reshape(-1).tolist()
        selected = max(
            sorted(counts),
            key=lambda s: (counts[s], sum(p for n, p in zip(names, scores) if n == s)),
        )
        source = next(i for i, name in enumerate(names) if name == selected)
        box = inst.obb.cpu().reshape(-1)
        box.set_text(string2tensor(pad_string(selected, max_len=128, silent=True)))
        box.set_sem_id(members[source].sem_id)
        fused.append(box)
    boxes = _stack(fused)
    if len(boxes):
        boxes.set_inst_id(torch.full((len(boxes),), -1, dtype=torch.int32))
    return boxes


def _valid_boxes(boxes):
    if not len(boxes):
        return boxes
    ok = (
        torch.isfinite(boxes.T_world_object._data).all(-1)
        & torch.isfinite(boxes.bb3_diagonal).all(-1)
        & (boxes.bb3_diagonal > 0).all(-1)
        & torch.isfinite(boxes.prob).reshape(-1)
    )
    return boxes[ok]


def read_results(path, camera_mode=None):
    return _validate_results(read_result_file(path), camera_mode)


def _validate_results(results, camera_mode=None):
    if results.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported x-nav results schema")
    parameters = results["parameters"]
    if results["schema_version"] == 1:
        parameters.update(
            camera_mode="virtual_pinhole",
            preprocess_version=1,
            hemisphere="per_view",
        )
    saved_mode = parameters.get("camera_mode")
    if saved_mode not in ("native", "virtual_pinhole"):
        raise ValueError("Saved results have no valid x-nav camera mode")
    if parameters.get("preprocess_version") != PREPROCESS_VERSION:
        raise ValueError(
            "Saved results use an incompatible x-nav preprocessing version"
        )
    expected_hemisphere = "front" if saved_mode == "native" else "per_view"
    if parameters.get("hemisphere") != expected_hemisphere:
        raise ValueError("Saved results use an incompatible x-nav hemisphere policy")
    if camera_mode is not None and parameters.get("camera_mode") != camera_mode:
        raise ValueError("Saved results use a different x-nav camera mode")
    return results


class XNavReplay:
    """Random access to saved state; seeking never executes a tracker."""

    def __init__(self, path, camera_mode=None, live=False):
        self._follower = ResultFollower(path) if live else None
        self._camera_mode = camera_mode
        if live:
            self.refresh()
        else:
            self.results = read_results(path, camera_mode)
            self.frames = self.results["frames"]
        if not self.frames and not live:
            raise ValueError("No x-nav result frames to replay")

    def refresh(self):
        """Append newly committed frames without running detection or tracking."""
        if self._follower is None or not self._follower.poll():
            return False
        results = _validate_results(self._follower.results, self._camera_mode)
        if hasattr(self, "results") and results.get(
            "source_signature"
        ) != self.results.get("source_signature"):
            raise ValueError("Source map changed while following inference")
        self.results = results
        self.frames = self.results["frames"]
        return True

    def frame(self, index):
        return self.frames[index]

    def boxes(self, index, tracked=True, confirmed_only=True):
        return records_to_obbs(
            self.frames[index]["tracks" if tracked else "detections"],
            confirmed_only=confirmed_only if tracked else False,
        )


def _preview(frame, detections, tracks, path, size=960):
    if not frame["views"]:
        panel = np.zeros((size, size * 2, 3), dtype=np.uint8)
        put_text(
            panel, f"Frame {processing_frame_id(frame)}: no image observation", line=0
        )
        if not cv2.imwrite(str(path), panel):
            raise OSError(f"Could not write x-nav preview: {path}")
        return
    datum = frame["views"][0]
    image = (datum["img0"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    panels = []
    for title, boxes, labels in (
        ("3D detections", detections, detections.text_string()),
        (
            "Static tracks",
            tracks,
            [
                f"#{int(o.inst_id.item())} {s}"
                for o, s in zip(tracks, tracks.text_string())
            ],
        ),
    ):
        viz = draw_bb3s(
            image.copy(), datum["T_world_rig0"], datum["cam0"], boxes, texts=labels
        )
        put_text(viz, f"{title} | frame {processing_frame_id(frame)}", line=0)
        panels.append(viz)
    if not cv2.imwrite(str(path), np.hstack(panels)):
        raise OSError(f"Could not write x-nav preview: {path}")


def _infer_keyframe(
    frame,
    args,
    labels,
    sem_map,
    owl,
    boxer,
    previous,
    cached_frames,
    profiler,
    dtype,
    detector_state,
    device,
):
    """Model-only work, independent of the ordered tracker and output owner."""
    key = processing_frame_id(frame)
    frame_record = {
        **frame_identity(frame),
        "time_ns": frame["time_ns"],
        "T_map_base": frame["T_map_base"].tolist(),
        "sources": frame["sources"],
        "observations": [],
        "detections": [],
        "tracks": [],
    }
    raw = []
    views = {}
    cached_views = {
        (o["camera_id"], o["view_id"]): o
        for o in cached_frames.get(key, {}).get("observations", [])
    }
    detected = None
    if not previous and detector_state["effective"] == 2:
        detected = []
        for offset in range(0, len(frame["views"]), 2):
            group = frame["views"][offset : offset + 2]
            images = torch.cat([d["img0"] for d in group]) * 255
            prepared = [d.get("_owl_pixels") for d in group]
            prepared = (
                torch.cat(prepared) if all(x is not None for x in prepared) else None
            )
            values = None
            clear_failed_batch = False
            if detector_state["effective"] == 2:
                try:
                    with profiler.stage("detector", cuda=True):
                        values = owl.forward_batch(images, prepared=prepared)
                except torch.cuda.OutOfMemoryError:
                    detector_state["effective"] = 1
                    detector_state["oom_fallbacks"] += 1
                    clear_failed_batch = True
            if values is None:
                # Leave the exception scope before retrying so its traceback
                # cannot retain the failed batch's CUDA intermediates.
                if clear_failed_batch:
                    torch.cuda.empty_cache()
                values = []
                for datum in group:
                    with profiler.stage("detector", cuda=True):
                        values.append(
                            owl.forward(
                                datum["img0"] * 255, prepared=datum.get("_owl_pixels")
                            )
                        )
            detected.extend(values)
    for view_index, datum in enumerate(frame["views"]):
        cid, view_id = datum["camera_id"], datum["view_id"]
        rect = datum["view_geometry"]
        view_key = f"cam{cid}/{view_id}"
        views[view_key] = {
            "camera_id": cid,
            "view_id": view_id,
            **rect.metadata(),
        }
        if previous:
            if (cid, view_id) not in cached_views:
                raise ValueError(f"Missing cached observation {(key, cid, view_id)}")
            cached = cached_views[cid, view_id]
            bb2 = torch.tensor(cached["boxes_2d"], dtype=torch.float32).reshape(-1, 4)
            scores = torch.tensor(cached["scores_2d"], dtype=torch.float32)
            names = cached["labels_2d"]
        else:
            if detected is not None:
                bb2, scores, ids, _ = detected[view_index]
            else:
                keywords = {}
                if "_owl_pixels" in datum:
                    keywords["prepared"] = datum["_owl_pixels"]
                with profiler.stage("detector", cuda=True):
                    bb2, scores, ids, _ = owl.forward(
                        datum["img0"] * 255,
                        resize_to_HW=(args.detector_hw, args.detector_hw),
                        **keywords,
                    )
            names = [labels[int(i)] for i in ids]
            bb2, scores = bb2.cpu(), scores.cpu()
            keep = torch.from_numpy(rect.box_is_valid(bb2.numpy()))
            bb2, scores = bb2[keep], scores[keep]
            names = [name for name, ok in zip(names, keep) if ok]
        observation = {
            **frame_identity(frame),
            "camera_id": cid,
            "view_id": view_id,
            "image_stamp_ns": datum["time_ns0"],
            "T_map_view": datum["T_world_rig0"].matrix.tolist(),
            "boxes_2d": bb2.tolist(),
            "scores_2d": scores.tolist(),
            "labels_2d": names,
            "detections": [],
        }
        if args.cache3d:
            boxes = records_to_obbs(cached["detections"])
        elif len(bb2):
            datum["bb2d"] = bb2
            autocast = (
                torch.autocast(device_type=device, dtype=dtype)
                if device == "cuda"
                else contextlib.nullcontext()
            )
            with profiler.stage("boxer", cuda=True), autocast:
                predictions = boxer.forward(datum)
            boxes = predictions["obbs_pr_w"].cpu()[0]
            keep = (boxes.prob >= args.thresh3d).reshape(-1)
            boxes = boxes[keep].clone()
            selected = [name for name, ok in zip(names, keep) if ok]
            if len(boxes):
                boxes.set_prob((boxes.prob.reshape(-1) + scores[keep]) / 2)
                boxes.set_sem_id(
                    torch.tensor(
                        [sem_map[name] for name in selected], dtype=torch.int32
                    )
                )
                boxes.set_text(
                    torch.stack(
                        [
                            string2tensor(pad_string(name, max_len=128, silent=True))
                            for name in selected
                        ]
                    )
                )
                boxes.set_inst_id(torch.full((len(boxes),), -1, dtype=torch.int32))
            boxes = _valid_boxes(boxes)
        else:
            boxes = empty_obbs()
        with profiler.stage("records"):
            observation["detections"] = box_records(boxes)
        raw.extend(list(boxes))
        frame_record["observations"].append(observation)
    return frame_record, raw, views


@torch.inference_mode()
def run_xnav(args):
    profiler = XNavProfiler(getattr(args, "profile", False))
    if getattr(args, "pinhole", False):
        raise ValueError(
            "For x-nav use --xnav_camera_mode virtual_pinhole instead of --pinhole"
        )
    if not 0 <= args.thresh2d <= 1 or not 0 <= args.thresh3d <= 1:
        raise ValueError("Confidence thresholds must be between zero and one")
    with profiler.stage("init.loader"):
        loader = XNavMapLoader(
            args.input,
            cameras=args.xnav_cameras,
            camera_mode=getattr(args, "xnav_camera_mode", "native"),
            start_n=args.start_n,
            skip_n=args.skip_n,
            max_n=args.max_n,
        )
    with profiler.stage("init.validation"):
        validation = loader.validate()
    print(
        "==> x-nav input validation:",
        json.dumps(validation, ensure_ascii=False),
        flush=True,
    )
    if args.validate_input:
        return
    if getattr(args, "fuse", False):
        raise ValueError(
            "x-nav fuses keyframes automatically; use --track for global fusion"
        )
    if args.gt2d:
        raise ValueError("x-nav maps contain no ground-truth object boxes; omit --gt2d")
    labels = load_text_labels(args.labels)
    if not labels:
        raise ValueError("At least one text label is required")
    output = loader.output_path(args.output_dir)
    result_path = output / "results.json"
    with profiler.stage("init.signature"):
        signature = source_signature(loader)
    parameters = {
        "camera_mode": loader.camera_mode,
        "preprocess_version": PREPROCESS_VERSION,
        "hemisphere": "front" if loader.camera_mode == "native" else "per_view",
        "labels": labels,
        "thresh2d": args.thresh2d,
        "thresh3d": args.thresh3d,
        "detector_hw": args.detector_hw,
        "image_size": loader.size,
        "fov_degrees": loader.fov,
        "max_points": loader.max_points,
        "no_sdp": args.no_sdp,
        "checkpoint": str(Path(args.ckpt).resolve()),
        "camera_ids": loader.camera_ids,
    }
    previous = None
    if args.cache2d or args.cache3d:
        with profiler.stage("init.cache"):
            previous = read_results(
                loader.result_file(args.output_dir), loader.camera_mode
            )
        loader.size = previous.get("parameters", {}).get("image_size", loader.size)
        parameters["image_size"] = loader.size
        if (
            previous.get("source_signature") != signature
            or previous.get("parameters") != parameters
        ):
            raise ValueError(
                "Cached results do not match source map or inference parameters"
            )
    cached_frames = (
        {processing_frame_id(f): f for f in previous["frames"]} if previous else {}
    )
    if previous and not set(loader.frame_ids) <= cached_frames.keys():
        raise ValueError("Cache does not cover selected frames")

    device = (
        "cpu"
        if args.force_cpu
        else (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    parallel = device == "cuda" and getattr(args, "xnav_execution", "auto") != "serial"
    detector_state = {
        "requested": getattr(args, "xnav_detector_batch_size", 1),
        "effective": getattr(args, "xnav_detector_batch_size", 1) if parallel else 1,
        "oom_fallbacks": 0,
    }
    owl = boxer = None
    profiler.set_device(device)
    embeddings = {}
    if not args.cache3d:
        from boxernet.boxernet import BoxerNet

        with profiler.stage("init.boxer", cuda=True):
            boxer = BoxerNet.load_from_checkpoint(args.ckpt, device=device)
        if previous and boxer.hw != loader.size:
            raise ValueError(
                "Cached rectification does not match Boxer checkpoint image size"
            )
        loader.size = boxer.hw
        parameters["image_size"] = loader.size
        if not args.cache2d:
            from owl.owl_wrapper import OwlWrapper

            with profiler.stage("init.owl", cuda=True):
                owl = OwlWrapper(
                    device,
                    text_prompts=labels,
                    min_confidence=args.thresh2d,
                    precision=args.force_precision,
                    warmup=False,
                )
            values = owl.text_embeddings.detach().float().cpu().reshape(len(labels), -1)
            embeddings = {
                label.lower(): torch.nn.functional.normalize(values[i], dim=0)
                for i, label in enumerate(labels)
            }
    if previous:
        embeddings = {
            k: torch.tensor(v)
            for k, v in previous.get("semantic_embeddings", {}).items()
        }
    min_hits = args.track_min_hits if args.track_min_hits is not None else 3
    if min_hits < 1:
        raise ValueError("--track_min_hits must be positive")
    tracker = (
        BoundingBox3DTracker(
            min_hits=min_hits,
            min_confidence_mass=float("inf"),
            # The two detector thresholds have already been applied. Scores
            # stored below are their mean, which can be below thresh3d.
            conf_threshold=0,
            max_missed=90,
            force_cpu=args.force_cpu,
            verbose=False,
            merge_semantic_threshold=0.7 if embeddings else 1.0,
            merge_iou_threshold=0,
        )
        if args.track
        else None
    )
    # Multi-view merges use the same semantic dictionary as the detector.
    if tracker:
        tracker.merge_iou_threshold = 0.5
        tracker._embedding_cache.update(embeddings)
        tracker.xnav_accelerated = parallel
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "schema_version": 2,
        "status": "running",
        "map_root": str(loader.root),
        "manifest": loader.manifest,
        "frame_unit": loader.frame_unit,
        "source_signature": signature,
        "coordinate_frame": "map",
        "unit": "metre",
        "up_axis": "+Z",
        "device": device,
        "parameters": parameters,
        "execution": {
            "mode": "pipeline" if parallel else "serial",
            "detector_batch_size": detector_state["effective"],
            "boxer_batch_size": 1,
            "max_uncommitted_frames": 2 if parallel else 1,
        },
        "tracking": {
            "enabled": args.track,
            "min_hits": min_hits,
            "iou_threshold": 0.25,
            "max_missed": 90,
        },
        "calibration": loader.calibration,
        "views": {},
        "frames": [],
        "semantic_embeddings": {k: v.tolist() for k, v in embeddings.items()},
        "validation": validation,
        "id_aliases": {},
    }
    prefix = args.write_name
    writers = {}
    if not parallel and not args.no_csv:
        for kind, suffix in (
            ("detections", "3dbbs"),
            ("snapshots", "3dbbs_tracks"),
            ("final", "3dbbs_tracked"),
        ):
            writers[kind] = ObbCsvWriter2(str(output / f"{prefix}_{suffix}.csv"))
        # Always create a header, even when every source frame has zero detections.
        save_bb2d_csv(
            str(output / "owl_2dbbs.csv"), 0, np.empty((0, 4)), [], [], append=False
        )
    if not parallel and not args.skip_viz:
        preview_dir = output / "preview_frames"
        preview_dir.mkdir(exist_ok=True)
        # A shorter rerun must not encode stale frames from the previous run.
        for path in preview_dir.glob("frame_*.jpg"):
            path.unlink()
    dtype = (
        torch.bfloat16
        if (
            args.force_precision == "bfloat16"
            or (
                args.force_precision is None
                and device == "cuda"
                and torch.cuda.is_bf16_supported()
            )
        )
        else torch.float32
    )
    sem_map = {label: i for i, label in enumerate(labels)}
    observed_index, start = 0, time.perf_counter()
    store = None
    counts = Counter()
    processed_anchors = set()
    frame_record = None
    live_viewer = None
    execution = output_worker = source = None
    profiles = {}
    tracking_stream = torch.cuda.Stream() if parallel else None
    if tracking_stream is not None:
        tracking_stream.wait_stream(torch.cuda.current_stream())

    def infer(frame):
        return _infer_keyframe(
            frame,
            args,
            labels,
            sem_map,
            owl,
            boxer,
            previous,
            cached_frames,
            profiler,
            dtype,
            detector_state,
            device,
        )

    def prepare(frame):
        if owl is not None:
            for datum in frame["views"]:
                datum["_owl_pixels"] = owl.prepare_image(datum["img0"] * 255)

    def committed(ack):
        profile = profiles.pop(ack["index"])
        profiler.finish_frame(
            profile, ack["stages"], ack["result_bytes"], execution.inflight
        )
        execution.acknowledge()

    def serial_source():
        for index, frame in enumerate(
            profiler.iterate(loader.iter_frames(with_points=not args.no_sdp))
        ):
            yield InferredKeyframe(index, frame, infer(frame), profiler.current)

    try:
        with profiler.stage("init.output"):
            if parallel:
                output_worker = XNavOutput(
                    {
                        "output": str(output),
                        "metadata": results,
                        "prefix": prefix,
                        "no_csv": args.no_csv,
                        "skip_viz": args.skip_viz,
                        "sem_map": sem_map,
                        "size": loader.size,
                    },
                    on_commit=committed,
                )
                execution = XNavExecution(
                    loader.iter_frames(with_points=not args.no_sdp),
                    infer,
                    profiler,
                    prepare=prepare,
                    on_wait=output_worker.poll,
                )
                execution.__enter__()
                source = execution
            else:
                store = ResultStore(result_path, results)
                source = serial_source()
        if getattr(args, "live_viz", False):
            from utils.xnav_live import start_live_viewer

            live_viewer = start_live_viewer(args, output)
        for work in tqdm(source, total=len(loader), desc=f"x-nav {loader.frame_unit} frames"):
            with (
                profiler.bind(work.profile),
                (
                    torch.cuda.stream(tracking_stream)
                    if parallel
                    else contextlib.nullcontext()
                ),
            ):
                frame = work.frame
                frame_record, raw, view_metadata = work.prediction
                results["views"].update(view_metadata)
                boxes = _stack(raw)
                embed = (
                    torch.stack([embeddings[s.lower()] for s in boxes.text_string()])
                    if len(boxes) and embeddings
                    else None
                )
                with profiler.stage("fusion", cuda=True):
                    fused = fuse_keyframe(boxes, embed, args.force_cpu)
                with profiler.stage("records"):
                    frame_record["detections"] = box_records(fused)
                if tracker:
                    observations = [
                        ViewObservation(
                            d["cam0"],
                            d["T_world_rig0"],
                            None if args.no_sdp else d["sdp_w"],
                            torch.from_numpy(d["view_geometry"].valid_mask),
                        )
                        for d in frame["views"]
                    ]
                    with profiler.stage("tracking", cuda=True):
                        tracker.update(fused, observed_index, observations=observations)
                    if observations:
                        observed_index += 1
                    with profiler.stage("records"):
                        all_tracks = tracker.get_all_tracks()
                        frame_record["tracks"] = box_records(
                            [t.obb for t in all_tracks], all_tracks
                        )
                    results["id_aliases"] = dict(tracker.id_aliases)
                with profiler.stage("records"):
                    tracks = records_to_obbs(
                        frame_record["tracks"], confirmed_only=True
                    )
                if writers:
                    for observation in frame_record["observations"]:
                        bb2 = np.asarray(
                            observation["boxes_2d"], dtype=np.float32
                        ).reshape(-1, 4)
                        save_bb2d_csv(
                            str(output / "owl_2dbbs.csv"),
                            processing_frame_id(frame),
                            bb2[:, [0, 2, 1, 3]],
                            np.asarray(observation["scores_2d"], dtype=np.float32),
                            observation["labels_2d"],
                            sem_name_to_id=sem_map,
                            time_ns=observation["image_stamp_ns"],
                            img_width=loader.size,
                            img_height=loader.size,
                            sensor=f"cam{observation['camera_id']}/{observation['view_id']}",
                            device="x-nav",
                        )
                    writers["detections"].write(fused, frame["time_ns"])
                    writers["snapshots"].write(tracks, frame["time_ns"])
                results["elapsed_seconds"] = time.perf_counter() - start
                profiler.computation_finished()
                if parallel:
                    profiles[work.index] = work.profile
                    preview = None
                    if not args.skip_viz and frame["views"]:
                        datum = frame["views"][0]
                        preview = {
                            "image": datum["img0"].numpy().copy(),
                            "camera": datum["cam0"]._data.numpy().copy(),
                            "pose": datum["T_world_rig0"]._data.numpy().copy(),
                        }
                    with profiler.stage("queue.output"):
                        output_worker.submit(
                            {
                                "index": work.index,
                                "record": frame_record,
                                "views": copy.deepcopy(results["views"]),
                                "id_aliases": dict(results["id_aliases"]),
                                "elapsed_seconds": results["elapsed_seconds"],
                                "fused": fused._data.numpy().copy(),
                                "tracks": tracks._data.numpy().copy(),
                                "preview": preview,
                            }
                        )
                else:
                    with profiler.stage("persistence"):
                        store.append(
                            frame_record,
                            results["views"],
                            results["id_aliases"],
                            results["elapsed_seconds"],
                        )
                    profiler.committed()
                counts.update(
                    processed_frames=1,
                    image_frames=int(loader.frame_unit == "image" or any(
                        s["status"] == "saved" for s in frame["sources"]
                    )),
                    source_images=sum(s["status"] == "saved" for s in frame["sources"]),
                    view_observations=len(frame_record["observations"]),
                    detections_2d=sum(
                        len(o["boxes_2d"]) for o in frame_record["observations"]
                    ),
                    detections_3d=len(frame_record["detections"]),
                )
                processed_anchors.add(frame_record["lidar_keyframe_id"])
                counts["keyframes"] = len(processed_anchors)
                if profiler.enabled:
                    profiler.record_frame(
                        len(frame_record["tracks"]),
                        0 if parallel else store.result_bytes,
                    )
                if not parallel and not args.skip_viz:
                    with profiler.stage("preview"):
                        _preview(
                            frame,
                            fused,
                            tracks,
                            output
                            / "preview_frames"
                            / f"frame_{store.committed_frames - 1:05d}.jpg",
                            loader.size,
                        )
                if live_viewer is not None and live_viewer.poll() is not None:
                    if live_viewer.returncode:
                        print(
                            "==> Live viewer exited with an error; see",
                            output / "live_viewer.log",
                            flush=True,
                        )
                    live_viewer = None
                if parallel:
                    output_worker.poll()
        final = frame_record["tracks"] if tracker and frame_record else []
        results["objects"] = [o for o in final if o["state"] != "TENTATIVE"]
        if writers:
            writers["final"].write(records_to_obbs(results["objects"]), timestamps_ns=0)
        results["status"] = "complete"
        results["statistics"] = {**counts, "confirmed_objects": len(results["objects"])}
        results["execution"].update(
            detector_batch_size=detector_state["effective"],
            detector_oom_fallbacks=detector_state["oom_fallbacks"],
        )
        if parallel:
            execution.close()
            results["execution"]["max_inflight_frames"] = execution.max_inflight
            geometry = tracker._xnav_geometry if tracker is not None else None
            results["execution"]["geometry_cpu_repairs"] = (
                geometry.repaired_boxes if geometry is not None else 0
            )
            output_worker.finish(results)
            if profiler.enabled:
                for name, value in output_worker.final_stages.items():
                    profiler.other_stages.setdefault(name, []).append(value)
        else:
            for writer in writers.values():
                writer.close()
            if not args.skip_viz:
                from utils.video import make_mp4

                with profiler.stage("final.video"):
                    make_mp4(
                        str(output / "preview_frames"),
                        5,
                        output_dir=str(output),
                        image_glob="frame_*.jpg",
                        output_name=f"{prefix}_viz_final.mp4",
                    )
            with profiler.stage("final.export"):
                store.finish(results)
    except BaseException as error:
        results["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        if execution is not None:
            try:
                execution.close()
            except Exception as close_error:
                add_exception_note(
                    error, f"Could not stop inference worker: {close_error}"
                )
        if output_worker is not None:
            try:
                output_worker.fail(error)
            except BaseException as save_error:
                add_exception_note(error, f"Could not stop output worker: {save_error}")
        if store is not None:
            try:
                store.fail(error)
            except BaseException as save_error:
                add_exception_note(
                    error, f"Could not publish failure status: {save_error}"
                )
        raise
    finally:
        import sys

        pending_error = sys.exc_info()[1]
        for worker in (execution, output_worker, source if not parallel else None):
            if worker is not None:
                try:
                    worker.close()
                except Exception as close_error:
                    if pending_error is None:
                        raise
                    add_exception_note(
                        pending_error,
                        f"Could not close execution worker: {close_error}",
                    )
        if store is not None:
            try:
                store.close()
            except Exception as close_error:
                if pending_error is None:
                    raise
                add_exception_note(
                    pending_error, f"Could not close result journal: {close_error}"
                )
        for writer in writers.values():
            try:
                writer.close()
            except Exception as close_error:
                if pending_error is None:
                    raise
                add_exception_note(
                    pending_error, f"Could not close CSV writer: {close_error}"
                )
        if results["status"] != "complete":
            try:
                profiler.write(output / "performance.json", results["status"])
            except Exception as profile_error:
                if pending_error is None:
                    raise
                add_exception_note(
                    pending_error, f"Could not write timings: {profile_error}"
                )
    profiler.write(output / "performance.json", results["status"])
    print("==> x-nav results:", result_path, flush=True)
    print("==>", json.dumps(results["statistics"]), flush=True)
