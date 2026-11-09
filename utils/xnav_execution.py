"""Bounded keyframe inference and ordered, durable CPU output.

Only the inference thread owns the models. Loader state belongs to its producer
thread; the output process receives snapshots containing no CUDA tensors.
"""

import copy
import multiprocessing as mp
import queue
import threading
import time
import traceback
from dataclasses import dataclass

from utils.exception_compat import add_exception_note


@dataclass
class InferredKeyframe:
    index: int
    frame: dict
    prediction: object
    profile: object


@dataclass
class _Failure:
    error: BaseException


_END = object()


class XNavExecution:
    """Yield inference in source order, allowing at most two uncommitted frames."""

    def __init__(
        self, source, infer, profiler, *, prepare=None, cuda=True, on_wait=None
    ):
        self.source = source
        self.infer = infer
        self.prepare = prepare
        self.profiler = profiler
        self.cuda = cuda
        self.on_wait = on_wait
        self._input = queue.Queue(maxsize=1)
        self._output = queue.Queue(maxsize=2)
        self._credits = threading.Semaphore(2)
        self._stop = threading.Event()
        self._threads = []
        self._lock = threading.Lock()
        self.inflight = 0
        self.max_inflight = 0

    def _put(self, target, item):
        while not self._stop.is_set():
            try:
                target.put(item, timeout=0.05)
                return True
            except queue.Full:
                pass
        return False

    def _load(self):
        try:
            iterator = iter(self.source)
            index = 0
            while not self._stop.is_set():
                profile = self.profiler.new_frame()
                with self.profiler.bind(profile):
                    try:
                        with self.profiler.stage("data"):
                            frame = next(iterator)
                            if self.prepare is not None:
                                self.prepare(frame)
                    except StopIteration:
                        break
                if profile is not None:
                    profile["keyframe_id"] = frame["keyframe_id"]
                if not self._put(self._input, (index, frame, profile)):
                    return
                index += 1
            self._put(self._input, _END)
        except BaseException as error:
            self._put(self._input, _Failure(error))

    def _predict(self):
        import contextlib

        import torch

        try:
            stream = torch.cuda.Stream() if self.cuda else None
            if stream is not None:
                stream.wait_stream(self._initial_stream)
            context = (
                torch.cuda.stream(stream)
                if stream is not None
                else contextlib.nullcontext()
            )
            with torch.inference_mode(), context:
                while not self._stop.is_set():
                    credit_started = time.perf_counter() if self.profiler.enabled else 0
                    while not self._stop.is_set() and not self._credits.acquire(
                        timeout=0.05
                    ):
                        pass
                    if self._stop.is_set():
                        return
                    credit_ms = (
                        (time.perf_counter() - credit_started) * 1000
                        if self.profiler.enabled
                        else 0
                    )
                    data_started = time.perf_counter() if self.profiler.enabled else 0
                    while not self._stop.is_set():
                        try:
                            item = self._input.get(timeout=0.05)
                            break
                        except queue.Empty:
                            continue
                    else:
                        return
                    if item is _END:
                        self._put(self._output, _END)
                        return
                    if isinstance(item, _Failure):
                        raise item.error
                    index, frame, profile = item
                    if profile is not None:
                        profile["stages_ms"]["queue.credit"] = credit_ms
                        profile["stages_ms"]["queue.data"] = (
                            time.perf_counter() - data_started
                        ) * 1000
                    with self._lock:
                        self.inflight += 1
                        self.max_inflight = max(self.max_inflight, self.inflight)
                    with self.profiler.bind(profile):
                        prediction = self.infer(frame)
                    if not self._put(
                        self._output,
                        InferredKeyframe(index, frame, prediction, profile),
                    ):
                        return
        except BaseException as error:
            self._put(self._output, _Failure(error))

    def __enter__(self):
        import torch

        self._initial_stream = torch.cuda.current_stream() if self.cuda else None
        self._threads = [
            threading.Thread(target=self._load, name="xnav-loader", daemon=True),
            threading.Thread(target=self._predict, name="xnav-inference", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        return self

    def acknowledge(self):
        with self._lock:
            if self.inflight <= 0:
                raise RuntimeError("Unmatched x-nav output acknowledgement")
            self.inflight -= 1
        self._credits.release()

    def __iter__(self):
        return self

    def __next__(self):
        if self._stop.is_set():
            raise StopIteration
        started = time.perf_counter() if self.profiler.enabled else 0
        while True:
            if self.on_wait is not None:
                self.on_wait()
            try:
                value = self._output.get(timeout=0.05)
            except queue.Empty:
                if self._stop.is_set():
                    raise StopIteration
                continue
            if value is _END:
                raise StopIteration
            if isinstance(value, _Failure):
                raise value.error
            if value.profile is not None:
                value.profile["stages_ms"]["queue.result"] = (
                    time.perf_counter() - started
                ) * 1000
            return value

    def close(self):
        self._stop.set()
        # Unblock a worker waiting for durable output while shutting down.
        self._credits.release()
        self._credits.release()
        for thread in self._threads:
            thread.join(timeout=30)
            if thread.is_alive():
                raise RuntimeError(f"x-nav worker did not stop: {thread.name}")
        self.infer = None
        self.prepare = None
        self.source = ()
        self.on_wait = None

    def __exit__(self, exc_type, error, trace):
        try:
            self.close()
        except Exception as close_error:
            if error is None:
                raise
            add_exception_note(error, f"Could not stop execution worker: {close_error}")


def _output_worker(config, incoming, outgoing):
    """Spawn target: one owner for every result/CSV handle, CPU operations only."""
    from pathlib import Path

    import numpy as np
    import torch

    from loaders.xnav_loader import frame_identity, processing_frame_id
    from utils.file_io import ObbCsvWriter2, save_bb2d_csv
    from utils.tw.camera import CameraTW
    from utils.tw.obb import ObbTW
    from utils.tw.pose import PoseTW
    from utils.xnav_export import save_map_vocabulary
    from utils.xnav_features import DinoFeatureCsvStore
    from utils.xnav_pipeline import _preview, records_to_obbs
    from utils.xnav_result_store import ResultStore

    torch.set_num_threads(1)
    output = Path(config["output"])
    map_root = Path(config["map_root"])
    metadata = config["metadata"]
    prefix = config["prefix"]
    writers = {}
    store = None
    try:
        if not config["no_csv"]:
            for kind, suffix in (
                ("detections", "3dbbs"),
                ("snapshots", "3dbbs_tracks"),
                ("final", "3dbbs_tracked"),
            ):
                writers[kind] = ObbCsvWriter2(str(output / f"{prefix}_{suffix}.csv"))
            save_bb2d_csv(
                str(output / "owl_2dbbs.csv"), 0, np.empty((0, 4)), [], [], append=False
            )
        if not config["skip_viz"]:
            preview = output / "preview_frames"
            preview.mkdir(exist_ok=True)
            for path in preview.glob("frame_*.jpg"):
                path.unlink()
        store = ResultStore(output / "results.json", metadata)
        feature_store = DinoFeatureCsvStore(map_root)
        outgoing.put(("ready", None))
        while True:
            kind, value = incoming.get()
            if kind == "frame":
                started = time.perf_counter()
                record = value["record"]
                feature_store.upsert(value.get("dino_features", []))
                features_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                if writers:
                    for observation in record["observations"]:
                        boxes = np.asarray(
                            observation["boxes_2d"], dtype=np.float32
                        ).reshape(-1, 4)
                        save_bb2d_csv(
                            str(output / "owl_2dbbs.csv"),
                            processing_frame_id(record),
                            boxes[:, [0, 2, 1, 3]],
                            np.asarray(observation["scores_2d"], dtype=np.float32),
                            observation["labels_2d"],
                            sem_name_to_id=config["sem_map"],
                            time_ns=observation["image_stamp_ns"],
                            img_width=config["size"],
                            img_height=config["size"],
                            sensor=f"cam{observation['camera_id']}/{observation['view_id']}",
                            device="x-nav",
                        )
                    writers["detections"].write(
                        ObbTW(torch.from_numpy(value["fused"])), record["time_ns"]
                    )
                    writers["snapshots"].write(
                        ObbTW(torch.from_numpy(value["tracks"])), record["time_ns"]
                    )
                csv_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                if not config["skip_viz"]:
                    frame = {**frame_identity(record), "views": []}
                    payload = value["preview"]
                    if payload is not None:
                        frame["views"] = [
                            {
                                "img0": torch.from_numpy(payload["image"]),
                                "cam0": CameraTW(torch.from_numpy(payload["camera"])),
                                "T_world_rig0": PoseTW(
                                    torch.from_numpy(payload["pose"])
                                ),
                            }
                        ]
                    _preview(
                        frame,
                        ObbTW(torch.from_numpy(value["fused"])),
                        ObbTW(torch.from_numpy(value["tracks"])),
                        output / "preview_frames" / f"frame_{value['index']:05d}.jpg",
                        config["size"],
                    )
                preview_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                store.append(
                    record,
                    value["views"],
                    value["id_aliases"],
                    value["elapsed_seconds"],
                )
                durable_ns = time.perf_counter_ns()
                outgoing.put(
                    (
                        "committed",
                        {
                            "index": value["index"],
                            "durable_ns": durable_ns,
                            "result_bytes": store.result_bytes,
                            "stages": {
                                "features": features_ms,
                                "csv": csv_ms,
                                "preview": preview_ms,
                                "persistence": (time.perf_counter() - started) * 1000,
                            },
                        },
                    )
                )
            elif kind == "finish":
                started = time.perf_counter()
                if writers:
                    writers["final"].write(
                        records_to_obbs(value["objects"]), timestamps_ns=0
                    )
                for writer in writers.values():
                    writer.close()
                stages = {"final.export": (time.perf_counter() - started) * 1000}
                if not config["skip_viz"]:
                    from utils.video import make_mp4

                    started = time.perf_counter()
                    make_mp4(
                        str(output / "preview_frames"),
                        5,
                        output_dir=str(output),
                        image_glob="frame_*.jpg",
                        output_name=f"{prefix}_viz_final.mp4",
                    )
                    stages["final.video"] = (time.perf_counter() - started) * 1000
                # Keep the durable prefix available on any export failure.
                # Publish complete only after every requested output succeeds.
                started = time.perf_counter()
                labels = value.get("parameters", {}).get("labels")
                if labels is not None:
                    save_map_vocabulary(map_root, labels)
                store.finish(value)
                stages["final.export"] += (time.perf_counter() - started) * 1000
                outgoing.put(("finished", stages))
                return
            elif kind == "fail":
                store.fail(value)
                outgoing.put(("failed", None))
                return
            else:
                raise ValueError(f"Unknown x-nav output request: {kind}")
    except BaseException as error:
        detail = traceback.format_exc()
        if store is not None:
            try:
                store.fail(error)
            except BaseException as failure:
                add_exception_note(
                    error, f"Could not publish failure status: {failure}"
                )
        outgoing.put(("error", (error, detail)))
    finally:
        if store is not None:
            store.close()
        for writer in writers.values():
            writer.close()


class XNavOutput:
    """Receive durable ACKs independently; dispatch public callbacks on the caller.

    Only the receiver reads the process channel. Its internal ``on_durable``
    callback may release inference credits, but must not touch main-thread state.
    The bounded local channel preserves commit/terminal order and backpressure.
    """

    def __init__(
        self, config, *, on_commit=None, on_durable=None, target=_output_worker
    ):
        context = mp.get_context("spawn")
        self._incoming = context.Queue(maxsize=2)
        self._outgoing = context.Queue(maxsize=4)
        self._process = context.Process(
            target=target,
            args=(copy.deepcopy(config), self._incoming, self._outgoing),
            name="xnav-output",
            daemon=True,
        )
        self.on_commit = on_commit
        self.on_durable = on_durable
        self._messages = queue.Queue(maxsize=4)
        self._receiver_stop = threading.Event()
        self._pending = set()
        self._pending_lock = threading.Lock()
        self._last_durable = -1
        self._terminal = False
        self._closed = False
        self.final_stages = {}
        self._process.start()
        self._receiver = threading.Thread(
            target=self._receive_messages, name="xnav-output-ack", daemon=True
        )
        self._receiver.start()
        try:
            self._wait_for("ready", 60)
        except BaseException:
            self.close()
            raise

    def _post(self, message):
        while not self._receiver_stop.is_set():
            try:
                self._messages.put(message, timeout=0.05)
                return
            except queue.Full:
                pass

    def _receive_messages(self):
        try:
            while not self._receiver_stop.is_set():
                try:
                    kind, value = self._outgoing.get(timeout=0.05)
                except queue.Empty:
                    if self._process.exitcode is not None:
                        raise RuntimeError(
                            "x-nav output worker exited unexpectedly "
                            f"({self._process.exitcode})"
                        )
                    continue
                if kind == "committed":
                    value["ack_received_ns"] = time.perf_counter_ns()
                    index = value["index"]
                    with self._pending_lock:
                        if index <= self._last_durable:
                            # Replayed ACKs must never return another credit.
                            continue
                        if (
                            index != self._last_durable + 1
                            or index not in self._pending
                        ):
                            raise RuntimeError(f"Unexpected x-nav durable ACK: {index}")
                        self._pending.remove(index)
                        self._last_durable = index
                    if self.on_durable is not None:
                        self.on_durable(value)
                if kind in ("finished", "failed"):
                    with self._pending_lock:
                        if self._pending:
                            raise RuntimeError(
                                "x-nav output closed with unacknowledged frames"
                            )
                self._post((kind, value))
                if kind in ("finished", "failed", "error"):
                    return
        except BaseException as error:
            self._post(("error", (error, traceback.format_exc())))

    def _receive(self, timeout=0):
        try:
            kind, value = self._messages.get(timeout=timeout)
        except queue.Empty:
            return None
        if kind == "error":
            self._terminal = True
            error, detail = value
            add_exception_note(error, "x-nav output worker traceback:\n" + detail)
            raise error
        if kind == "committed" and self.on_commit is not None:
            value["ack_processed_ns"] = time.perf_counter_ns()
            self.on_commit(value)
        if kind in ("finished", "failed"):
            self._terminal = True
        if kind == "finished":
            self.final_stages = value
        return kind

    def poll(self):
        while self._receive() is not None:
            pass

    def _wait_for(self, wanted, timeout=None):
        started = time.monotonic()
        while True:
            if self._receive(timeout=0.05) == wanted:
                return
            if self._terminal:
                raise RuntimeError(f"x-nav output worker closed before {wanted}")
            if timeout is not None and time.monotonic() - started > timeout:
                raise TimeoutError(f"x-nav output worker did not send {wanted}")

    def _send(self, kind, value):
        if self._terminal:
            raise RuntimeError("x-nav output worker is closed")
        while True:
            self.poll()
            if self._terminal or self._closed:
                raise RuntimeError("x-nav output worker is closed")
            try:
                self._incoming.put((kind, value), timeout=0.05)
                return
            except queue.Full:
                continue

    def submit(self, snapshot):
        snapshot = copy.deepcopy(snapshot)
        index = snapshot["index"]
        with self._pending_lock:
            if index in self._pending or index <= self._last_durable:
                raise ValueError(f"Duplicate x-nav output index: {index}")
            self._pending.add(index)
        try:
            self._send("frame", snapshot)
        except BaseException:
            with self._pending_lock:
                self._pending.discard(index)
            raise

    def finish(self, metadata):
        self._send("finish", copy.deepcopy(metadata))
        self._wait_for("finished")

    def fail(self, error):
        if not self._terminal:
            self._send("fail", error)
            self._wait_for("failed", 30)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._receiver_stop.set()
        self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._receiver.join(timeout=5)
        if self._receiver.is_alive():
            raise RuntimeError("x-nav output ACK receiver did not stop")
        self.on_commit = None
        self.on_durable = None
        # An unread feeder pipe must never block interpreter shutdown.
        for channel in (self._incoming, self._outgoing):
            channel.cancel_join_thread()
            channel.close()
