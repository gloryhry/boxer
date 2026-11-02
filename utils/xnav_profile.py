"""Opt-in per-frame host timings and asynchronous CUDA event timings."""

import contextlib
import json
import resource
import threading
import time
from pathlib import Path

import numpy as np
import torch


def _summary(rows):
    return {
        name: {
            "count": len(values),
            "total_ms": sum(values),
            "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
        }
        for name, values in rows.items()
        if values
    }


class XNavProfiler:
    """A disabled profiler never queries, records or synchronizes CUDA."""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.started = time.perf_counter() if enabled else 0
        self.cuda = False
        self.frames = []
        self.other_stages = {}
        self._local = threading.local()
        self._lock = threading.Lock()
        self._events = []
        self._next_index = 0

    @property
    def current(self):
        return getattr(self._local, "frame", None)

    @current.setter
    def current(self, value):
        self._local.frame = value

    def set_device(self, device):
        if self.enabled and device == "cuda":
            self.cuda = True
            torch.cuda.reset_peak_memory_stats()

    def new_frame(self):
        if not self.enabled:
            return None
        with self._lock:
            index = self._next_index
            self._next_index += 1
        return {"index": index, "stages_ms": {}, "_started": time.perf_counter()}

    @contextlib.contextmanager
    def bind(self, frame):
        previous = self.current
        self.current = frame
        try:
            yield
        finally:
            self.current = previous

    @contextlib.contextmanager
    def stage(self, name, cuda=False):
        if not self.enabled:
            yield
            return
        frame = self.current
        events = None
        if cuda and self.cuda:
            events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            events[0].record()
        started = time.perf_counter()
        try:
            yield
        finally:
            if events is not None:
                events[1].record()
            elapsed = (time.perf_counter() - started) * 1000
            with self._lock:
                if frame is not None:
                    stages = frame["stages_ms"]
                    stages[name] = stages.get(name, 0) + elapsed
                else:
                    self.other_stages.setdefault(name, []).append(elapsed)
                if events is not None:
                    self._events.append((frame, name, events))

    def finish_frame(self, frame, stages=None, result_bytes=None, commit_lag=None):
        if not self.enabled or frame is None:
            return
        with self._lock:
            if "wall_ms" in frame:
                return
            for name, value in (stages or {}).items():
                frame["stages_ms"][name] = frame["stages_ms"].get(name, 0) + value
            now = time.perf_counter()
            frame["wall_ms"] = (now - frame.pop("_started")) * 1000
            frame.setdefault("committed_seconds", now - self.started)
            if "_computed_at" in frame:
                frame["commit_latency_ms"] = (now - frame.pop("_computed_at")) * 1000
            if result_bytes is not None:
                frame["result_bytes"] = result_bytes
            if commit_lag is not None:
                frame["commit_lag"] = commit_lag
            self.frames.append(frame)

    def computation_finished(self):
        if self.enabled and self.current is not None:
            self.current["_computed_at"] = time.perf_counter()

    def committed(self):
        """Serial commits happen before preview; pipeline commits arrive by ACK."""
        if self.enabled and self.current is not None:
            now = time.perf_counter()
            self.current["committed_seconds"] = now - self.started
            if "_computed_at" in self.current:
                self.current["commit_latency_ms"] = (
                    now - self.current.pop("_computed_at")
                ) * 1000

    def iterate(self, source):
        if not self.enabled:
            yield from source
            return
        iterator = iter(source)
        while True:
            frame = self.new_frame()
            with self.bind(frame):
                try:
                    with self.stage("data"):
                        value = next(iterator)
                except StopIteration:
                    return
                frame["keyframe_id"] = value["keyframe_id"]
                try:
                    yield value
                finally:
                    self.finish_frame(frame)

    def record_frame(self, tracks, result_bytes):
        if self.enabled and self.current is not None:
            self.current.update(track_count=tracks, result_bytes=result_bytes)

    def _resolve_events(self):
        # Called after the execution workers stop, never at a hot-path boundary.
        events, self._events = self._events, []
        for frame, name, (start, end) in events:
            end.synchronize()
            if frame is not None:
                stages = frame.setdefault("cuda_stages_ms", {})
                stages[name] = stages.get(name, 0) + start.elapsed_time(end)

    def report(self, status):
        if self.enabled:
            self._resolve_events()
        frames = sorted(self.frames, key=lambda row: row["index"])
        steady = frames[2:]
        groups, gpu_groups = {}, {}
        for row in steady:
            for name, value in row["stages_ms"].items():
                groups.setdefault(name, []).append(value)
            for name, value in row.get("cuda_stages_ms", {}).items():
                gpu_groups.setdefault(name, []).append(value)
        segments = {}
        for name, rows in zip(("early", "middle", "late"), np.array_split(steady, 3)):
            segments[name] = _summary({"frame": [r["wall_ms"] for r in rows]})
        committed_span = (
            frames[-1]["committed_seconds"] - frames[1]["committed_seconds"]
            if len(frames) > 2
            else 0
        )
        return {
            "status": status,
            "wall_seconds": time.perf_counter() - self.started,
            "warmup_frames": 2,
            "frames": frames,
            "steady_stages": _summary(groups),
            "steady_cuda_stages": _summary(gpu_groups),
            "steady_frames": _summary({"frame": [r["wall_ms"] for r in steady]}),
            "steady_commit_latency": _summary(
                {
                    "commit": [
                        r["commit_latency_ms"]
                        for r in steady
                        if "commit_latency_ms" in r
                    ]
                }
            ),
            "steady_throughput_fps": len(steady) / committed_span
            if committed_span > 0
            else None,
            "segments": segments,
            "other_stages": self.other_stages and _summary(self.other_stages),
            "cpu_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
            if self.cuda
            else None,
            "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20
            if self.cuda
            else None,
        }

    def write(self, path, status):
        if not self.enabled:
            return
        path = Path(path)
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(self.report(status), indent=2) + "\n")
        temp.replace(path)
