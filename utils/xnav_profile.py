"""Opt-in wall-clock profiling for the x-nav inference pipeline."""

import contextlib
import json
import resource
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
    """Disabled profiling never synchronizes or queries CUDA memory."""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.started = time.perf_counter() if enabled else 0
        self.cuda = False
        self.frames = []
        self.current = None
        self.other_stages = {}

    def set_device(self, device):
        if self.enabled and device == "cuda":
            self.cuda = True
            torch.cuda.reset_peak_memory_stats()

    @contextlib.contextmanager
    def stage(self, name, cuda=False):
        if not self.enabled:
            yield
            return
        if cuda and self.cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            if cuda and self.cuda:
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            if self.current is not None:
                stages = self.current["stages_ms"]
                stages[name] = stages.get(name, 0) + elapsed
            else:
                self.other_stages.setdefault(name, []).append(elapsed)

    def iterate(self, source):
        if not self.enabled:
            yield from source
            return
        iterator = iter(source)
        while True:
            self.current = {"index": len(self.frames), "stages_ms": {}}
            started = time.perf_counter()
            try:
                with self.stage("data"):
                    frame = next(iterator)
            except StopIteration:
                self.current = None
                return
            self.current["keyframe_id"] = frame["keyframe_id"]
            try:
                yield frame
            finally:
                self.current["wall_ms"] = (time.perf_counter() - started) * 1000
                self.frames.append(self.current)
                self.current = None

    def record_frame(self, tracks, result_bytes):
        if self.enabled and self.current is not None:
            self.current.update(track_count=tracks, result_bytes=result_bytes)

    def report(self, status):
        steady = self.frames[2:]
        groups = {}
        for row in steady:
            for name, value in row["stages_ms"].items():
                groups.setdefault(name, []).append(value)
        segments = {}
        for name, rows in zip(("early", "middle", "late"), np.array_split(steady, 3)):
            segments[name] = _summary({"frame": [r["wall_ms"] for r in rows]})
        return {
            "status": status,
            "wall_seconds": time.perf_counter() - self.started,
            "warmup_frames": 2,
            "frames": self.frames,
            "steady_stages": _summary(groups),
            "steady_frames": _summary({"frame": [r["wall_ms"] for r in steady]}),
            "segments": segments,
            "other_stages": _summary(self.other_stages),
            "cpu_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "cuda_peak_allocated_mib": (
                torch.cuda.max_memory_allocated() / 2**20 if self.cuda else None
            ),
            "cuda_peak_reserved_mib": (
                torch.cuda.max_memory_reserved() / 2**20 if self.cuda else None
            ),
        }

    def write(self, path, status):
        if not self.enabled:
            return
        path = Path(path)
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(self.report(status), indent=2) + "\n")
        temp.replace(path)
