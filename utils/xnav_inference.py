"""CUDA input ownership and explicit, grouped CPU readback dependencies."""

import contextlib
from dataclasses import dataclass

import torch


class CpuReadback:
    """Keep both ends alive until one event makes all CPU values readable."""

    def __init__(self, values):
        self.sources = tuple(values)
        self.event = None
        if not any(value.is_cuda for value in self.sources):
            self.values = tuple(value.cpu() for value in self.sources)
            return
        self.values = tuple(
            torch.empty_like(value, device="cpu", pin_memory=True)
            for value in self.sources
        )
        for target, source in zip(self.values, self.sources):
            target.copy_(source, non_blocking=True)
        self.event = torch.cuda.Event()
        self.event.record()

    def wait(self):
        if self.event is not None:
            self.event.synchronize()
            self.event = None
        return self.values


@dataclass
class DeviceView:
    inputs: dict
    owl_pixels: object
    ready: object

    def wait(self):
        stream = torch.cuda.current_stream()
        stream.wait_event(self.ready)
        for value in (*self.inputs.values(), self.owl_pixels):
            tensor = (
                value
                if isinstance(value, torch.Tensor)
                else getattr(value, "_data", None)
            )
            if tensor is not None and tensor.is_cuda:
                tensor.record_stream(stream)
        return self


def _unique_views(frame):
    seen = set()
    for datum in (*frame.get("native_images", []), *frame["views"]):
        if id(datum) not in seen:
            seen.add(id(datum))
            yield datum


class XNavDeviceInputs:
    """One copy stream; buffers belong to their frame, never to reusable slots."""

    def __init__(self, profiler):
        self.profiler = profiler
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        self._pending = []

    def pin(self, frame):
        with self.profiler.stage("transfer.pin"):
            for datum in _unique_views(frame):
                for key in ("img0", "cam0", "T_world_rig0", "sdp_w", "_owl_pixels"):
                    value = datum.get(key)
                    if value is None:
                        continue
                    tensor = value if isinstance(value, torch.Tensor) else value._data
                    if not tensor.is_pinned():
                        pinned = tensor.pin_memory()
                        datum[key] = (
                            pinned
                            if isinstance(value, torch.Tensor)
                            else type(value)(pinned)
                        )

    @contextlib.contextmanager
    def stage(self, frame, boxer):
        # A slow consumer cannot retain an unbounded list of pinned frames.
        self._pending = [row for row in self._pending if not row[0].query()]
        if len(self._pending) >= 2:
            self._pending.pop(0)[0].synchronize()
        entries = {}
        try:
            with (
                torch.cuda.stream(self.stream),
                self.profiler.stage("transfer.h2d", cuda=True),
            ):
                for datum in _unique_views(frame):
                    inputs = (
                        boxer.prepare_inputs(
                            datum, non_blocking=True, include_boxes=False
                        )
                        if "cam0" in datum
                        else {
                            "img0": datum["img0"].to("cuda", non_blocking=True).float()
                        }
                    )
                    inputs["_profiler"] = self.profiler
                    pixels = datum.get("_owl_pixels")
                    if pixels is not None:
                        pixels = pixels.to("cuda", non_blocking=True)
                    ready = torch.cuda.Event()
                    ready.record()
                    entries[id(datum)] = DeviceView(inputs, pixels, ready)
                finished = torch.cuda.Event()
                finished.record()
                self._pending.append((finished, frame))
            yield entries
        finally:
            # Also cover inputs unused by cache replay or by an empty detector.
            # record_stream protects allocations after the frame is released.
            for entry in entries.values():
                entry.wait()

    def close(self):
        self.stream.synchronize()
        self._pending.clear()
