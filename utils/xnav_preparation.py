"""Bounded CPU preparation with unchanged camera arithmetic and point order."""

import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np


class XNavPreparation:
    CHUNK_SIZE = 65536

    def __init__(self, workers=1, profiler=None):
        if workers not in (1, 2):
            raise ValueError("x-nav preparation workers must be 1 or 2")
        self.workers = workers
        self.profiler = profiler
        self._pool = None
        self._closed = False
        self._local = threading.local()

    def _executor(self):
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self.workers, thread_name_prefix="xnav-prepare"
            )
        return self._pool

    def map(self, function, values):
        if self._closed:
            raise RuntimeError("x-nav preparation is closed")
        if self.workers == 1 or len(values) <= 1:
            return [function(value) for value in values]
        frame = self.profiler.current if self.profiler is not None else None

        def call(value):
            # Camera jobs and large inverse-projection chunks share the same
            # two threads. A job must never wait on its own executor.
            self._local.active = True
            context = (
                self.profiler.bind(frame)
                if self.profiler is not None
                else contextlib.nullcontext()
            )
            try:
                with context:
                    return function(value)
            finally:
                self._local.active = False

        return list(self._executor().map(call, values))

    def stage(self, name):
        return (
            self.profiler.stage(name)
            if self.profiler is not None
            else contextlib.nullcontext()
        )

    def unproject(self, camera, pixels):
        if self._closed:
            raise RuntimeError("x-nav preparation is closed")
        with self.stage("data.unproject"):
            pixels = np.asarray(pixels, dtype=np.float64)
            shape = pixels.shape[:-1]
            flat = pixels.reshape(-1, 2)
            if (
                self.workers == 1
                or len(flat) <= self.CHUNK_SIZE
                or getattr(self._local, "active", False)
            ):
                return camera.unproject(pixels)
            # map preserves input order. Each chunk uses the original NumPy
            # implementation, including all branch and convergence checks.
            chunks = (
                flat[offset : offset + self.CHUNK_SIZE]
                for offset in range(0, len(flat), self.CHUNK_SIZE)
            )
            values = list(self._executor().map(camera.unproject, chunks))
            return (
                np.concatenate([value[0] for value in values]).reshape(*shape, 3),
                np.concatenate([value[1] for value in values]).reshape(shape),
            )

    def close(self):
        self._closed = True
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
