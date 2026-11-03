"""Private socket control for the server's single, process-isolated model owner."""

import contextlib
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path


class InferenceWorker:
    def __init__(self, root, *, command=None):
        repo = Path(__file__).resolve().parents[1]
        parent, child = socket.socketpair()
        self.socket = parent
        self.messages = queue.Queue()
        self.process = None
        self.reader = None
        try:
            command = command or [sys.executable, "-u", str(repo / "server_worker.py")]
            with (Path(root) / "worker.log").open("ab") as log:
                self.process = subprocess.Popen(
                    [*command, str(os.getpid()), "--serve", str(child.fileno())],
                    cwd=repo,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(child.fileno(),),
                    start_new_session=True,
                )
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        self.reader = threading.Thread(
            target=self._receive, name="inference-control", daemon=True
        )
        self.reader.start()

    def _receive(self):
        try:
            with self.socket.makefile("r", encoding="utf-8") as stream:
                for line in stream:
                    self.messages.put(json.loads(line))
        except (OSError, ValueError) as error:
            self.messages.put({"event": "control_error", "error": str(error)})

    def send(self, message):
        self.socket.sendall((json.dumps(message, ensure_ascii=False) + "\n").encode())

    def events(self):
        if self.process.poll() is not None:
            self.reader.join(timeout=1)
        while True:
            try:
                yield self.messages.get_nowait()
            except queue.Empty:
                return

    def signal_group(self, sig):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.process.pid, sig)

    def close(self):
        if self.process.poll() is None:
            with contextlib.suppress(OSError):
                self.send({"op": "shutdown"})
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.signal_group(signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.signal_group(signal.SIGKILL)
                    self.process.wait(timeout=5)
        # Reap output children left behind by a failed model owner.
        self.signal_group(signal.SIGKILL)
        with contextlib.suppress(OSError):
            self.socket.shutdown(socket.SHUT_RDWR)
        self.socket.close()
        self.reader.join(timeout=1)
