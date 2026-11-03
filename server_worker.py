"""Resident model owner; its private socket never shares the inference log."""

import contextlib
import gc
import json
import multiprocessing
import os
import signal
import socket
import sys
import threading
import traceback


@contextlib.contextmanager
def job_log(path):
    """Redirect native/subprocess output as well as Python logging per job."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(fd) for fd in (1, 2)]
    try:
        with open(path, "ab", buffering=0) as log:
            for fd in (1, 2):
                os.dup2(log.fileno(), fd)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        for fd, backup in zip((1, 2), saved):
            os.dup2(backup, fd)
            os.close(backup)


def serve(fd, *, runtime_factory=None, run_inference=None, owner_alive=lambda: True):
    from utils.xnav_runtime import XNavRuntime, inference_args

    if run_inference is None:
        from utils.xnav_pipeline import run_xnav

        run_inference = run_xnav
    runtime = (runtime_factory or XNavRuntime)()
    interruptible = False
    previous_handler = signal.getsignal(signal.SIGINT)

    def interrupt(sig, frame):
        if interruptible or not owner_alive():
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)
    with (
        socket.socket(fileno=fd) as channel,
        channel.makefile("r", encoding="utf-8") as incoming,
    ):

        def emit(**message):
            channel.sendall((json.dumps(message, ensure_ascii=False) + "\n").encode())

        try:
            while owner_alive():
                try:
                    line = incoming.readline()
                    if not line:
                        break
                    message = json.loads(line)
                    if message["op"] == "shutdown":
                        break
                    if message["op"] == "configure":
                        try:
                            interruptible = True
                            runtime.configure(inference_args(message["options"]))
                        except Exception as error:
                            traceback.print_exc()
                            emit(event="configure_failed", error=str(error))
                        else:
                            emit(event="configured", device=runtime.device)
                        finally:
                            interruptible = False
                    elif message["op"] == "run":
                        identity = message["job_id"]
                        status, error, reusable = "complete", None, True
                        try:
                            interruptible = True
                            with job_log(message["log_path"]):
                                emit(event="job_started", job_id=identity)
                                try:
                                    run_inference(
                                        inference_args(
                                            message["options"],
                                            input=message["input"],
                                            output_dir=message["output_dir"],
                                        ),
                                        runtime=runtime,
                                    )
                                except BaseException:
                                    traceback.print_exc()
                                    raise
                        except KeyboardInterrupt:
                            status = "interrupted"
                        except Exception as failure:
                            status, error, reusable = "failed", str(failure), False
                        finally:
                            # The server still honors stop_requested after this
                            # point. Do not lose job_finished to a late SIGINT.
                            interruptible = False
                        # A timed-out execution thread must never share its model
                        # with the next task, even after run_xnav returned an error.
                        reusable = (
                            reusable
                            and not any(
                                t.is_alive()
                                and t.name in ("xnav-loader", "xnav-inference")
                                for t in threading.enumerate()
                            )
                            # torch.compile can own persistent compiler children.
                            # Only the per-task output process must have exited.
                            and not any(
                                child.name == "xnav-output"
                                for child in multiprocessing.active_children()
                            )
                        )
                        gc.collect()
                        emit(
                            event="job_finished",
                            job_id=identity,
                            status=status,
                            error=error,
                            reusable=reusable,
                        )
                        if not reusable:
                            break
                    else:
                        raise ValueError(f"Unknown worker operation: {message['op']}")
                except KeyboardInterrupt:
                    # A stop can race with completion; it must not kill an idle
                    # model owner. The server defers early stops until job_started.
                    continue
        finally:
            try:
                runtime.close()
            finally:
                signal.signal(signal.SIGINT, previous_handler)


def main(*, runtime_factory=None, run_inference=None):
    owner = int(sys.argv.pop(1))
    if os.getppid() != owner:
        raise SystemExit("The HTTP service exited before inference started")
    done = threading.Event()

    def watch_owner():
        while not done.wait(1):
            if os.getppid() != owner:
                os.kill(os.getpid(), signal.SIGINT)
                if not done.wait(60):
                    os.killpg(os.getpgrp(), signal.SIGKILL)
                return

    threading.Thread(target=watch_owner, name="http-owner-watch", daemon=True).start()
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "--serve":
            serve(
                int(sys.argv[2]),
                runtime_factory=runtime_factory,
                run_inference=run_inference,
                owner_alive=lambda: os.getppid() == owner,
            )
        else:
            # Preserve the internal wrapper's CLI/help compatibility.
            from run_boxer import main as run_cli

            run_cli()
    finally:
        done.set()


if __name__ == "__main__":
    main()
