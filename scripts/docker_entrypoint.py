#!/usr/bin/env python3
"""Validate bind mounts and CUDA before replacing this process with the server."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKPOINTS = (
    "boxernet_hw960in2x6d768-c88128f8.ckpt",
    "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "owlv2-base-patch16-ensemble.pt",
)


def writable_directory(path, *, create=False):
    path = Path(path)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError(f"Required directory is missing: {path}")
    try:
        with tempfile.TemporaryFile(prefix=".boxer-write-check-", dir=path):
            pass
    except OSError as error:
        raise ValueError(f"Directory must be writable: {path}: {error}") from error


def check_mounts(repo, map_dir, output_dir):
    """Probe actual writes, including read-only mounts when running as root."""
    repo = Path(repo)
    for name in CHECKPOINTS:
        checkpoint = repo / "ckpts" / name
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise ValueError(f"Required model is missing or empty: {checkpoint}")
        with checkpoint.open("rb") as stream:
            stream.read(1)

    labels_dir = Path(os.environ.get("BOXER_LABELS_DIR") or repo / "owl")
    vocabulary = labels_dir / "indoor_classes.csv"
    if not vocabulary.is_file():
        raise ValueError(f"Default vocabulary is missing: {vocabulary}")
    if not any(line.strip() for line in vocabulary.read_text().splitlines()):
        raise ValueError(f"Default vocabulary is empty: {vocabulary}")

    writable_directory(repo / "ckpts")
    writable_directory(map_dir)
    writable_directory(output_dir, create=True)


def main(argv=None):
    command = list(sys.argv[1:] if argv is None else argv)
    if not command or command[0].startswith("-"):
        command = [sys.executable, str(REPO / "server.py"), *command]

    # Let explicit diagnostic commands and --help run without models or a GPU.
    is_server = len(command) > 1 and Path(command[1]).resolve() == REPO / "server.py"
    if is_server and not {"-h", "--help"}.intersection(command[2:]):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--output_dir", default=str(REPO / "output" / "server"))
        options, _ = parser.parse_known_args(command[2:])
        try:
            check_mounts(REPO, Path("/nav_map"), Path(options.output_dir))
            for variable in (
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "CUDA_CACHE_PATH",
                "XDG_CACHE_HOME",
                "TMPDIR",
            ):
                if os.environ.get(variable):
                    writable_directory(os.environ[variable], create=True)
            print("Boxer mounts verified; checking CUDA kernels...", flush=True)
            subprocess.run(
                [sys.executable, str(REPO / "scripts" / "check_cuda.py")],
                check=True,
            )
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            print(f"Boxer startup failed: {error}", file=sys.stderr, flush=True)
            return 1

    os.execvp(command[0], command)


if __name__ == "__main__":
    sys.exit(main())
