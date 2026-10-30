"""Start a separate desktop viewer of the inference run's committed results."""

import subprocess
import sys
from pathlib import Path


def start_live_viewer(args, output):
    """Closing the viewer leaves inference running; completion leaves it open."""
    script = Path(__file__).resolve().parents[1] / "view_tracker.py"
    command = [
        sys.executable,
        str(script),
        "--input",
        str(Path(args.input).resolve()),
        "--output_dir",
        str(Path(args.output_dir).expanduser().resolve()),
        "--xnav_camera_mode",
        args.xnav_camera_mode,
        "--live",
    ]
    log_path = output / "live_viewer.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(
        f"==> Live 3D viewer started (PID {process.pid}); log: {log_path}", flush=True
    )
    return process
