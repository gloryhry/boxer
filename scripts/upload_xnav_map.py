#!/usr/bin/env python3
"""Upload one ZIP using durable chunks; rerun the same command to resume."""

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(base, path="", *, method="GET", data=None, headers=None):
    headers = dict(headers or {})
    if isinstance(data, dict):
        data = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    for attempt in range(4):
        try:
            query = urllib.request.Request(
                base + "/api/maps/uploads" + path,
                method=method,
                data=data,
                headers=headers,
            )
            with urllib.request.urlopen(query, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code < 500 and error.code != 429 or attempt == 3:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(2**attempt)


def save(path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:18765")
    parser.add_argument(
        "--state", type=Path, help="Local resume record (default: ZIP.upload.json)"
    )
    parser.add_argument("--new", action="store_true", help="Create a separate upload")
    args = parser.parse_args(argv)
    base = args.url.rstrip("/")
    state = args.state or args.zip.with_name(args.zip.name + ".upload.json")
    try:
        size, digest = args.zip.stat().st_size, hashlib.sha256()
        print("Hashing ZIP...", flush=True)
        with args.zip.open("rb") as stream:
            while chunk := stream.read(8 * 2**20):
                digest.update(chunk)
        checksum = digest.hexdigest()
        upload = None
        if state.exists() and not args.new:
            previous = json.loads(state.read_text())
            if (
                previous["url"] != base
                or previous["sha256"] != checksum
                or previous["size"] != size
            ):
                raise ValueError(
                    "Resume record does not match the ZIP/server; "
                    "use --new for a separate upload"
                )
            try:
                upload = request(base, f"/{previous['id']}")
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
                print("Previous upload expired; creating a new session.", flush=True)
        if upload is None:
            upload = request(
                base,
                method="POST",
                data={
                    "filename": args.zip.name,
                    "size": size,
                    "sha256": checksum,
                },
            )
            save(
                state,
                {"url": base, "id": upload["id"], "size": size, "sha256": checksum},
            )
        identity = upload["id"]
        print(f"upload_id={identity}", flush=True)
        if upload["status"] not in ("complete", "processing"):
            received = set(upload["received_chunks"])
            with args.zip.open("rb") as stream:
                for index in range(upload["total_chunks"]):
                    if index in received:
                        continue
                    stream.seek(index * upload["chunk_size"])
                    data = stream.read(upload["chunk_size"])
                    upload = request(
                        base,
                        f"/{identity}/chunks/{index}",
                        method="PUT",
                        data=data,
                        headers={
                            "Content-Type": "application/octet-stream",
                            "X-Chunk-SHA256": hashlib.sha256(data).hexdigest(),
                        },
                    )
                    print(
                        f"Saved {upload['received_bytes']} / {size} bytes", flush=True
                    )
            upload = request(base, f"/{identity}/complete", method="POST")
        while upload["status"] == "processing":
            print(f"{upload['stage']}: {upload['progress']:.0%}", flush=True)
            time.sleep(1)
            upload = request(base, f"/{identity}")
        if upload["status"] != "complete":
            raise ValueError(
                upload["error"] or f"Upload is {upload['status']}; rerun to resume"
            )
        print(json.dumps(upload["map"], ensure_ascii=False, indent=2))
        return 0
    except urllib.error.HTTPError as error:
        print(f"HTTP {error.code}: {error.read().decode(errors='replace')}")
    except (OSError, ValueError, KeyError) as error:
        print(str(error))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
