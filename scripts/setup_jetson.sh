#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"
if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "This environment is for Jetson ARM64 / JetPack 6.2 / CUDA 12.6." >&2
    exit 1
fi
python3 -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 is required"'

runtime_root="$repo_root/.runtime"
mkdir -p -- "$runtime_root/bin" "$runtime_root/lib" "$runtime_root/tmp" \
    "$runtime_root/downloads" "$runtime_root/cudss"
export TMPDIR="$runtime_root/tmp"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check \
    --index-url https://pypi.org/simple 'uv==0.13.6'
.venv/bin/uv pip install --python .venv/bin/python \
    --index-url https://pypi.org/simple -r requirements-jetson.lock.txt
.venv/bin/uv pip check --python .venv/bin/python

# Use Jetson's system cuBLAS. The cuDSS pip package pulls newer CUDA libraries
# that PyTorch can preload ahead of the JetPack versions.
cudss_archive="$runtime_root/downloads/libcudss-linux-aarch64-0.7.1.4_cuda12-archive.tar.xz"
cudss_sha256="f283c31b6badf4a5277014295705eac8e6e28f27de30d746719ea1cef7a750b8"
if ! printf '%s  %s\n' "$cudss_sha256" "$cudss_archive" | sha256sum --check --status 2>/dev/null; then
    curl --fail --location --retry 3 \
        'https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-aarch64/libcudss-linux-aarch64-0.7.1.4_cuda12-archive.tar.xz' \
        --output "$cudss_archive"
    printf '%s  %s\n' "$cudss_sha256" "$cudss_archive" | sha256sum --check
fi
tar -xJf "$cudss_archive" --strip-components=1 --directory "$runtime_root/cudss"
for library in "$runtime_root"/cudss/lib/*.so*; do
    ln -sfn -- "$library" "$runtime_root/lib/$(basename -- "$library")"
done

ffmpeg_binary="$(.venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
ln -sfn -- "$ffmpeg_binary" "$runtime_root/bin/ffmpeg"

LD_LIBRARY_PATH="$runtime_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    .venv/bin/python scripts/check_cuda.py
