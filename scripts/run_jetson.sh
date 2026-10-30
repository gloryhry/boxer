#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="$repo_root/.runtime"

# JetPack wheels can run CUDA eagerly without an ARM Triton installation.
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$runtime_root/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$runtime_root/cache/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$runtime_root/cache/cuda}"
export TMPDIR="${TMPDIR:-$runtime_root/tmp}"
export PATH="$runtime_root/bin:$PATH"
export LD_LIBRARY_PATH="$runtime_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec bash "$repo_root/scripts/run_local_cuda.sh" "$@"
