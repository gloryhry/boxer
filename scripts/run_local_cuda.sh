#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"
python="$repo_root/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    echo "Missing .venv. See docs/LOCAL_CUDA.md." >&2
    exit 1
fi

runtime_root="$(dirname -- "$(readlink -f -- "$repo_root/.venv")")"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$runtime_root/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$runtime_root/cache/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$runtime_root/cache/cuda}"
export TMPDIR="${TMPDIR:-$runtime_root/tmp}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
mkdir -p -- "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"

entrypoint="run_boxer.py"
case "${1:-}" in
    run_boxer.py|view_prompt.py|view_fusion.py|view_tracker.py)
        entrypoint="$1"
        shift
        ;;
esac
for argument in "$@"; do
    if [[ "$argument" == "--force_cpu" ]]; then
        echo "This launcher requires CUDA; omit --force_cpu." >&2
        exit 2
    fi
done

"$python" "$repo_root/scripts/check_cuda.py"
exec "$python" "$repo_root/$entrypoint" "$@"
