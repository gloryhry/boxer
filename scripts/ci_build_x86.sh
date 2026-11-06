#!/usr/bin/env bash
# Do not trace registry credentials even if the invoking shell enables tracing.
set +x
set -euo pipefail

# Also guard direct invocations; only push pipelines may publish images.
if [[ "${CI_PIPELINE_SOURCE:-}" != "push" ]] ||
    [[ "${CI_COMMIT_BRANCH:-}" != "master" && -z "${CI_COMMIT_TAG:-}" ]]; then
    echo "Skipping publication: expected a master branch push or a Git tag push."
    exit 0
fi

fail() {
    echo "Boxer CI: $*" >&2
    exit 1
}

for variable in CI_REGISTRY CI_REGISTRY_IMAGE CI_REGISTRY_USER CI_REGISTRY_PASSWORD \
    CI_COMMIT_SHA CI_JOB_ID; do
    [[ -n "${!variable:-}" ]] || fail "Missing required variable: $variable"
done
[[ "$CI_COMMIT_SHA" =~ ^[0-9a-f]{40}$ ]] || fail "Expected a full Git commit SHA."
[[ "$CI_JOB_ID" =~ ^[0-9]+$ ]] || fail "Expected a numeric CI_JOB_ID."
[[ "$(uname -s)/$(uname -m)" == "Linux/x86_64" ]] ||
    fail "The Shell Runner must run on Linux x86_64."
command -v docker >/dev/null || fail "Install Docker Engine and the Buildx plugin."

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
version_tag="${CI_COMMIT_TAG:-}"
docker_tag_pattern='^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$'
if [[ -n "$version_tag" && ! "$version_tag" =~ $docker_tag_pattern ]]; then
    version_tag="tag-${CI_COMMIT_REF_SLUG:-}-${CI_COMMIT_SHORT_SHA:-${CI_COMMIT_SHA:0:8}}"
    [[ "$version_tag" =~ $docker_tag_pattern ]] || fail "Invalid fallback image tag."
    printf 'Git tag %s maps to image tag %s.\n' "$CI_COMMIT_TAG" "$version_tag"
fi

# A separate config keeps job credentials out of the checkout and the Runner's
# persistent Docker login. The default builder still uses the daemon's cache.
umask 077
docker_auth_dir="$(mktemp -d "${TMPDIR:-/tmp}/boxer-ci-docker.XXXXXXXX")"
export DOCKER_CONFIG="$docker_auth_dir"
local_image="boxer-ci-x64:job-${CI_JOB_ID}-${docker_auth_dir##*.}"
smoke_container="boxer-ci-x64-${CI_JOB_ID}-${docker_auth_dir##*.}"

cleanup() {
    task_status=$?
    trap - EXIT
    docker container rm --force "$smoke_container" >/dev/null 2>&1 || true
    docker image rm "$local_image" >/dev/null 2>&1 || true
    rm -rf -- "$docker_auth_dir"
    exit "$task_status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

docker buildx version
daemon_platform="$(docker info --format '{{.OSType}}/{{.Architecture}}')"
case "$daemon_platform" in
    linux/amd64|linux/x86_64) ;;
    *) fail "Docker daemon must be Linux x64; found $daemon_platform." ;;
esac
printf '%s' "$CI_REGISTRY_PASSWORD" |
    docker login "$CI_REGISTRY" --username "$CI_REGISTRY_USER" --password-stdin

docker buildx build --builder default --platform linux/amd64 --pull --load \
    --progress plain --file "$repo_root/Dockerfile.x86" \
    --tag "$local_image" "$repo_root"

image_platform="$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$local_image")"
[[ "$image_platform" == "linux/amd64" ]] || fail "Unexpected image platform: $image_platform"

# Override the service entrypoint: imports and bundled assets need neither a
# GPU nor mounted model weights. Disable networking to catch hidden downloads.
docker run --rm --interactive --name "$smoke_container" --network none \
    --no-healthcheck --env NVIDIA_VISIBLE_DEVICES=void --entrypoint python "$local_image" - <<'PY'
from pathlib import Path

import cv2
import dill
import fastapi
import numpy as np
import torch
import uvicorn
import yaml

import server

assert callable(server.create_app)
assert torch.version.cuda is not None, "Expected a CUDA-enabled PyTorch build"
assert torch.ones(2, 2).sum().item() == 4
assert cv2.cvtColor(np.zeros((2, 2, 3), dtype=np.uint8), cv2.COLOR_BGR2GRAY).shape == (2, 2)
web = Path("/app/web/dist")
for relative in ("index.html", "docs-assets/swagger-ui-bundle.js", "docs-assets/swagger-ui.css"):
    asset = web / relative
    assert asset.is_file() and asset.stat().st_size, f"Missing frontend asset: {asset}"
assert any(asset.stat().st_size for asset in (web / "assets").glob("*.js")), "Missing frontend JS"
print(f"Image smoke check passed: torch={torch.__version__}, CUDA build={torch.version.cuda}")
PY

sha_image="$CI_REGISTRY_IMAGE:sha-$CI_COMMIT_SHA"
docker image tag "$local_image" "$sha_image"
docker image push "$sha_image"

if [[ -n "$version_tag" && "$version_tag" != "latest" && "$version_tag" != "sha-$CI_COMMIT_SHA" ]]; then
    docker image tag "$local_image" "$CI_REGISTRY_IMAGE:$version_tag"
    docker image push "$CI_REGISTRY_IMAGE:$version_tag"
fi

# Publish latest only after the build, smoke check and all fixed-tag pushes.
docker image tag "$local_image" "$CI_REGISTRY_IMAGE:latest"
docker image push "$CI_REGISTRY_IMAGE:latest"
printf 'Published %s and %s:latest\n' "$sha_image" "$CI_REGISTRY_IMAGE"
