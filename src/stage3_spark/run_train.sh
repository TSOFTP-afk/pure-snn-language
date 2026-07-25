#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${STAGE3_IMAGE:-flux-train:latest}"
NAME="${STAGE3_CONTAINER:-pure-snn-stage3}"
cd "$ROOT"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run --rm --name "$NAME" --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/stage3-home \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$ROOT:/workspace/pure-snn-language" \
  -w /workspace/pure-snn-language \
  --entrypoint python3 \
  "$IMAGE" src/stage3_spark/train.py "$@"
