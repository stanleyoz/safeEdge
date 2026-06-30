#!/bin/bash
# Build the SafeEdge backend image (linux/amd64 for Function Compute) and push
# to a public Docker Hub repo. FC pulls the public image at deploy time — no
# paid registry required.
#
# The image contains NO secrets (injected as FC env vars at runtime; .env is
# .dockerignored), so a public repo is safe.
#
# Prereqs:
#   • free Docker Hub account
#   • docker login            (Docker Hub is the default registry)
#
# Usage:
#   DOCKERHUB_USER=youruser ./deploy/build_and_push.sh [tag]

set -euo pipefail

DOCKERHUB_USER="${DOCKERHUB_USER:?set DOCKERHUB_USER to your Docker Hub username}"
REPO="${REPO:-safeedge-backend}"
TAG="${1:-latest}"
IMAGE="docker.io/${DOCKERHUB_USER}/${REPO}:${TAG}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Building ${IMAGE} (linux/amd64) ==="
# FC runs amd64; buildx makes this work from arm64 hosts too.
docker buildx build \
  --platform linux/amd64 \
  -f "${ROOT}/backend/Dockerfile" \
  -t "${IMAGE}" \
  --load \
  "${ROOT}"

echo "=== Pushing to Docker Hub ==="
docker push "${IMAGE}"

echo ""
echo "Done. Image: ${IMAGE}"
echo "Set vars.image in deploy/s.yaml to this value, then: cd deploy && s deploy"
echo "NOTE: ensure the Docker Hub repo '${REPO}' is PUBLIC (Docker Hub → repo → Settings → Make public)."
