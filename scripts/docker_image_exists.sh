#!/usr/bin/env bash
# Exit 0 if every given image reference (registry/repo:tag) exists in its
# registry; exit 1 if any is missing.
# Usage: docker_image_exists.sh repo:tag [repo:tag ...]
set -euo pipefail

for ref in "$@"; do
  docker buildx imagetools inspect "${ref}" > /dev/null 2>&1 || exit 1
done
