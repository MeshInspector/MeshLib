#!/usr/bin/env bash
# Exit 0 if every given public Docker Hub image reference (repo:tag) exists in
# the registry; exit 1 if any is missing.
# Usage: docker_image_exists.sh repo:tag [repo:tag ...]
set -euo pipefail

for ref in "$@"; do
  repo="${ref%:*}"
  tag="${ref##*:}"
  token=$(curl -fsSL "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${repo}:pull" | jq -r .token)
  code=$(curl -s -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer ${token}" \
    -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
    -H "Accept: application/vnd.docker.distribution.manifest.list.v2+json" \
    -H "Accept: application/vnd.oci.image.index.v1+json" \
    "https://registry-1.docker.io/v2/${repo}/manifests/${tag}")
  [ "${code}" = "200" ] || exit 1
done
