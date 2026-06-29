#!/usr/bin/env bash
# Print "true" if any of the given public Docker Hub image references
# (repo:tag) is absent from the registry, otherwise "false".
# Usage: docker_image_missing.sh repo:tag [repo:tag ...]
set -euo pipefail

missing=false
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
  echo "  ${ref} -> HTTP ${code}" >&2
  if [ "${code}" != "200" ]; then
    missing=true
  fi
done
echo "${missing}"
