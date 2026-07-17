#!/usr/bin/env bash
# Usage: docker_image_leaf_digests.sh <repository> <tag>
# Prints the sorted leaf manifest digests that repository:tag resolves to on
# Docker Hub, or "absent" when the tag is missing or the registry cannot be
# queried. Manifest lists are unwrapped to their children: `imagetools create`
# wraps a single manifest into a list, so a conventional tag and the
# source-checksum-* tag point at the same image while having different own
# digests. Never fails: callers treat "absent" as "needs rebuilding".
set -euo pipefail

repo=$1
tag=$2

out=""
if token=$(curl -fsSL "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${repo}:pull" | jq -r .token); then
  hdr=$(mktemp)
  if body=$(curl -fsS -D "${hdr}" \
    -H "Authorization: Bearer ${token}" \
    -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
    -H "Accept: application/vnd.docker.distribution.manifest.list.v2+json" \
    -H "Accept: application/vnd.oci.image.index.v1+json" \
    -H "Accept: application/vnd.oci.image.manifest.v1+json" \
    "https://registry-1.docker.io/v2/${repo}/manifests/${tag}"); then
    if grep -qiE '^content-type:.*(manifest\.list|image\.index)' "${hdr}"; then
      out=$(jq -r '[.manifests[].digest] | sort | join(",")' <<<"${body}") || out=""
    else
      out=$(tr -d '\r' < "${hdr}" | grep -i '^docker-content-digest:' | awk '{print $2}') || out=""
    fi
  fi
  rm -f "${hdr}"
fi
echo "${out:-absent}"
