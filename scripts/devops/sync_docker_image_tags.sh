#!/usr/bin/env bash
# Usage: sync_docker_image_tags.sh <linux|linux-vcpkg> <tag>
# Points <tag> of every family image at its source-checksum-* tag with
# `docker buildx imagetools create` (needs a prior docker login; run from
# the repository root). Exits 1 when some source-checksum-* tag is absent,
# i.e. the image must be built first; 0 when there is nothing to build.
set -euo pipefail

family=$1
tag=$2

matrix_dir=.github/workflows/matrix

# "<Docker Hub repository> <checksum-script distro>" lines
case "${family}" in
  linux)
    pairs=$(
      jq -r '.[] | "meshlib/meshlib-\(.distro)\(if .arch == "arm64" then "-arm64" else "" end) \(.distro)"' "${matrix_dir}/docker-images-linux.json"
      jq -r '.[] | "meshlib/meshlib-emscripten-build-c-bindings-\(.emsdk_image) emscripten-build-c-bindings"' "${matrix_dir}/docker-images-emscripten-c-bindings.json"
    ) ;;
  linux-vcpkg)
    pairs=$(jq -r '.[] | "meshlib/meshlib-\(.os)-vcpkg-\(.arch) \(.os)-vcpkg"' "${matrix_dir}/docker-images-linux-vcpkg.json") ;;
  *)
    echo "unknown image family ${family}: nothing to sync" >&2
    exit 0 ;;
esac

while read -r repo distro; do
  hash_tag=$("$(dirname "$0")"/docker_image_source_checksum.sh "${distro}")
  if docker buildx imagetools create -t "${repo}:${tag}" "${repo}:${hash_tag}" >&2; then
    echo "${repo}:${tag} -> ${hash_tag}" >&2
  else
    echo "${repo}:${hash_tag} absent, needs building" >&2
    exit 1
  fi
done <<< "${pairs}"
