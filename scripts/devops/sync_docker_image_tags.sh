#!/usr/bin/env bash
# Usage: sync_docker_image_tags.sh <linux|linux-vcpkg> <tag>
#
# Points <tag> of every image of the family at its content-addressed
# source-checksum-* tag (docker_image_source_checksum.sh) with an
# unconditional `docker buildx imagetools create` (requires a prior docker
# login): a no-op when it already points there, a cheap retag when it
# doesn't, and a failure exactly when the source-checksum-* tag is absent,
# i.e. when the image really must be built — then this script exits 1;
# it exits 0 when there is nothing to build. Must run from the repository
# root. The image inventory comes from the matrix/docker-images-*.json
# files, shared with the build matrices in
# .github/workflows/prepare-images.yml; Docker Hub repository names are
# derived here the same way its jobs derive them from the matrix.
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
