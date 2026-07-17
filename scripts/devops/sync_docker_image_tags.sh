#!/usr/bin/env bash
# Usage: sync_docker_image_tags.sh <linux|linux-vcpkg> <tag>
#
# Points <tag> of every image of the family at its content-addressed
# source-checksum-* tag (docker_image_source_checksum.sh) with an
# unconditional `docker buildx imagetools create` (requires a prior docker
# login): a no-op when it already points there, a cheap retag when it
# doesn't, and a failure exactly when the source-checksum-* tag is absent,
# i.e. when the image really must be built.
# Exits 0 when every image is synced, 1 when some source-checksum-* tag is
# absent, so prepare-images must build it. Must run from the repository
# root. The image inventory comes from matrix/docker-images.json, shared
# with the build matrices in .github/workflows/prepare-images.yml; Docker
# Hub repository names are derived here the same way its jobs derive them
# from the matrix.
set -euo pipefail

family=$1
tag=$2

manifest=.github/workflows/matrix/docker-images.json

# "<Docker Hub repository> <checksum-script distro>" lines
case "${family}" in
  linux)
    pairs=$(jq -r '
      (.linux[] | "meshlib/meshlib-\(.distro)\(if .arch == "arm64" then "-arm64" else "" end) \(.distro)"),
      (."emscripten-c-bindings"[] | "meshlib/meshlib-emscripten-build-c-bindings-\(.emsdk_image) emscripten-build-c-bindings")
    ' "${manifest}") ;;
  linux-vcpkg)
    pairs=$(jq -r '."linux-vcpkg"[] | "meshlib/meshlib-\(.os)-vcpkg-\(.arch) \(.os)-vcpkg"' "${manifest}") ;;
  *)
    echo "unknown image family: ${family}" >&2
    exit 2 ;;
esac

here=$(dirname "$0")
synced=true
while read -r repo distro; do
  hash_tag=$("${here}"/docker_image_source_checksum.sh "${distro}")
  if docker buildx imagetools create -t "${repo}:${tag}" "${repo}:${hash_tag}" >&2; then
    echo "${repo}:${tag} -> ${hash_tag}" >&2
  else
    echo "${repo}:${hash_tag} absent, needs building" >&2
    synced=false
  fi
done <<< "${pairs}"

[ "${synced}" = true ]
