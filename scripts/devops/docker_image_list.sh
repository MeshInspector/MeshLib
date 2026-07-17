#!/usr/bin/env bash
# Usage: docker_image_list.sh <linux|linux-vcpkg>
# Prints one "<Docker Hub repository> <checksum-script distro>" line per image
# of the family. The inventory comes from matrix/docker-images.json, shared
# with the build matrices in .github/workflows/prepare-images.yml; repository
# names are derived here the same way its jobs derive them from the matrix.
# Must run from the repository root.
set -euo pipefail

family=$1

manifest=.github/workflows/matrix/docker-images.json

case "${family}" in
  linux)
    jq -r '
      (.linux[] | "meshlib/meshlib-\(.distro)\(if .arch == "arm64" then "-arm64" else "" end) \(.distro)"),
      (."emscripten-c-bindings"[] | "meshlib/meshlib-emscripten-build-c-bindings-\(.emsdk_image) emscripten-build-c-bindings")
    ' "${manifest}" ;;
  linux-vcpkg)
    jq -r '."linux-vcpkg"[] | "meshlib/meshlib-\(.os)-vcpkg-\(.arch) \(.os)-vcpkg"' "${manifest}" ;;
  *)
    echo "unknown image family: ${family}" >&2
    exit 1 ;;
esac
