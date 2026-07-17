#!/usr/bin/env bash
# Usage: select_docker_image_tags.sh <linux|linux-vcpkg> [branch-tag]
#
# Single criterion for Docker image reuse: every image of the family must
# have its content-addressed source-checksum-* tag
# (docker_image_source_checksum.sh) in the registry, with the conventional
# tag pointing at it. With TAG_WRITES=true (internal builds; requires a
# prior docker login) the conventional tag is synced right here by an
# unconditional `docker buildx imagetools create` — a no-op when it already
# points there, a cheap retag when it doesn't, and a failure exactly when
# the source-checksum-* tag is absent, i.e. when the image really must be
# built. Without TAG_WRITES (fork and Dependabot PRs, no secrets) the
# registry is not inspected at all: such runs just use `latest` as-is.
# Prints in GITHUB_OUTPUT format:
#   image_tag    — conventional tag the build jobs should pull: the branch
#                  tag on pull requests, `latest` otherwise.
#   need_rebuild — true when prepare-images must run: some image failed to
#                  sync (its source-checksum-* tag must be built, pushed and
#                  conventionally tagged there).
# Diagnostics go to stderr. Must run from the repository root. The image
# inventory comes from matrix/docker-images.json, shared with the build
# matrices in .github/workflows/prepare-images.yml; Docker Hub repository
# names are derived here the same way its jobs derive them from the matrix.
set -euo pipefail

family=$1
branch_tag=${2:-}

if [ "${TAG_WRITES:-false}" != true ]; then
  echo "image_tag=latest"
  echo "need_rebuild=false"
  exit 0
fi

conventional_tag=${branch_tag:-latest}
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
    exit 1 ;;
esac

here=$(dirname "$0")
need_rebuild=false
while read -r repo distro; do
  hash_tag=$("${here}"/docker_image_source_checksum.sh "${distro}")
  if docker buildx imagetools create -t "${repo}:${conventional_tag}" "${repo}:${hash_tag}" >&2; then
    echo "${repo}:${conventional_tag} -> ${hash_tag}" >&2
  else
    echo "${repo}:${hash_tag} absent, needs building" >&2
    need_rebuild=true
  fi
done <<< "${pairs}"

echo "image_tag=${conventional_tag}"
echo "need_rebuild=${need_rebuild}"
