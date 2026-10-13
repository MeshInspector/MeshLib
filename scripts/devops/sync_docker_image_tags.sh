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

# "<image repository> <checksum-script distro>" lines
case "${family}" in
  linux)
    pairs=$(
      jq -r '.[] | "ghcr.io/meshinspector/meshlib-\(.distro)\(if .arch == "arm64" then "-arm64" else "" end) \(.distro)"' "${matrix_dir}/docker-images-linux.json"
      jq -r '.[] | "ghcr.io/meshinspector/meshlib-emscripten-build-c-bindings-\(.emsdk_image) emscripten-build-c-bindings"' "${matrix_dir}/docker-images-emscripten-c-bindings.json"
    ) ;;
  linux-vcpkg)
    pairs=$(jq -r '.[] | "ghcr.io/meshinspector/meshlib-\(.os)-vcpkg-\(.arch) \(.os)-vcpkg"' "${matrix_dir}/docker-images-linux-vcpkg.json") ;;
  *)
    echo "unknown image family ${family}: nothing to sync" >&2
    exit 0 ;;
esac

# manifest digests of <repo>:<tag>... (public ghcr.io images, anonymous token)
digests() {
  local path=${1#ghcr.io/} t urls=()
  shift
  local token=$(curl -fsS "https://ghcr.io/token?scope=repository:${path}:pull" | jq -r .token)
  for t in "$@"; do urls+=("https://ghcr.io/v2/${path}/manifests/${t}"); done
  curl -sSI -H "Authorization: Bearer ${token}" \
    -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json" \
    "${urls[@]}" | tr -d '\r' | awk 'tolower($1) == "docker-content-digest:" { print $2 }'
}

while read -r repo distro; do
  hash_tag=$("$(dirname "$0")"/docker_image_source_checksum.sh "${distro}")
  d=( $(digests "${repo}" "${tag}" "${hash_tag}" || true) )
  if [ ${#d[@]} -eq 2 ] && [ "${d[0]}" = "${d[1]}" ]; then
    echo "${repo}:${tag} -> ${hash_tag} (already)" >&2
    continue
  fi
  # one-time migration: copy an image built before the move from Docker Hub
  if docker buildx imagetools create -t "${repo}:${tag}" "${repo}:${hash_tag}" >&2 ||
     { docker buildx imagetools create -t "${repo}:${hash_tag}" "docker.io/meshlib/${repo##*/}:${hash_tag}" >&2 &&
       docker buildx imagetools create -t "${repo}:${tag}" "${repo}:${hash_tag}" >&2; }; then
    echo "${repo}:${tag} -> ${hash_tag}" >&2
  else
    echo "${repo}:${hash_tag} absent, needs building" >&2
    exit 1
  fi
done <<< "${pairs}"
