#!/usr/bin/env bash
# Usage: select_docker_image_tags.sh <linux|linux-vcpkg> [branch-tag]
#
# Single criterion for Docker image reuse: the content-addressed
# source-checksum-* tag (docker_image_source_checksum.sh) compared against
# the registry by leaf manifest digests. Prints in GITHUB_OUTPUT format:
#   image_tag    — conventional tag the build jobs should pull: `latest` on
#                  push/schedule (no branch-tag argument) and on pull requests
#                  whose sources already match what `latest` points at;
#                  the branch tag otherwise.
#   need_rebuild — true when prepare-images must run: some source-checksum-*
#                  tag is absent (build & push it) or the selected
#                  conventional tag does not point at it (retarget it).
#                  Registry query failures count as absent, i.e. fail toward
#                  rebuilding.
# Diagnostics go to stderr. Must run from the repository root. The image
# lists mirror the build matrices in .github/workflows/prepare-images.yml.
set -euo pipefail

family=$1
branch_tag=${2:-}

# "<Docker Hub repository> <checksum-script distro>" pairs
case "${family}" in
  linux)
    images=(
      'meshlib/meshlib-ubuntu22 ubuntu22'
      'meshlib/meshlib-ubuntu24 ubuntu24'
      'meshlib/meshlib-ubuntu22-arm64 ubuntu22'
      'meshlib/meshlib-ubuntu24-arm64 ubuntu24'
      'meshlib/meshlib-emscripten-arm64 emscripten'
      'meshlib/meshlib-emscripten-generate-c-bindings-arm64 emscripten-generate-c-bindings'
      'meshlib/meshlib-emscripten-build-c-bindings-3.1.38 emscripten-build-c-bindings'
      'meshlib/meshlib-emscripten-build-c-bindings-4.0.19-arm64 emscripten-build-c-bindings'
    ) ;;
  linux-vcpkg)
    images=(
      'meshlib/meshlib-rockylinux8-vcpkg-x64 rockylinux8-vcpkg'
      'meshlib/meshlib-rockylinux8-vcpkg-arm64 rockylinux8-vcpkg'
    ) ;;
  *)
    echo "unknown image family: ${family}" >&2
    exit 1 ;;
esac

# Sorted leaf manifest digests that repo:tag resolves to, or "absent" when
# the tag is missing or the registry cannot be queried. Manifest lists are
# unwrapped to their children: `imagetools create` wraps a single manifest
# into a list, so the conventional tag and the source-checksum-* tag point at
# the same image while having different own digests.
leaves() {
  local repo=$1 tag=$2 token hdr body out=""
  token=$(curl -fsSL "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${repo}:pull" | jq -r .token) || { echo absent; return; }
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
  echo "${out:-absent}"
}

repos=()
hash_leaves_list=()
latest_synced=true
for image in "${images[@]}"; do
  repo=${image% *}
  distro=${image#* }
  hash_tag=$(scripts/devops/docker_image_source_checksum.sh "${distro}")
  hash_leaves=$(leaves "${repo}" "${hash_tag}")
  latest_leaves=$(leaves "${repo}" latest)
  echo "${repo}:${hash_tag} ${hash_leaves} (latest: ${latest_leaves})" >&2
  repos+=( "${repo}" )
  hash_leaves_list+=( "${hash_leaves}" )
  if [ "${hash_leaves}" = absent ] || [ "${latest_leaves}" != "${hash_leaves}" ]; then
    latest_synced=false
  fi
done

if [ -z "${branch_tag}" ] || [ "${latest_synced}" = true ]; then
  image_tag=latest
  need_rebuild=$([ "${latest_synced}" = true ] && echo false || echo true)
else
  image_tag=${branch_tag}
  need_rebuild=false
  for i in "${!repos[@]}"; do
    branch_leaves=$(leaves "${repos[$i]}" "${branch_tag}")
    echo "${repos[$i]}:${branch_tag} ${branch_leaves} (want: ${hash_leaves_list[$i]})" >&2
    if [ "${hash_leaves_list[$i]}" = absent ] || [ "${branch_leaves}" != "${hash_leaves_list[$i]}" ]; then
      need_rebuild=true
    fi
  done
fi

echo "image_tag=${image_tag}"
echo "need_rebuild=${need_rebuild}"
