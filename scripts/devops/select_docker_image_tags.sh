#!/usr/bin/env bash
# Usage: select_docker_image_tags.sh <linux|linux-vcpkg> [branch-tag]
#
# Single criterion for Docker image reuse: every image of the family
# (docker_image_list.sh) must have its content-addressed source-checksum-*
# tag (docker_image_source_checksum.sh) in the registry, with the
# conventional tag pointing at it. With TAG_WRITES=true (internal builds;
# requires a prior docker login) the conventional tag is synced right here
# by an unconditional `docker buildx imagetools create` — a no-op when it
# already points there, a cheap retag when it doesn't, and a failure exactly
# when the source-checksum-* tag is absent, i.e. when the image really must
# be built. Without TAG_WRITES (fork and Dependabot PRs, no secrets) the
# registry is not inspected at all: such runs just use `latest` as-is.
# Prints in GITHUB_OUTPUT format:
#   image_tag    — conventional tag the build jobs should pull: `latest` on
#                  push/schedule (no branch-tag argument) and on pull requests
#                  whose sources already match what `latest` points at;
#                  the branch tag otherwise.
#   need_rebuild — true when prepare-images must run: some image failed to
#                  sync (its source-checksum-* tag must be built, pushed and
#                  conventionally tagged there).
# Diagnostics go to stderr. Must run from the repository root.
set -euo pipefail

family=$1
branch_tag=${2:-}

if [ "${TAG_WRITES:-false}" != true ]; then
  echo "image_tag=latest"
  echo "need_rebuild=false"
  exit 0
fi

here=$(dirname "$0")
pairs=$("${here}"/docker_image_list.sh "${family}")

repos=()
hash_tags=()
while read -r repo distro; do
  repos+=( "${repo}" )
  hash_tags+=( "$("${here}"/docker_image_source_checksum.sh "${distro}")" )
done <<< "${pairs}"

# On a pull request, keep resolving to `latest` while it still points at
# every wanted image (read-only check): no tags are minted for branches
# that do not change image sources.
if [ -n "${branch_tag}" ]; then
  latest_synced=true
  for i in "${!repos[@]}"; do
    hash_leaves=$("${here}"/docker_image_leaf_digests.sh "${repos[$i]}" "${hash_tags[$i]}")
    latest_leaves=$("${here}"/docker_image_leaf_digests.sh "${repos[$i]}" latest)
    echo "${repos[$i]}:${hash_tags[$i]} ${hash_leaves} (latest: ${latest_leaves})" >&2
    if [ "${hash_leaves}" = absent ] || [ "${latest_leaves}" != "${hash_leaves}" ]; then
      latest_synced=false
    fi
  done
  if [ "${latest_synced}" = true ]; then
    echo "image_tag=latest"
    echo "need_rebuild=false"
    exit 0
  fi
  conventional_tag=${branch_tag}
else
  conventional_tag=latest
fi

need_rebuild=false
for i in "${!repos[@]}"; do
  if docker buildx imagetools create -t "${repos[$i]}:${conventional_tag}" "${repos[$i]}:${hash_tags[$i]}" >&2; then
    echo "${repos[$i]}:${conventional_tag} -> ${hash_tags[$i]}" >&2
  else
    echo "${repos[$i]}:${hash_tags[$i]} absent, needs building" >&2
    need_rebuild=true
  fi
done

echo "image_tag=${conventional_tag}"
echo "need_rebuild=${need_rebuild}"
