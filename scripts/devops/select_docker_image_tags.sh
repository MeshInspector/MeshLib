#!/usr/bin/env bash
# Usage: select_docker_image_tags.sh <linux|linux-vcpkg> [branch-tag]
#
# Single criterion for Docker image reuse: the content-addressed
# source-checksum-* tag (docker_image_source_checksum.sh) compared against
# the registry by leaf manifest digests (docker_image_leaf_digests.sh), for
# every image of the family (docker_image_list.sh). Prints in GITHUB_OUTPUT
# format:
#   image_tag    — conventional tag the build jobs should pull: `latest` on
#                  push/schedule (no branch-tag argument) and on pull requests
#                  whose sources already match what `latest` points at;
#                  the branch tag otherwise.
#   need_rebuild — true when prepare-images must run: some source-checksum-*
#                  tag is absent (build & push it) or the selected
#                  conventional tag does not point at it (retarget it).
# Diagnostics go to stderr. Must run from the repository root.
set -euo pipefail

family=$1
branch_tag=${2:-}

here=$(dirname "$0")
pairs=$("${here}"/docker_image_list.sh "${family}")

repos=()
hash_leaves_list=()
latest_synced=true
while read -r repo distro; do
  hash_tag=$("${here}"/docker_image_source_checksum.sh "${distro}")
  hash_leaves=$("${here}"/docker_image_leaf_digests.sh "${repo}" "${hash_tag}")
  latest_leaves=$("${here}"/docker_image_leaf_digests.sh "${repo}" latest)
  echo "${repo}:${hash_tag} ${hash_leaves} (latest: ${latest_leaves})" >&2
  repos+=( "${repo}" )
  hash_leaves_list+=( "${hash_leaves}" )
  if [ "${hash_leaves}" = absent ] || [ "${latest_leaves}" != "${hash_leaves}" ]; then
    latest_synced=false
  fi
done <<< "${pairs}"

if [ -z "${branch_tag}" ] || [ "${latest_synced}" = true ]; then
  image_tag=latest
  need_rebuild=$([ "${latest_synced}" = true ] && echo false || echo true)
else
  image_tag=${branch_tag}
  need_rebuild=false
  for i in "${!repos[@]}"; do
    branch_leaves=$("${here}"/docker_image_leaf_digests.sh "${repos[$i]}" "${branch_tag}")
    echo "${repos[$i]}:${branch_tag} ${branch_leaves} (want: ${hash_leaves_list[$i]})" >&2
    if [ "${hash_leaves_list[$i]}" = absent ] || [ "${branch_leaves}" != "${hash_leaves_list[$i]}" ]; then
      need_rebuild=true
    fi
  done
fi

echo "image_tag=${image_tag}"
echo "need_rebuild=${need_rebuild}"
