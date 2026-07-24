#!/bin/bash

# Shallow-init the given submodules, recovering from broken leftover state.
#
# On a persistent runner a cancelled job can leave a submodule half checked
# out (unresolvable HEAD, partial modules dir), which makes a plain
# `git submodule update` fail. On failure this resets the requested
# submodules -- deinit plus removal of both the worktree and the repo's
# modules entry -- and retries once from scratch.

# Note: At the moment we only use this script on Linux, since the runners with network problems (where this matters) are all on Linux.

set -u

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <repo-dir> <submodule-path>..." >&2
  exit 1
fi

repo=$1
shift

update() {
  git -C "$repo" submodule update --init --force --depth 1 -- "$@"
}

git -C "$repo" submodule sync -- "$@"
if update "$@"; then
  exit 0
fi

echo "submodule update failed; resetting and retrying"
git_dir=$( git -C "$repo" rev-parse --absolute-git-dir ) || exit 1
for path in "$@"; do
  git -C "$repo" submodule deinit -f -- "$path"
  rm -rf "${git_dir}/modules/${path}" "${repo:?}/${path}"
done
git -C "$repo" submodule sync -- "$@"
update "$@"
