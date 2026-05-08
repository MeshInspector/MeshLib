#!/bin/bash
# Usage: msys2_download_packages.sh [SUFFIX]
#
# Downloads URLs from `msys2_package_urls<SUFFIX>.txt` (default suffix `''`)
# into `msys2_packages/`. Pass a suffix like `_clang18` to use the alternate
# lockfile `msys2_package_urls_clang18.txt`.
set -euo pipefail
cd "$(dirname "$BASH_SOURCE")"
SUFFIX="${1:-}"
URL_FILE="msys2_package_urls${SUFFIX}.txt"
echo "Downloading packages from ${URL_FILE}. This can take a while..."
# `-nc` (no-clobber) skips files already present locally without making a
# network call — the dominant case once the GitHub Actions cache is warm.
# A pinned lockfile means filenames already on disk are never stale.
# Partial / corrupt files are caught by the sha256 verify in
# msys2_install_packages.sh; the cure is invalidating the cache.
wget -P msys2_packages -i "${URL_FILE}" -q --show-progress -nc
