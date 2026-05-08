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
wget -P msys2_packages -i "${URL_FILE}" -q --show-progress -c
