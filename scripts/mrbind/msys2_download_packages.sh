#!/bin/bash
# Usage: msys2_download_packages.sh [SUFFIX]
#
# Downloads the packages whose `<sha256> *msys2_packages/<filename>`
# entries are listed in `msys2_package_hashes<SUFFIX>.txt` (default
# suffix `''`). Pass a suffix like `_clang18` to use the alternate
# lockfile `msys2_package_hashes_clang18.txt`.
#
# URLs are derived from each filename's package-type prefix (matching
# the mapping `msys2_remember_current_packages.sh` uses on the writing
# side), so we don't need a separate URLs lockfile in lockstep with
# the hashes one.
set -euo pipefail
cd "$(dirname "$BASH_SOURCE")"

SUFFIX="${1:-}"
HASH_FILE="msys2_package_hashes${SUFFIX}.txt"

mkdir -p msys2_packages

# Build the URL list from the hash file's filenames.
#   mingw-w64-clang-x86_64-*  → mingw/clang64
#   mingw-w64-clang-aarch64-* → mingw/clangarm64
#   mingw-w64-ucrt-x86_64-*   → mingw/ucrt64
#   mingw-w64-x86_64-*        → mingw/x86_64
#   mingw-w64-i686-*          → mingw/i686
#   anything else             → msys/x86_64  (msys2 base set)
url_prefix_for() {
  case $1 in
    mingw-w64-clang-x86_64-*)   echo mingw/clang64 ;;
    mingw-w64-clang-aarch64-*)  echo mingw/clangarm64 ;;
    mingw-w64-ucrt-x86_64-*)    echo mingw/ucrt64 ;;
    mingw-w64-x86_64-*)         echo mingw/x86_64 ;;
    mingw-w64-i686-*)           echo mingw/i686 ;;
    *)                          echo msys/x86_64 ;;
  esac
}

URL_LIST=$(mktemp)
trap 'rm -f "$URL_LIST"' EXIT

# `tr -d '\r'` defends against git autocrlf checking the lockfile out as CRLF.
while read -r _hash file ; do
  fn=${file#\*msys2_packages/}
  printf 'https://mirror.msys2.org/%s/%s\n' "$(url_prefix_for "$fn")" "$fn" >>"$URL_LIST"
done < <(tr -d '\r' <"${HASH_FILE}")

echo "Downloading packages listed in ${HASH_FILE}. This can take a while..."
# `-nc` (no-clobber) skips files already present locally without making
# a network call — the dominant case once the GitHub Actions cache is
# warm. A pinned lockfile means filenames already on disk are never
# stale. Partial / corrupt files are caught by the sha256 verify in
# msys2_install_packages.sh; the cure is invalidating the cache.
wget -P msys2_packages -i "$URL_LIST" -q --show-progress -nc
