#!/bin/bash
# Usage: msys2_download_packages.sh [SUFFIX]
# Downloads packages from `msys2_package_hashes<SUFFIX>.txt`. URLs are
# derived from each filename's package-type prefix (matches the writer
# side in `msys2_remember_current_packages.sh`).
set -euo pipefail
cd "$(dirname "$BASH_SOURCE")"

SUFFIX="${1:-}"
HASH_FILE="msys2_package_hashes${SUFFIX}.txt"

mkdir -p msys2_packages

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

# `tr -d '\r'` defends against git autocrlf on Windows checkouts.
while read -r _hash file ; do
  fn=${file#\*msys2_packages/}
  printf 'https://mirror.msys2.org/%s/%s\n' "$(url_prefix_for "$fn")" "$fn" >>"$URL_LIST"
done < <(tr -d '\r' <"${HASH_FILE}")

echo "Downloading packages listed in ${HASH_FILE}. This can take a while..."
# `-nc` skips files already present locally without contacting the
# server — sha256 verify in the install script catches partials.
wget -P msys2_packages -i "$URL_LIST" -q --show-progress -nc
