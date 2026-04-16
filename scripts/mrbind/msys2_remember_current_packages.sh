#!/bin/bash

if [[ "$#" != 0 ]]; then
    echo "Don't pass any arguments."
    exit 1
fi

set -euo pipefail

cd "$(dirname "$BASH_SOURCE")"

mapfile -t PACKAGES < <(pacman -Q)

URL_FILE=msys2_package_urls.txt
HASH_FILE=msys2_package_hashes.txt
PACKAGE_DIR=msys2_packages
PACKAGE_FILES=()

# Remove old package lists.
rm -f "$URL_FILE" "$HASH_FILE"

# This will receive the stale packages that we want to delete.
shopt -s nullglob
# This is an associative array.
declare -A UNWANTED_FILES
for f in "$PACKAGE_DIR"/*; do
    UNWANTED_FILES["$f"]=1
done

# Copy or download the packages.
mkdir -p "$PACKAGE_DIR"
for x in "${PACKAGES[@]}"; do
    name="${x% *}"
    ver="${x#* }"

    is_msys=0
    if [[ $name =~ ^mingw-w64-x86_64- ]]; then
        url_part="mingw/x86_64"
    elif [[ $name =~ ^mingw-w64-i686- ]]; then
        url_part="mingw/i686"
    elif [[ $name =~ ^mingw-w64-ucrt-x86_64- ]]; then
        url_part="mingw/ucrt64"
    elif [[ $name =~ ^mingw-w64-clang-x86_64- ]]; then
        url_part="mingw/clang64"
    elif [[ $name =~ ^mingw-w64-clang-x86_64- ]]; then
        url_part="mingw/clang64"
    elif [[ $name =~ ^mingw-w64-clang-aarch64- ]]; then
        url_part="mingw/clangarm64"
    else
        url_part="msys/x86_64"
        is_msys=1
    fi

    # Check the current directory.
    if [[ "$url_part" == "msys/x86_64" && -f "$PACKAGE_DIR/$name-$ver-x86_64.pkg.tar.zst" ]]; then
        echo "https://mirror.msys2.org/$url_part/$name-$ver-x86_64.pkg.tar.zst" >>"$URL_FILE"
        THIS_FILE="$PACKAGE_DIR/$name-$ver-x86_64.pkg.tar.zst"
    elif [[ -f "$PACKAGE_DIR/$name-$ver-any.pkg.tar.zst" ]]; then
        echo "https://mirror.msys2.org/$url_part/$name-$ver-any.pkg.tar.zst" >>"$URL_FILE"
        THIS_FILE="$PACKAGE_DIR/$name-$ver-any.pkg.tar.zst"
    # Check `pacman` cache.
    elif [[ "$url_part" == "msys/x86_64" && -f "/var/cache/pacman/pkg/$name-$ver-x86_64.pkg.tar.zst" ]]; then
        cp "/var/cache/pacman/pkg/$name-$ver-x86_64.pkg.tar.zst" "$PACKAGE_DIR"
        echo "https://mirror.msys2.org/$url_part/$name-$ver-x86_64.pkg.tar.zst" >>"$URL_FILE"
        THIS_FILE="$PACKAGE_DIR/$name-$ver-x86_64.pkg.tar.zst"
    elif [[ -f "/var/cache/pacman/pkg/$name-$ver-any.pkg.tar.zst" ]]; then
        cp "/var/cache/pacman/pkg/$name-$ver-any.pkg.tar.zst" "$PACKAGE_DIR"
        echo "https://mirror.msys2.org/$url_part/$name-$ver-any.pkg.tar.zst" >>"$URL_FILE"
        THIS_FILE="$PACKAGE_DIR/$name-$ver-any.pkg.tar.zst"
    # Check the repository.
    elif [[ "$url_part" == "msys/x86_64" ]] && ( wget -q --show-progress -c "https://mirror.msys2.org/$url_part/$name-$ver-x86_64.pkg.tar.zst" -O "$PACKAGE_DIR/$name-$ver-x86_64.pkg.tar.zst" || ( rm "$PACKAGE_DIR/$name-$ver-x86_64.pkg.tar.zst" && false ) ); then
        echo "https://mirror.msys2.org/$url_part/$name-$ver-x86_64.pkg.tar.zst" >>"$URL_FILE"
        THIS_FILE="$PACKAGE_DIR/$name-$ver-x86_64.pkg.tar.zst"
    else
        wget -q --show-progress -c "https://mirror.msys2.org/$url_part/$name-$ver-any.pkg.tar.zst" -O "$PACKAGE_DIR/$name-$ver-any.pkg.tar.zst"
        echo "https://mirror.msys2.org/$url_part/$name-$ver-any.pkg.tar.zst" >>"$URL_FILE"
        THIS_FILE="$PACKAGE_DIR/$name-$ver-any.pkg.tar.zst"
    fi

    PACKAGE_FILES+=("$THIS_FILE")
    unset "UNWANTED_FILES["$THIS_FILE"]"
done

# Write the file hashes.
sha256sum "${PACKAGE_FILES[@]}" >"$HASH_FILE"

# Delete the stale archives from the package directory.
rm -f "${!UNWANTED_FILES[@]}"
