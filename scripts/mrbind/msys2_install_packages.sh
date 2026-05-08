#!/bin/bash
# Usage: msys2_install_packages.sh [SUFFIX]
#
# Verifies sha256s and `pacman -U`s the packages listed in
# `msys2_package_hashes<SUFFIX>.txt` (default suffix `''`). Pass a suffix
# like `_clang18` to use the alternate lockfile
# `msys2_package_hashes_clang18.txt`.
set -euo pipefail

cd "$(dirname "$BASH_SOURCE")"

SUFFIX="${1:-}"
HASH_FILE="msys2_package_hashes${SUFFIX}.txt"

sha256sum -c "${HASH_FILE}"

mapfile -t ENTRIES <"${HASH_FILE}"

FILES=()
for ENTRY in "${ENTRIES[@]}"; do
    FILES+=("${ENTRY#*" *"}")
done

# Adding `|| true` because this is prone to crashing after install, if the core packages were touched. Though I haven't checked if the crash affects the exit code or not.
pacman -U --noconfirm --needed "${FILES[@]}" || true
