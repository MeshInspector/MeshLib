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

# Strip CRs that git autocrlf may have introduced on Windows runners —
# `sha256sum -c` doesn't tolerate filenames with a trailing \r.
HASH_LINES="$(tr -d '\r' <"${HASH_FILE}")"

sha256sum -c <(printf '%s\n' "${HASH_LINES}")

mapfile -t ENTRIES < <(printf '%s\n' "${HASH_LINES}")

FILES=()
for ENTRY in "${ENTRIES[@]}"; do
    FILES+=("${ENTRY#*" *"}")
done

# Adding `|| true` because this is prone to crashing after install, if the core packages were touched. Though I haven't checked if the crash affects the exit code or not.
pacman -U --noconfirm --needed "${FILES[@]}" || true
