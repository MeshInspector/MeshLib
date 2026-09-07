#!/bin/bash
# Usage: msys2_install_packages.sh [SUFFIX]
# Verifies sha256s and `pacman -U`s the packages listed in
# `msys2_package_hashes<SUFFIX>.txt`.
set -euo pipefail

cd "$(dirname "$BASH_SOURCE")"

SUFFIX="${1:-}"
HASH_FILE="msys2_package_hashes${SUFFIX}.txt"

# Strip CRs git autocrlf may have introduced on Windows checkouts —
# `sha256sum -c` doesn't tolerate filenames with a trailing \r.
HASH_LINES="$(tr -d '\r' <"${HASH_FILE}")"

sha256sum -c <(printf '%s\n' "${HASH_LINES}")

mapfile -t ENTRIES < <(printf '%s\n' "${HASH_LINES}")

FILES=()
for ENTRY in "${ENTRIES[@]}"; do
    FILES+=("${ENTRY#*" *"}")
done

# pacman sometimes dies here taking the whole msys2 shell (and any
# not-yet-flushed console output) with it, so tee to a file the caller
# can dump post-mortem. Its exit code is deliberately tolerated: it's
# prone to crashing after install when the core packages were touched.
echo "Installing ${#FILES[@]} packages with pacman -U..."
RC=0
pacman -U --noconfirm --needed "${FILES[@]}" 2>&1 | tee msys2_pacman_install.log || RC=$?
echo "pacman exit status: ${RC}"
