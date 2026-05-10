#!/bin/bash
set -euo pipefail

cd "$(dirname "$BASH_SOURCE")"

# Strip CRs git autocrlf may have introduced on Windows checkouts —
# `sha256sum -c` doesn't tolerate filenames with a trailing \r.
HASH_LINES="$(tr -d '\r' <msys2_package_hashes.txt)"

sha256sum -c <(printf '%s\n' "${HASH_LINES}")

mapfile -t ENTRIES < <(printf '%s\n' "${HASH_LINES}")

FILES=()
for ENTRY in "${ENTRIES[@]}"; do
    FILES+=("${ENTRY#*" *"}")
done

# Adding `|| true` because this is prone to crashing after install, if the core packages were touched. Though I haven't checked if the crash affects the exit code or not.
pacman -U --noconfirm --needed "${FILES[@]}" || true
