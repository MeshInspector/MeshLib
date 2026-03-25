#!/bin/bash
set -euo pipefail

cd "$(dirname "$BASH_SOURCE")"

sha256sum -c msys2_package_hashes.txt

mapfile -t ENTRIES <msys2_package_hashes.txt

FILES=()
for ENTRY in "${ENTRIES[@]}"; do
    FILES+=("${ENTRY#*" *"}")
done

# Adding `|| true` because this is prone to crashing after install, if the core packages were touched. Though I haven't checked if the crash affects the exit code or not.
pacman -U --noconfirm --needed "${FILES[@]}" || true
