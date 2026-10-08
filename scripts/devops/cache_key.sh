#!/usr/bin/env bash
set -euo pipefail

files=()
# NOTE: paths enter the hash as spelled, so spell them identically wherever keys must match.
for path in "$@"; do
    if [ -d "$path" ]; then
        found=$(find "$path" -type f ! -name .DS_Store)
        while IFS= read -r file; do
            files+=("$file")
        done <<< "$found"
    else
        files+=("$path")
    fi
done
checksums=$(sha256sum "${files[@]}")
printf '%s\n' "$checksums" | LC_ALL=C sort | sha256sum | cut -d' ' -f1
