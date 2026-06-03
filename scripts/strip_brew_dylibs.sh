#!/bin/bash
set -e

# Strip dylibs we'll bundle (brew keeps full symbol tables for symbolication).
# NOTE: `strip` prints a sig-invalidation warning and exits 1 on signed dylibs.
BREW_PREFIX=$(brew --prefix)
find "$BREW_PREFIX/Cellar" -type f -name '*.dylib' -print0 | while IFS= read -r -d '' f; do
  chmod u+w "$f"
  strip -x "$f" 2>/dev/null || true
  codesign --force --sign - "$f" 2>/dev/null || true
done
