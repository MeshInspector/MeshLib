#!/bin/bash

# Prints the Homebrew LLVM prefix to build with.
# Self-hosted runners set `LLVM_PREFIX` (in the runner environment or via the
# workflow) to select a keg explicitly -- e.g. the main `llvm` keg, whose
# Apple Silicon bottle is PGO+LTO-optimized unlike the versioned `llvm@N`
# bottles (~12% compile-time difference). Without it, `llvm@N` is used.

set -euo pipefail

SCRIPT_DIR="$(dirname "$BASH_SOURCE")"
CLANG_VER="$(cat "$SCRIPT_DIR/clang_version_macos.txt" | xargs)"
[[ $CLANG_VER ]] || { echo "Not sure what version of Clang to use." >&2; exit 1; }

if [[ -n "${LLVM_PREFIX:-}" ]]; then
    # Explicit configuration must not be silently ignored: verify the keg.
    if ! "$LLVM_PREFIX/bin/clang" --version 2>/dev/null | head -n1 | grep -q "clang version $CLANG_VER\."; then
        echo "LLVM_PREFIX=$LLVM_PREFIX does not hold a Clang $CLANG_VER." >&2
        exit 1
    fi
    echo "$LLVM_PREFIX"
else
    brew --prefix "llvm@$CLANG_VER"
fi
