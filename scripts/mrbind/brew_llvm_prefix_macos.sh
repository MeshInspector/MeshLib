#!/bin/bash

# Prints the Homebrew LLVM prefix to build with: the main `llvm` keg when its
# major matches `clang_version_macos.txt`, otherwise the versioned `llvm@N` one.
# Homebrew ships PGO+LTO-optimized bottles only for the main `llvm` formula on
# Apple Silicon, never for versioned `llvm@N` (~12% compile-time difference), so
# runners keep a matching main keg around via `brew pin llvm`.

set -euo pipefail

SCRIPT_DIR="$(dirname "$BASH_SOURCE")"
CLANG_VER="$(cat "$SCRIPT_DIR/clang_version_macos.txt" | xargs)"
[[ $CLANG_VER ]] || { echo "Not sure what version of Clang to use." >&2; exit 1; }

MAIN_LLVM_PREFIX="$(brew --prefix llvm 2>/dev/null || true)"
if [[ -x "$MAIN_LLVM_PREFIX/bin/clang" ]] && \
   [[ "$("$MAIN_LLVM_PREFIX/bin/clang" --version 2>/dev/null | head -n1)" == *"clang version $CLANG_VER."* ]]; then
    echo "$MAIN_LLVM_PREFIX"
else
    brew --prefix "llvm@$CLANG_VER"
fi
