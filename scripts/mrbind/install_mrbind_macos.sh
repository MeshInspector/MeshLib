#!/bin/bash

# Builds the MRBind submodule at `MeshLib/thirdparty/mrbind/build`.

set -euxo pipefail

SCRIPT_DIR="$(dirname "$BASH_SOURCE")"

[[ ${MRBIND_DIR:=} ]] || MRBIND_DIR="$SCRIPT_DIR/../../thirdparty/mrbind"

# Read the Clang version from `clang_version_macos.txt`. `xargs` trims the whitespace.
# Some versions of MacOS seem to lack `realpath`, so not using it here.
CLANG_VER="$(cat "$SCRIPT_DIR/clang_version_macos.txt" | xargs)"
[[ ${CLANG_VER:=} ]] || (echo "Not sure what version of Clang to use." && false)

cd "$MRBIND_DIR"
rm -rf build

# Should be `/opt/homebrew` on ARM and `/usr/local` on x86.
[[ ${HOMEBREW_DIR:=} ]] || HOMEBREW_DIR="$(brew --prefix)"

# Add `make` to PATH.
export PATH="$HOMEBREW_DIR/opt/make/libexec/gnubin:$PATH"
# Add Clang to PATH. LLVM_VERSION (set for self-hosted runners) selects an
# exact keg of the main formula.
if [[ -n "${LLVM_VERSION:-}" ]]; then
    export PATH="$HOMEBREW_DIR/Cellar/llvm/$LLVM_VERSION/bin:$PATH"
else
    export PATH="$HOMEBREW_DIR/opt/llvm@$CLANG_VER/bin:$PATH"
fi


# Guess the number of build threads.
[[ ${JOBS:=} ]] || JOBS=$(sysctl -n hw.ncpu)

CC=clang CXX=clang++ cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_LINKER_TYPE=LLD
cmake --build build -j$JOBS
