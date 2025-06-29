#!/bin/bash

# Builds the MRBind submodule at `MeshLib/thirdparty/mrbind/build`.

set -euxo pipefail

SCRIPT_DIR="$(dirname "$BASH_SOURCE")"

[[ ${MRBIND_DIR:=} ]] || MRBIND_DIR="$SCRIPT_DIR/../../thirdparty/mrbind"

CLANG_VER="$("$SCRIPT_DIR/select_clang_version.sh")"
[[ ${CLANG_VER:=} ]] || (echo "Not sure what version of Clang to use." && false)

cd "$MRBIND_DIR"
rm -rf build

# Should be `/opt/homebrew` on ARM and `/usr/local` on x86.
[[ ${HOMEBREW_DIR:=} ]] || HOMEBREW_DIR="$(brew --prefix)"

# Add `make` to PATH.
export PATH="$HOMEBREW_DIR/opt/make/libexec/gnubin:$PATH"
# Add Clang to PATH.
export PATH="$HOMEBREW_DIR/opt/llvm@$CLANG_VER/bin:$PATH"


# Guess the number of build threads.
[[ ${JOBS:=} ]] || JOBS=$(sysctl -n hw.ncpu)

CC=clang CXX=clang++ cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j$JOBS
