#!/bin/bash

# Build the MRBind submodule at `MeshLib/thirdparty/mrbind/build`.

set -euxo pipefail

SCRIPT_DIR="$(realpath "$(dirname "$BASH_SOURCE")")"

[[ -v MRBIND_DIR ]] || MRBIND_DIR="$(realpath "$SCRIPT_DIR/../../thirdparty/mrbind")"

# Read the Clang version from `clang_version.txt`. `xargs` trims the whitespace.
CLANG_VER="$(cat "$SCRIPT_DIR/clang_version.txt" | xargs)"
[[ $CLANG_VER ]] || (echo "Not sure what version of Clang to use." && false)

cd "$MRBIND_DIR"
rm -rf build


# Guess the number of build threads.
[[ ${JOBS:=} ]] || JOBS=$(nproc)

# `Clang_DIR` is needed when several versions of libclang are installed.
# By default CMake picks an arbitrary one. Supposedly whatever globbing `clang-*` returns first.
CC=clang-$CLANG_VER CXX=clang++-$CLANG_VER cmake -B build -DClang_DIR=/usr/lib/cmake/clang-$CLANG_VER -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j$JOBS
