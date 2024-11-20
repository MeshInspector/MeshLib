#!/bin/bash

# Downloads the source code for MRBind and builds it at `MeshLib/mrbind/build`.

set -euxo pipefail

SCRIPT_DIR="$(realpath "$(dirname "$BASH_SOURCE")")"

[[ -v MRBIND_DIR ]] || MRBIND_DIR="$(realpath "$SCRIPT_DIR/../../mrbind")"
[[ -v MRBIND_COMMIT ]] || MRBIND_COMMIT="$(cat "$SCRIPT_DIR/mrbind_commit.txt" | xargs)"

# Read the Clang version from `clang_version.txt`. `xargs` trims the whitespace.
CLANG_VER="$(cat "$SCRIPT_DIR/clang_version.txt" | xargs)"
[[ $CLANG_VER ]] || (echo "Not sure what version of Clang to use." && false)

# Clone mrbind source.
if [[ ! -d $MRBIND_DIR ]]; then
    mkdir -p "$MRBIND_DIR"
    git clone https://github.com/MeshInspector/mrbind "$MRBIND_DIR"
    cd "$MRBIND_DIR"
else
    cd "$MRBIND_DIR"
    git fetch
fi

git checkout "$MRBIND_COMMIT"

rm -rf build


# Guess the number of build threads.
[[ ${JOBS:=} ]] || JOBS=$(nproc)

# `Clang_DIR` is needed when several versions of libclang are installed.
# By default CMake picks an arbitrary one. Supposedly whatever globbing `clang-*` returns first.
CC=clang-$CLANG_VER CXX=clang++-$CLANG_VER cmake -B build -DClang_DIR=/usr/lib/cmake/clang-$CLANG_VER
cmake --build build -j$JOBS
