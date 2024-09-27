#!/bin/bash

# Downloads the source code for MRBind and builds it at `~/mrbind/build`.

set -euxo pipefail

[[ $MRBIND_DIR ]] || MRBIND_DIR=~/mrbind

# Read the Clang version from `preferred_clang_version.txt`. `xargs` trims the whitespace.
SCRIPT_DIR="$(realpath "$(dirname "$BASH_SOURCE")")"
CLANG_VER="$(cat $SCRIPT_DIR/preferred_clang_version.txt | xargs)"
[[ $CLANG_VER ]] || (echo "Not sure what version of Clang to use." && false)

# Clone mrbind, or pull the latest version.
if [[ -d $MRBIND_DIR ]]; then
    cd "$MRBIND_DIR"
    git checkout master
    git pull
else
    mkdir -p "$MRBIND_DIR"
    git clone https://github.com/MeshInspector/mrbind "$MRBIND_DIR"
    cd "$MRBIND_DIR"
fi

rm -rf build


# Guess the number of build threads.
if [[ ! -v JOBS || $JOBS == "" ]]; then
    if command -v nproc >/dev/null 2>/dev/null; then
        JOBS=$(nproc)
    else
        # Some default.
        JOBS=4
    fi
fi

# `Clang_DIR` is needed when several versions of libclang are installed.
# By default CMake picks an arbitrary one. Supposedly whatever globbing `clang-*` returns first.
CC=clang-$CLANG_VER CXX=clang++-$CLANG_VER cmake -B build -DClang_DIR=/usr/lib/cmake/clang-$CLANG_VER
cmake --build build -j$(JOBS)
