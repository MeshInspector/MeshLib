#!/bin/bash

# Build the MRBind submodule at `MeshLib/thirdparty/mrbind/build`.

set -euxo pipefail

SCRIPT_DIR="$(realpath "$(dirname "$BASH_SOURCE")")"

[[ -v MRBIND_DIR ]] || MRBIND_DIR="$(realpath "$SCRIPT_DIR/../../thirdparty/mrbind")"

cd "$MRBIND_DIR"
rm -rf build


# Guess the number of build threads.
[[ ${JOBS:=} ]] || JOBS=$(nproc)

CC=clang CXX=clang++ cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j$JOBS
