#!/bin/bash

# Build the MRBind submodule at `MeshLib/thirdparty/mrbind/build`.

set -euxo pipefail

SCRIPT_DIR="$(realpath "$(dirname "$BASH_SOURCE")")"

[[ -v MRBIND_DIR ]] || MRBIND_DIR="$(realpath "$SCRIPT_DIR/../../thirdparty/mrbind")"

cd "$MRBIND_DIR"
rm -rf build


# Guess the number of build threads.
[[ ${JOBS:=} ]] || JOBS=$(nproc)

# The CI workflows pass LLVM_PREFIX in the environment (a single definition
# next to the image's Dockerfile pin); local runs must set it explicitly.
: "${LLVM_PREFIX:?point LLVM_PREFIX at the clang installation, e.g. /opt/llvm-pgo-dylib-22.1.8}"
export PATH="$LLVM_PREFIX/bin:$PATH"
export CMAKE_PREFIX_PATH="$LLVM_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

# Optional GCC_INSTALL_DIR: the GCC whose libstdc++ to build against.
GCC_INSTALL_DIR_FLAG=
[ -n "${GCC_INSTALL_DIR:-}" ] && GCC_INSTALL_DIR_FLAG="--gcc-install-dir=$GCC_INSTALL_DIR"

# mrbind links the keg's libLLVM.so/libclang-cpp.so; rpath them for runtime.
CC=clang CXX=clang++ cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_FLAGS="$GCC_INSTALL_DIR_FLAG" \
    -DCMAKE_EXE_LINKER_FLAGS="$GCC_INSTALL_DIR_FLAG -Wl,-rpath,$LLVM_PREFIX/lib"
cmake --build build -j$JOBS
