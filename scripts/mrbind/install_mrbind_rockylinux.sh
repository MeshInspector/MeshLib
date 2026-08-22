#!/bin/bash

# Build the MRBind submodule at `MeshLib/thirdparty/mrbind/build`.

set -euxo pipefail

SCRIPT_DIR="$(realpath "$(dirname "$BASH_SOURCE")")"

[[ -v MRBIND_DIR ]] || MRBIND_DIR="$(realpath "$SCRIPT_DIR/../../thirdparty/mrbind")"

cd "$MRBIND_DIR"
rm -rf build


# Guess the number of build threads.
[[ ${JOBS:=} ]] || JOBS=$(nproc)

# LLVM_PREFIX selects a keg by full path; default to the image's PGO clang.
[[ -n "${LLVM_PREFIX:-}" ]] || LLVM_PREFIX=/opt/llvm-pgo-22.1.8
export PATH="$LLVM_PREFIX/bin:$PATH"
export CMAKE_PREFIX_PATH="$LLVM_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

# The keg ships no libLLVM.so, so link LLVM statically; its archives hold
# ThinLTO bitcode, which only the keg's own lld can link (and CMake < 3.29
# ignores CMAKE_LINKER_TYPE). No libunwind.a in the keg: libgcc_eh unwinds.
CC=clang CXX=clang++ cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DMRBIND_STATIC_BUILD=ON -DMRBIND_FORCE_LLVM_STATIC=ON -DUNWIND_STATIC=-lgcc_eh \
    -DCMAKE_EXE_LINKER_FLAGS=--ld-path="$LLVM_PREFIX/bin/ld.lld"
cmake --build build -j$JOBS
