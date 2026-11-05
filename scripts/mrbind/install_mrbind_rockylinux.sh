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

# With `MRBIND_GENERATORS_ONLY=1`, only the generators working from the C bindings description: no parser (so no libclang) and no `mrbind_gen_c`.
CONFIGURE_ARGS=() BUILD_ARGS=()
if [[ ${MRBIND_GENERATORS_ONLY:-} == 1 ]]; then
    CONFIGURE_ARGS=( -DMRBIND_BUILD_PARSER=OFF )
    BUILD_ARGS=( --target mrbind_gen_csharp mrbind_gen_rust mrbind_gen_java mrbind_gen_python )
    # Ninja builds those targets in parallel, `make` one after another.
    if command -v ninja >/dev/null 2>&1 ; then
        CONFIGURE_ARGS+=( -G Ninja )
    fi
fi

# mrbind links the keg's libLLVM.so/libclang-cpp.so; rpath them for runtime.
CC=clang CXX=clang++ cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_FLAGS="$GCC_INSTALL_DIR_FLAG" \
    -DCMAKE_EXE_LINKER_FLAGS="$GCC_INSTALL_DIR_FLAG -Wl,-rpath,$LLVM_PREFIX/lib" ${CONFIGURE_ARGS[@]+"${CONFIGURE_ARGS[@]}"}
cmake --build build -j$JOBS ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}
