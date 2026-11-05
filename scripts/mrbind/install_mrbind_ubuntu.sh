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

if [[ -n "${LLVM_PREFIX:-}" ]]; then
    export CC="$LLVM_PREFIX/bin/clang" CXX="$LLVM_PREFIX/bin/clang++"
    # mrbind links this Clang's libLLVM.so/libclang-cpp.so, so rpath them for runtime.
    EXTRA_FLAGS=( -DCMAKE_PREFIX_PATH="$LLVM_PREFIX" -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,"$LLVM_PREFIX/lib" )
else
    export CC=clang-$CLANG_VER CXX=clang++-$CLANG_VER
    # `Clang_DIR` is needed when several versions of libclang are installed.
    # By default CMake picks an arbitrary one. Supposedly whatever globbing `clang-*` returns first.
    EXTRA_FLAGS=( -DClang_DIR=/usr/lib/cmake/clang-$CLANG_VER -DCMAKE_LINKER_TYPE=LLD )
fi

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

cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo "${EXTRA_FLAGS[@]}" ${CONFIGURE_ARGS[@]+"${CONFIGURE_ARGS[@]}"}
cmake --build build -j$JOBS ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}
