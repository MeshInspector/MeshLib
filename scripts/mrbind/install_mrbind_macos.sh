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
# Add Clang to PATH.
if [[ -n "${LLVM_PREFIX:-}" ]]; then
    export PATH="$LLVM_PREFIX/bin:$PATH"
else
    export PATH="$HOMEBREW_DIR/opt/llvm@$CLANG_VER/bin:$PATH"
fi


# Guess the number of build threads.
[[ ${JOBS:=} ]] || JOBS=$(sysctl -n hw.ncpu)

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

CC=clang CXX=clang++ cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_LINKER_TYPE=LLD ${CONFIGURE_ARGS[@]+"${CONFIGURE_ARGS[@]}"}
cmake --build build -j$JOBS ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}
