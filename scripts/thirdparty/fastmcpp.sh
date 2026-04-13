#!/bin/bash
set -exo pipefail

SOURCE_DIR="$1"
BUILD_DIR="${2:-./fastmcpp_build}"
INSTALL_DIR="${3:-./fastmcpp_install}"

# Sync those flags with `source/fastmcpp/CMakeLists.txt`.
CMAKE_OPTIONS="${CMAKE_OPTIONS} \
  -D FASTMCPP_BUILD_TESTS=OFF \
  -D FASTMCPP_BUILD_EXAMPLES=OFF \
  -D FASTMCPP_FETCH_CURL=OFF \
"

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" -D CMAKE_C_FLAGS="${CFLAGS}" ${CMAKE_OPTIONS}
cmake --build "${BUILD_DIR}" -j `nproc`

# Fastmcpp doesn't install any files via CMake. We use the headers directly from the submodule, and just copy the library.
mkdir -p "$INSTALL_DIR/lib"
cp -R "$SOURCE_DIR"/include/fastmcpp* "$INSTALL_DIR/include"
cp "$BUILD_DIR/libfastmcpp_core.a" "$INSTALL_DIR/lib"
