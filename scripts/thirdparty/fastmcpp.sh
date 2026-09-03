#!/bin/bash
set -eo pipefail

SOURCE_DIR="$1"
BUILD_DIR="${2:-./fastmcpp_build}"

CMAKE_OPTIONS="${CMAKE_OPTIONS} \
  -D FASTMCPP_BUILD_TESTS=OFF \
  -D FASTMCPP_BUILD_EXAMPLES=OFF \
  -D FASTMCPP_FETCH_CURL=OFF \
  -D CMAKE_CXX_FLAGS=-fPIC \
"

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" -D CMAKE_C_FLAGS="${CFLAGS}" ${CMAKE_OPTIONS}
cmake --build "${BUILD_DIR}" -j `nproc`
cmake --install "${BUILD_DIR}"
