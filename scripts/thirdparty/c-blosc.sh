#!/bin/bash
set -eo pipefail

SOURCE_DIR="$1"
BUILD_DIR="${2:-./c-blosc_build}"

CMAKE_OPTIONS="${CMAKE_OPTIONS} \
  -D CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -D BUILD_TESTS=OFF \
  -D BUILD_FUZZERS=OFF \
  -D BUILD_BENCHMARKS=OFF \
  -D BUILD_STATIC=OFF \
  -D BLOSC_INSTALL=ON \
"

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" -D CMAKE_C_FLAGS="${CFLAGS}" ${CMAKE_OPTIONS}
cmake --build "${BUILD_DIR}" -j `nproc`
cmake --install "${BUILD_DIR}"
