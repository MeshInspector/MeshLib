#!/bin/bash
set -eo pipefail

SOURCE_DIR="$1"
BUILD_DIR="${2:-./libjpeg-turbo_build}"

CMAKE_OPTIONS="${CMAKE_OPTIONS} \
  -D WITH_JAVA=OFF \
  -D WITH_JPEG8=ON \
  -D WITH_SIMD=OFF \
  -D BUILD_TESTING=OFF \
  -D ENABLE_STATIC=OFF \
"

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" -D CMAKE_C_FLAGS="${CFLAGS}" ${CMAKE_OPTIONS}
cmake --build "${BUILD_DIR}" -j `nproc`
cmake --install "${BUILD_DIR}"
