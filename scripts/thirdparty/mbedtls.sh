#!/bin/bash
set -eo pipefail

SOURCE_DIR="$1"
BUILD_DIR="${2:-./mbedtls_build}"

CMAKE_OPTIONS="${CMAKE_OPTIONS} \
  -D ENABLE_PROGRAMS=OFF \
  -D ENABLE_TESTING=OFF \
"

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" -D CMAKE_C_FLAGS="${CFLAGS}" ${CMAKE_OPTIONS}
cmake --build "${BUILD_DIR}" -j `nproc`
cmake --install "${BUILD_DIR}"
