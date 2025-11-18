#!/bin/bash
set -eo pipefail

SOURCE_DIR="$1"
BUILD_DIR="${2:-./clip_build}"

CMAKE_OPTIONS="${CMAKE_OPTIONS} \
  -D CLIP_EXAMPLES=OFF \
  -D CLIP_TESTS=OFF \
  -D CLIP_X11_WITH_PNG=OFF \
  -D BUILD_SHARED_LIBS=ON \
"

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" -D CMAKE_C_FLAGS="${CFLAGS}" ${CMAKE_OPTIONS}
cmake --build "${BUILD_DIR}" -j `nproc`
cmake --install "${BUILD_DIR}"
