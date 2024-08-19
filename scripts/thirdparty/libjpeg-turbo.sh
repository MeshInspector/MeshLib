#!/bin/bash
set -eo pipefail

SOURCE_DIR="$1"
BUILD_DIR="${2:-./libjpeg-turbo_build}"

CMAKE_OPTIONS="${CMAKE_OPTIONS} \
  -D WITH_JAVA=OFF \
  -D WITH_JPEG8=ON \
  -D BUILD_TESTING=OFF \
"

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" ${CMAKE_OPTIONS}
# FIXME: build might fail on the first try due to linkage's race condition (?)
set +e
cmake --build "${BUILD_DIR}" -j `nproc`
set -e
cmake --build "${BUILD_DIR}" -j `nproc`
cmake --install "${BUILD_DIR}"
