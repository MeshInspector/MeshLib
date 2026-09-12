#!/bin/bash
set -eo pipefail

SOURCE_DIR="$1"
BUILD_DIR="${2:-./libjpeg-turbo_build}"

# sharedlib/ builds example programs that link the .so, and on wasm a .so is a side module,
# so emcc links them as a main module, where every object has to be PIC. emsdk 4.0.19 let
# that pass; the wasm-ld in 6.0.9 rejects it.
CMAKE_OPTIONS="${CMAKE_OPTIONS} \
  -D WITH_JAVA=OFF \
  -D WITH_JPEG8=ON \
  -D WITH_SIMD=OFF \
  -D BUILD_TESTING=OFF \
  -D ENABLE_STATIC=OFF \
  -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
"

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" -D CMAKE_C_FLAGS="${CFLAGS}" ${CMAKE_OPTIONS}
cmake --build "${BUILD_DIR}" -j `nproc`
cmake --install "${BUILD_DIR}"
