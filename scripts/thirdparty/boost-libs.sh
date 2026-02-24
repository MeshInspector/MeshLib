#!/bin/bash
set -eo pipefail

SOURCE_DIR="$1"
BUILD_DIR="${2:-./boost-libs_build}"

CMAKE_OPTIONS="${CMAKE_OPTIONS} \
  -D CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -D BOOST_INCLUDE_LIBRARIES=locale \
  -D BOOST_LOCALE_ENABLE_ICONV=OFF \
"

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" -D CMAKE_C_FLAGS="${CFLAGS}" ${CMAKE_OPTIONS}
cmake --build "${BUILD_DIR}" -j `nproc`
cmake --install "${BUILD_DIR}"
