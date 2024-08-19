#!/bin/bash
set -eo pipefail

SOURCE_DIR="$1"
BUILD_DIR="${2:-./libjpeg-turbo_build}"

CMAKE_OPTIONS="${CMAKE_OPTIONS} \
  -D WITH_JAVA=OFF \
  -D WITH_JPEG8=ON \
  -D BUILD_TESTING=OFF \
"

source "$( dirname $0 )"/functions.sh
build_install "${SOURCE_DIR}" "${BUILD_DIR}" "${CMAKE_OPTIONS}"
