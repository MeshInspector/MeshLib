#!/bin/bash
set -eo pipefail

SOURCE_DIR="$1"
BUILD_DIR="${2:-./libE57Format_build}"

CMAKE_OPTIONS="${CMAKE_OPTIONS} \
  -D E57_XML_PARSER=TinyXML2 \
"

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" ${CMAKE_OPTIONS}
cmake --build "${BUILD_DIR}" -j `nproc`
cmake --install "${BUILD_DIR}"
