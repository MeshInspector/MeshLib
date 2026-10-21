#!/bin/bash
set -eo pipefail

SOURCE_DIR="$1"
BUILD_DIR="${2:-./mimalloc_build}"

# must stay equal to thirdparty/cpm's and vcpkg's version, see mr_enable_mimalloc
MIMALLOC_VERSION="3.4.3"
MIMALLOC_SHA256="7b043c26b64dde66344d144fbcac3664858a4eb32197793ea2a18feb32741ca1"
if [ ! -d "${SOURCE_DIR}" ] ; then
  ARCHIVE="$(dirname "${SOURCE_DIR}")/mimalloc-${MIMALLOC_VERSION}.tar.gz"
  curl -fsSL -o "${ARCHIVE}" https://github.com/microsoft/mimalloc/archive/refs/tags/v${MIMALLOC_VERSION}.tar.gz
  echo "${MIMALLOC_SHA256}  ${ARCHIVE}" | sha256sum -c -
  mkdir -p "${SOURCE_DIR}"
  tar -xzf "${ARCHIVE}" -C "${SOURCE_DIR}" --strip-components=1
  rm "${ARCHIVE}"
fi

CMAKE_OPTIONS="${CMAKE_OPTIONS} \
  -D MI_USE_CXX=ON \
  -D MI_BUILD_STATIC=OFF \
  -D MI_BUILD_OBJECT=OFF \
  -D MI_BUILD_TESTS=OFF \
  -D MI_INSTALL_TOPLEVEL=ON \
"

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" ${CMAKE_OPTIONS}
cmake --build "${BUILD_DIR}" -j `nproc`
cmake --install "${BUILD_DIR}"
