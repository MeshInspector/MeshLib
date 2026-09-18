#!/bin/bash
set -eo pipefail

SOURCE_DIR="$1"
BUILD_DIR="${2:-./openvdb_build}"

CXXFLAGS="${CXXFLAGS} -Wno-missing-template-arg-list-after-template-kw"

# a wasm .so is a side module, so its link wants every object PIC - including the zlib
# emsdk ships, which is not ours to rebuild. Nothing on wasm needs the shared library.
if [ "${MR_EMSCRIPTEN}" == "ON" ] ; then
  OPENVDB_SHARED=OFF
  OPENVDB_STATIC=ON
else
  OPENVDB_SHARED=ON
  OPENVDB_STATIC=OFF
fi

CMAKE_OPTIONS="${CMAKE_OPTIONS} \
  -D OPENVDB_ENABLE_UNINSTALL=OFF \
  -D OPENVDB_ENABLE_INSTALL=OFF \
  -D OPENVDB_CORE_SHARED=${OPENVDB_SHARED} \
  -D OPENVDB_CORE_STATIC=${OPENVDB_STATIC} \
  -D OPENVDB_BUILD_BINARIES=OFF \
  -D OPENVDB_BUILD_VDB_PRINT=OFF \
  -D OPENVDB_USE_DELAYED_LOADING=OFF \
  -D USE_EXPLICIT_INSTANTIATION=OFF \
  -D Tbb_VERSION=2021.12 \
"

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" -D CMAKE_C_FLAGS="${CFLAGS}" ${CMAKE_OPTIONS}
cmake --build "${BUILD_DIR}" -j `nproc`
cmake --install "${BUILD_DIR}"
