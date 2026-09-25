#!/bin/bash

SCRIPT_DIR="$(dirname "$BASH_SOURCE")"

if [ ! -n "$MESHLIB_BUILD_RELEASE" ]; then
  read -t 5 -p "Build MeshLib Release? Press (n) in 5 seconds to cancel (Y/n)" -rsn 1
  echo;
  if [[ $REPLY =~ ^[Nn]$ ]]; then
    MESHLIB_BUILD_RELEASE="OFF"
  else
    MESHLIB_BUILD_RELEASE="ON"
  fi
  echo "Release ${MESHLIB_BUILD_RELEASE}"
fi

if [ ! -n "$MESHLIB_BUILD_DEBUG" ]; then
  read -t 5 -p "Build MeshLib Debug? Press (y) in 5 seconds to build (y/N)" -rsn 1
  echo;
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    MESHLIB_BUILD_DEBUG="ON"
  else
    MESHLIB_BUILD_DEBUG="OFF"
  fi
  echo "Debug ${MESHLIB_BUILD_DEBUG}"
fi

# add env options to cmake
MR_CMAKE_OPTIONS="${MR_CMAKE_OPTIONS:-} \
  -D MESHLIB_USE_VCPKG=ON \
  -D VCPKG_MANIFEST_MODE=ON \
  -D VCPKG_TARGET_TRIPLET=${VCPKG_TRIPLET:-$("$SCRIPT_DIR"/detect_vcpkg_triplet.sh)} \
"

if command -v ninja >/dev/null 2>&1 ; then
  MR_CMAKE_OPTIONS="${MR_CMAKE_OPTIONS} -G Ninja"
fi

if [ "${MR_EMSCRIPTEN}" != "ON" ] ; then
  if [ -n "${CMAKE_C_COMPILER}" ] ; then
    MR_CMAKE_OPTIONS="${MR_CMAKE_OPTIONS} -D CMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
  fi
  if [ -n "${CMAKE_CXX_COMPILER}" ] ; then
    MR_CMAKE_OPTIONS="${MR_CMAKE_OPTIONS} -D CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
  fi
  if [ -n "${CMAKE_LINKER_TYPE}" ] ; then
    MR_CMAKE_OPTIONS="${MR_CMAKE_OPTIONS} -D CMAKE_LINKER_TYPE=${CMAKE_LINKER_TYPE}"
  fi
fi

if [[ $OSTYPE == 'darwin'* ]]; then
  NPROC=$(sysctl -n hw.logicalcpu)
else
  NPROC=$(nproc)
fi
echo "The number of concurrent build threads NPROC=${NPROC}"

# exit if any command failed
set -eo pipefail

# build MeshLib
if [ "${MESHLIB_KEEP_BUILD}" != "ON" ]; then
  rm -rf ./build
fi

# build Release
if [ "${MESHLIB_BUILD_RELEASE}" = "ON" ]; then
  if [ "${MESHLIB_KEEP_BUILD}" != "ON" ]; then
    mkdir -p build/Release
  fi
  cd build/Release
    cmake -S ../.. -B . -D CMAKE_BUILD_TYPE=Release ${MR_CMAKE_OPTIONS} $@
    cmake --build . -j ${NPROC} ${MR_CMAKE_BUILD_OPTIONS}
  cd ../..
fi

# build Debug
if [ "${MESHLIB_BUILD_DEBUG}" = "ON" ]; then
  if [ "${MESHLIB_KEEP_BUILD}" != "ON" ]; then
    mkdir -p build/Debug
  fi
  cd build/Debug
    cmake -S ../.. -B . -D CMAKE_BUILD_TYPE=Debug ${MR_CMAKE_OPTIONS} $@
    cmake --build . -j ${NPROC} ${MR_CMAKE_BUILD_OPTIONS}
  cd ../..
fi

if [ "${MESHLIB_BUILD_RELEASE}" = "ON" ]; then
  printf "\rBuild script successfully finished. You could run ./build/Release/bin/MRTest next\n\n"
else
  if [ "${MESHLIB_BUILD_DEBUG}" = "ON" ]; then
    printf "\rBuild script successfully finished. You could run ./build/Debug/bin/MRTest next\n\n"
  else
    printf "\rNothing was built\n\n"
  fi
fi
