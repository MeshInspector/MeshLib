#!/bin/bash

# This script creates `*.tar.xz` packages with built thirdparty and project libs
# usage: first argument - `v1.2.3.4` - version with "v" prefix
# ./distribution_vcpkg.sh v1.2.3.4

# exit if any command failed
set -eo pipefail

if [ ! -f "./build/Release/bin/libMRMesh.so" ]; then
  echo "Project release build was not found. Building..."
  export MESHLIB_BUILD_RELEASE="ON"
  export MESHLIB_BUILD_DEBUG="OFF"
  ./scripts/build_source.sh
fi

#modify mr.version
version=0.0.0.0
if [ ${1} ]; then
  version=${1:1} #v1.2.3.4 -> 1.2.3.4
fi
echo $version > build/Release/bin/mr.version

# create distr dirs
DISTR_DIR=./vcpkg-distr
if [ -d ${DISTR_DIR} ]; then
  rm -rf ${DISTR_DIR}
fi

mkdir ${DISTR_DIR}
# copy vcpkg files
VCPKG_INSTALLED_DIR=${VCPKG_INSTALLED_DIR:=${VCPKG_ROOT}/installed}
cp -a ${VCPKG_INSTALLED_DIR}/${VCPKG_TRIPLET}/* ${DISTR_DIR}/
# install MeshLib files
cmake --install ./build/Release --prefix ${DISTR_DIR} --strip
# mrbind's modules are built outside CMake, so `cmake --install` above misses them.
# lib/MeshLib/meshlib is MR_PY_LIB_DIR for the vcpkg layout; the rpath generate.mk
# baked in already resolves there via $ORIGIN/.. so it is left alone.
PY_LIB_DIR="${DISTR_DIR}/lib/MeshLib/meshlib"
if [ -f build/Release/bin/meshlib/mrmeshpy.so ] ; then
  echo "Installing the generated bindings..."
  install -Dt "${PY_LIB_DIR}" build/Release/bin/meshlib/__init__.py
  install -sDt "${PY_LIB_DIR}" build/Release/bin/meshlib/mrmeshpy.so
  if [ -f build/Release/bin/meshlib/mrcudapy.so ] ; then
    install -sDt "${PY_LIB_DIR}" build/Release/bin/meshlib/mrcudapy.so
  fi
else
  echo "WARNING: build/Release/bin/meshlib/mrmeshpy.so not found; the archive will have no Python bindings"
fi

# create tar.xz file
tar --create --use-compress-program='xz -9 -T0' --file=meshlib_linux-vcpkg.tar.xz --directory=${DISTR_DIR} .

rm -rf ${DISTR_DIR}
