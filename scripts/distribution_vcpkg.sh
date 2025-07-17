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
cp -a ${VCPKG_ROOT}/installed/${VCPKG_TRIPLET}/* ${DISTR_DIR}/
# install MeshLib files
cmake --install ./build/Release --prefix ${DISTR_DIR}
# create tar.xz file
tar --create --use-compress-program='xz -9 -T0' --file=meshlib_linux-vcpkg.tar.xz --directory=${DISTR_DIR} .

rm -rf ${DISTR_DIR}
