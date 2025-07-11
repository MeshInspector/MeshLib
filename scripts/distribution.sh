#!/bin/bash

# This script creates `*.deb` packages with built thirdparty and project libs
# usage: first argument - `v1.2.3.4` - version with "v" prefix,
# ./distribution.sh v1.2.3.4

# exit if any command failed
set -eo pipefail

if [ ! -f "./lib/libcpr.so" ]; then
  echo "Thirdparty build was not found. Building..."
  ./scripts/build_thirdparty.sh
fi

if [ ! -f "./build/Release/bin/libMRMesh.so" ]; then
  echo "Project release build was not found. Building..."
  export MESHLIB_BUILD_RELEASE="ON"
  export MESHLIB_BUILD_DEBUG="OFF"
  ./scripts/build_source.sh
fi

#create distr dirs
if [ -d "./distr/" ]; then
 rm -rf distr
fi

cmake --install ./build/Release --prefix "./distr/meshlib-dev/usr/local"

MR_INSTALL_LIB_DIR="/usr/local/lib/MeshLib"
MR_INSTALL_INCLUDE_DIR="/usr/local/include/MeshLib"
MR_INSTALL_RES_DIR="/usr/local/share/MeshLib"

# Install the generated bindings, if needed.
if [ ! -f "distr/meshlib-dev$MR_INSTALL_LIB_DIR/meshlib/mrmeshpy.so" ] && [ -f "build/Release/bin/meshlib/mrmeshpy.so" ]; then
  echo "Installing the generated bindings..."
  install -Dt "distr/meshlib-dev$MR_INSTALL_LIB_DIR/meshlib" build/Release/bin/meshlib/{mrmeshpy.so,mrmeshnumpy.so,__init__.py}
  install -Dt "distr/meshlib-dev$MR_INSTALL_LIB_DIR"         build/Release/bin/meshlib/{mrmeshpy.so,mrmeshnumpy.so,__init__.py}
  patchelf --set-rpath '' "distr/meshlib-dev$MR_INSTALL_LIB_DIR/"{,meshlib/}mrmeshpy.so

  if [ -f "build/Release/bin/meshlib/mrcudapy.so" ]; then
    echo "CUDA bindings found, installing with mrcudapy.so..."
    install -Dt "distr/meshlib-dev$MR_INSTALL_LIB_DIR/meshlib" build/Release/bin/meshlib/mrcudapy.so
    install -Dt "distr/meshlib-dev$MR_INSTALL_LIB_DIR"         build/Release/bin/meshlib/mrcudapy.so
    patchelf --set-rpath '' "distr/meshlib-dev$MR_INSTALL_LIB_DIR/"{,meshlib/}mrcudapy.so
  fi
fi

MR_VERSION="0.0.0.0"
if [ "${1}" ]; then
  MR_VERSION="${1:1}"
fi
echo ${MR_VERSION} > distr/meshlib-dev${MR_INSTALL_RES_DIR}/mr.version

BASE_DIR=$(realpath $(dirname "$0")/..)
REQUIREMENTS_FILE="${BASE_DIR}/requirements/ubuntu.txt"
# convert multi-line file to comma-separated string
DEPENDS_LINE=$(cat ${REQUIREMENTS_FILE} | tr '\n' ',' | sed -e "s/,\s*$//" -e "s/,/, /g")

#create control file
mkdir -p distr/meshlib-dev/DEBIAN
cat << EOF > ./distr/meshlib-dev/DEBIAN/control
Package: meshlib-dev
Essential: no
Priority: optional
Section: model
Maintainer: Adalisk team
Architecture: all
Description: Advanced mesh modeling library
Version: ${MR_VERSION}
Depends: ${DEPENDS_LINE}
EOF

cp "./scripts/preinstall_trick.sh" ./distr/meshlib-dev/DEBIAN/preinst
chmod +x ./distr/meshlib-dev/DEBIAN/preinst

cp "./scripts/postinstall.sh" ./distr/meshlib-dev/DEBIAN/postinst
chmod +x ./distr/meshlib-dev/DEBIAN/postinst

mkdir -p ./distr/meshlib-dev/usr/local/lib/udev/rules.d/
cp "./scripts/70-space-mouse-meshlib.rules" ./distr/meshlib-dev/usr/local/lib/udev/rules.d/

#copy lib dir
CURRENT_DIR="`pwd`"
cp -rL ./lib "${CURRENT_DIR}/distr/meshlib-dev${MR_INSTALL_LIB_DIR}/"
cp -rL ./include "${CURRENT_DIR}/distr/meshlib-dev${MR_INSTALL_INCLUDE_DIR}/"
echo "Thirdparty libs and include copy done"

#call dpkg
cd distr
dpkg-deb --build -Zxz ./meshlib-dev

if [ -f "./meshlib-dev.deb" ]; then
  echo "Dev deb package has been built."
else
  echo "Failed to build dev.deb package!"
  exit 8
fi
