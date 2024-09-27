#!/bin/bash

# This script creates `*.deb` packages with built thirdparty and project libs
# usage: first argument - `*-dev.deb` package name, `*.deb` package name
# ./distribution.sh /path/to/file/meshlib.deb

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

# create distr dirs
if [ -d "./distr/" ]; then
  rm -rf distr
fi

cmake --install build/Release/ --prefix "./distr/meshlib-dev/usr/local"

MR_INSTALL_INCLUDE_DIR="/usr/local/include/MeshLib"
MR_INSTALL_LIB_DIR="/usr/local/lib/MeshLib"
MR_INSTALL_RES_DIR="/usr/local/share/MeshLib"

mkdir -p distr/meshlib-dev${MR_INSTALL_RES_DIR}
MR_VERSION="0.0.0.0"
if [ "${1}" ]; then
  MR_VERSION="${1:1}"
fi
echo ${MR_VERSION} > distr/meshlib-dev${MR_INSTALL_RES_DIR}/mr.version

# create control file
BASEDIR=$(dirname "$0")
requirements_file="$BASEDIR"/../requirements/ubuntu.txt
# convert multi-line file to comma-separated string
DEPENDS_LINE=$(cat $requirements_file | tr '\n' ',' | sed -e "s/,$//" -e "s/,/, /g")

mkdir -p distr/meshlib-dev/DEBIAN
CONTROL_FILE="./distr/meshlib-dev/DEBIAN/control"
cat <<EOF > ${CONTROL_FILE}
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

# copy lib dir
cp -rL ./include "./distr/meshlib-dev${MR_INSTALL_INCLUDE_DIR}/"
cp -rL ./lib "./distr/meshlib-dev${MR_INSTALL_LIB_DIR}/"
printf "Thirdparty libs and include copy done\n"

# call dpkg
cd distr
dpkg-deb --build ./meshlib-dev

if [ -f "./meshlib-dev.deb" ]; then
 echo "Dev deb package has been built."
else
 echo "Failed to build dev.deb package!"
 exit 8
fi
