#!/bin/bash

# This script creates `*.deb` packages with built thirdparty and project libs
# usage: first argument - `*-dev.deb` package name, `*.deb` package name
# ./distribution.sh /path/to/file/meshrus.deb

# exit if any command failed
set -eo pipefail

if [ ! -f "./lib/libcpr.so" ]; then
 printf "Thirdparty build was not found. Building...\n"
 ./scripts/build_thirdparty.sh
fi

if [ ! -f "./build/Release/bin/libMRMesh.so" ]; then
 printf "Project release build was not found. Building...\n"
 export MESHLIB_BUILD_RELEASE="ON"
 export MESHLIB_BUILD_DEBUG="OFF"
 ./scripts/build_source.sh
fi

#create distr dirs
if [ -d "./distr/" ]; then
 rm -rf distr
fi

cd ./build/Release
cmake --install . --prefix "../../distr/meshlib-dev"
cd -

MR_INSTALL_LIB_DIR="/usr/local/lib/MeshLib"
MR_INSTALL_INCLUDE_DIR="/usr/local/include/MeshLib"
MR_INSTALL_RES_DIR="/usr/local/etc/MeshLib"

mkdir -p distr/meshlib-dev${MR_INSTALL_RES_DIR}
MR_VERSION="0.0.0.0"
if [ "${1}" ]; then
  MR_VERSION="${1:1}"
fi
echo ${MR_VERSION} > distr/meshlib-dev${MR_INSTALL_RES_DIR}/mr.version

#create control file
mkdir -p distr/meshlib-dev/DEBIAN
CONTROL_FILE="./distr/meshlib-dev/DEBIAN/control"
echo "Package: meshlib-dev" > "$CONTROL_FILE"
echo "Essential: no" >> "$CONTROL_FILE"
echo "Priority: optional" >> "$CONTROL_FILE"
echo "Section: model" >> "$CONTROL_FILE"
echo "Maintainer: Adalisk team" >> "$CONTROL_FILE"
echo "Architecture: all" >> "$CONTROL_FILE"
echo "Description: Advanced mesh modeling library" >> "$CONTROL_FILE"
printf "Version: %s\n" $(echo ${MR_VERSION}) >> "$CONTROL_FILE"
DEPENDS_LINE="Depends:"
req_counter=0
BASEDIR=$(dirname "$0")

. /etc/lsb-release
UBUNTU_MAJOR_VERSION=${DISTRIB_RELEASE%.*}

requirements_file="$BASEDIR"/../requirements/ubuntu.txt
if [ "$UBUNTU_MAJOR_VERSION" == "22" ]; then
  requirements_file="$BASEDIR"/../requirements/ubuntu22.txt
fi

for req in `cat $requirements_file`
do
  if [ $req_counter -le 0 ]; then
  	DEPENDS_LINE="${DEPENDS_LINE} ${req}"
  else
  	DEPENDS_LINE="${DEPENDS_LINE}, ${req}"
  fi
  ((req_counter=req_counter+1))
done
echo "${DEPENDS_LINE}" >> "$CONTROL_FILE"

cp "./scripts/preinstall_trick.sh" ./distr/meshlib-dev/DEBIAN/preinst
chmod +x ./distr/meshlib-dev/DEBIAN/preinst

cp "./scripts/postinstall.sh" ./distr/meshlib-dev/DEBIAN/postinst
chmod +x ./distr/meshlib-dev/DEBIAN/postinst

#copy lib dir
CURRENT_DIR="`pwd`"
cp -rL ./lib "${CURRENT_DIR}/distr/meshlib-dev${MR_INSTALL_LIB_DIR}/"
cp -rL ./include "${CURRENT_DIR}/distr/meshlib-dev${MR_INSTALL_INCLUDE_DIR}/"
printf "Thirdparty libs and include copy done\n"

#call dpkg
cd distr
dpkg-deb --build ./meshlib-dev

if [ -f "./meshlib-dev.deb" ]; then
 printf "Dev deb package has been built.\n"
else
 printf "Failed to build dev.deb package!"
 exit 8
fi
