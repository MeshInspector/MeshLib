#!/bin/bash

# This script creates `*.deb` packages with built thirdparty and project libs
# usage: first argument - `*-dev.deb` package name, `*.deb` package name
# ./distribution.sh /path/to/file/meshrus.deb

# exit if any command failed
set -eo pipefail

# copy prebuilded libs if they exist
if [ -d "/usr/local/lib/meshrus-thirdparty-lib/lib" ]; then
 cp -r /usr/local/lib/meshrus-thirdparty-lib/lib .
fi

if [ ! -f "./lib/libcpr.so" ]; then
 printf "Thirdparty build was not found. Building...\n"
 ./scripts/build_thirdparty.sh
fi

if [ ! -f "./build/Release/bin/libMRMesh.so" ]; then
 printf "Project release build was not found. Building...\n"
 export MESHRUS_BUILD_RELEASE="ON"
 export MESHRUS_BUILD_DEBUG="OFF"
 ./scripts/build_source.sh
fi

#create distr dirs
if [ -d "./distr/" ]; then
 rm -rf distr
fi

MR_LIB_DIR="lib/"
MR_BIN_DIR="build/Release/bin/"
MR_INSTALL_LIB_DIR="/usr/local/lib/MeshLib/"
MR_INSTALL_RES_DIR="/usr/local/etc/MeshLib/"
MR_INSTALL_THIRDPARTY_INCLUDE_DIR="/usr/local/include/"
MR_INSTALL_INCLUDE_DIR="/usr/local/include/MeshLib/"
PYTHON_DIR="/usr/lib/python3"

mkdir -p distr/meshrus-dev/DEBIAN
mkdir -p "distr/meshrus-dev${MR_INSTALL_LIB_DIR}"
mkdir -p "distr/meshrus-dev${MR_INSTALL_RES_DIR}"
mkdir -p "distr/meshrus-dev${MR_INSTALL_INCLUDE_DIR}"
mkdir -p "distr/meshrus-dev${MR_INSTALL_THIRDPARTY_INCLUDE_DIR}"

if [ ${1} ]; then
  echo ${1:1} > build/Release/bin/mr.version
else
  echo 0.0.0.0 > build/Release/bin/mr.version
fi

#create control file
CONTROL_FILE="./distr/meshrus-dev/DEBIAN/control"
echo "Package: meshrus-dev" > "$CONTROL_FILE"
echo "Essential: no" >> "$CONTROL_FILE"
echo "Priority: optional" >> "$CONTROL_FILE"
echo "Section: model" >> "$CONTROL_FILE"
echo "Maintainer: Adalisk team" >> "$CONTROL_FILE"
echo "Architecture: all" >> "$CONTROL_FILE"
echo "Description: Advanced mesh modeling library" >> "$CONTROL_FILE"
printf "Version: %s\n" `cat build/Release/bin/mr.version` >> "$CONTROL_FILE"
DEPENDS_LINE="Depends:"
req_counter=0
BASEDIR=$(dirname "$0")
requirements_file="$BASEDIR"/../requirements/ubuntu.txt
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

cp "./scripts/preinstall_trick.sh" ./distr/meshrus-dev/DEBIAN/preinst
chmod +x ./distr/meshrus-dev/DEBIAN/preinst

cp "./scripts/postinstall.sh" ./distr/meshrus-dev/DEBIAN/postinst
chmod +x ./distr/meshrus-dev/DEBIAN/postinst

#copy lib dir
CURRENT_DIR="`pwd`"
cd "${MR_LIB_DIR}"
find . -name '*.so*' -type f,l -exec cp -fP \{\} "${CURRENT_DIR}/distr/meshrus-dev${MR_INSTALL_LIB_DIR}" \;
cd -
printf "Thirdparty libs copy done\n"

#copy libs
cp -r build/Release/bin/*.so "distr/meshrus-dev${MR_INSTALL_LIB_DIR}"
printf "MR libs copy done\n"

#copy verison file
cp build/Release/bin/mr.version "distr/meshrus-dev${MR_INSTALL_RES_DIR}"
printf "MR version copy done\n"

#copy headers
cd "${MR_LIB_DIR}"
find . -name '*.h' -type f -exec cp -f --recursive --parents \{\} "${CURRENT_DIR}/distr/meshrus-dev${MR_INSTALL_THIRDPARTY_INCLUDE_DIR}" \;
cd -
cd thirdparty/eigen
find . -name '*.h' -type f -exec cp -f --recursive --parents \{\} "${CURRENT_DIR}/distr/meshrus-dev${MR_INSTALL_THIRDPARTY_INCLUDE_DIR}" \;
cd -
cd source
find . -name '*.h' -type f -exec cp -f --recursive --parents \{\} "${CURRENT_DIR}/distr/meshrus-dev${MR_INSTALL_INCLUDE_DIR}" \;
cd -
printf "Headers copy done\n"

#call dpkg
cd distr
dpkg-deb --build ./meshrus-dev

if [ -f "./meshrus-dev.deb" ]; then
 printf "Dev deb package has been built.\n"
 if [ ${1} ]; then
  cp ./meshrus-dev.deb "${1}"
 fi
else
 printf "Failed to build dev.deb package!"
 exit 8
fi
