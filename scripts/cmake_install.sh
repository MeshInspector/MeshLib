#!/bin/bash

# exit if any command failed
set -eo pipefail

# NOTE: realpath is not supported on older macOS versions
BASE_DIR=$( cd "$( dirname "$0" )"/.. ; pwd -P )

INSTALL_DIR=$1
if [ -z "${INSTALL_DIR}" ] ; then
    echo "Usage: scripts/install.sh INSTALL_DIR" >&2
    exit 1
fi

cmake_install () {
  if [ -d "$1" ] ; then
    cmake --install "$1" --prefix "${INSTALL_DIR}"
  fi
}

cmake_install ${BASE_DIR}/build/Release
cmake_install ${BASE_DIR}/thirdparty_build
# Emscripten dependencies
cmake_install ${BASE_DIR}/thirdparty_build/libE57Format_build
cmake_install ${BASE_DIR}/thirdparty_build/libjpeg-turbo_build
cmake_install ${BASE_DIR}/thirdparty_build/openvdb_build
