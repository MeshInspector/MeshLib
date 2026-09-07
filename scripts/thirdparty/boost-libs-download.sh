#!/bin/bash
set -eo pipefail

SOURCE_DIR="$1"

SCRIPT_DIR="$(realpath "$(dirname "$BASH_SOURCE")")"

# Boost libraries are so large it's unreasonable to download them via git
BOOST_VERSION="1.83.0"
if [ ! -d "${SOURCE_DIR}" ] ; then
  cd $(dirname "${SOURCE_DIR}")
  curl -L https://github.com/boostorg/boost/releases/download/boost-${BOOST_VERSION}/boost-${BOOST_VERSION}.tar.xz | tar xJ
  mv boost-${BOOST_VERSION} $(basename "${SOURCE_DIR}")

  # to support the single-threaded version, we must compile Boost.Locale without Boost.Thread
  cd "${SOURCE_DIR}/libs/locale/"
  sed -i '/Boost::thread/d' CMakeLists.txt
  patch -Np1 -i "${SCRIPT_DIR}/boost-locale-std-mutex.patch"
fi
