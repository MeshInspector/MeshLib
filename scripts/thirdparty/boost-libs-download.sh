#!/bin/bash
set -eo pipefail

SOURCE_DIR="$1"

# Boost libraries are so large it's unreasonable to download them via git
BOOST_VERSION="1.83.0"
if [ ! -d "${SOURCE_DIR}" ] ; then
  pushd $(dirname "${SOURCE_DIR}")
  curl -L https://github.com/boostorg/boost/releases/download/boost-${BOOST_VERSION}/boost-${BOOST_VERSION}.tar.xz | tar xJ
  mv boost-${BOOST_VERSION} $(basename "${SOURCE_DIR}")
fi
