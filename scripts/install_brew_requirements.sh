#!/bin/bash

# This script installs requirements by `brew` if not already installed

BASEDIR=$(dirname $(realpath "$0"))
MESHLIB_BREW_REQUIREMENTS=$(cat "$BASEDIR"/../requirements/macos.txt)
if [ -n "$MESHLIB_EXTRA_BREW_REQUIREMENTS" ] ; then
  MESHLIB_BREW_REQUIREMENTS=$MESHLIB_BREW_REQUIREMENTS$'\n'$MESHLIB_EXTRA_BREW_REQUIREMENTS
fi

# Match dylib minos to CMAKE_OSX_DEPLOYMENT_TARGET=12.0; avoids ld64.lld skew at mrbind link.
export MACOSX_DEPLOYMENT_TARGET=12.0
if [ "$(uname -m)" = "arm64" ]; then
  BOTTLE_TAG="arm64_monterey"
else
  BOTTLE_TAG="monterey"
fi

brew install --quiet --bottle-tag="$BOTTLE_TAG" $(echo "$MESHLIB_BREW_REQUIREMENTS" | tr '\n' ' ')

brew install --quiet --bottle-tag="$BOTTLE_TAG" pybind11

# check and upgrade python3 pip
python3.10 -m ensurepip --upgrade
python3.10 -m pip install --upgrade pip

# install requirements for python libs
python3.10 -m pip install -r requirements/python.txt

exit 0
