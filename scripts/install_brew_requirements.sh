#!/bin/bash

# This script installs requirements by `brew` if not already installed

BASEDIR=$(dirname $(realpath "$0"))
MESHLIB_BREW_REQUIREMENTS=$(cat "$BASEDIR"/../requirements/macos.txt)
if [ -n "$MESHLIB_EXTRA_BREW_REQUIREMENTS" ] ; then
  MESHLIB_BREW_REQUIREMENTS=$MESHLIB_BREW_REQUIREMENTS$'\n'$MESHLIB_EXTRA_BREW_REQUIREMENTS
fi

# Pin the libs that link into the pip wheel to monterey-tagged bottles so they
# match CMAKE_OSX_DEPLOYMENT_TARGET=12.0 (default Sequoia bottles target 14).
# brew install doesn't accept --bottle-tag, so we fetch the cross-tag bottle
# into brew's cache and install from there.
PINNED_FORMULAS="jsoncpp openvdb opencascade tbb"
if [ "$(uname -m)" = "arm64" ]; then
  BOTTLE_TAG="arm64_monterey"
else
  BOTTLE_TAG="monterey"
fi
export MACOSX_DEPLOYMENT_TARGET=12.0
brew fetch --force-bottle --bottle-tag="$BOTTLE_TAG" $PINNED_FORMULAS
for f in $PINNED_FORMULAS; do
  brew install --quiet --force-bottle "$(brew --cache --bottle-tag="$BOTTLE_TAG" "$f")"
done

brew install --quiet $(echo "$MESHLIB_BREW_REQUIREMENTS" | tr '\n' ' ')

brew install --quiet pybind11

# Strip dylibs we'll bundle (brew keeps full symbol tables for symbolication).
BREW_PREFIX=$(brew --prefix)
find "$BREW_PREFIX/lib" -type f -name '*.dylib' -not -type l \
  -exec chmod u+w {} + -exec strip -x {} + 2>/dev/null || true

# check and upgrade python3 pip
python3.10 -m ensurepip --upgrade
python3.10 -m pip install --upgrade pip

# install requirements for python libs
python3.10 -m pip install -r requirements/python.txt

exit 0
