#!/bin/bash

# This script installs requirements by `brew` if not already installed

BASEDIR=$(dirname $(realpath "$0"))
MESHLIB_BREW_REQUIREMENTS=$(cat "$BASEDIR"/../requirements/macos.txt)
if [ -n "$MESHLIB_EXTRA_BREW_REQUIREMENTS" ] ; then
  MESHLIB_BREW_REQUIREMENTS=$MESHLIB_BREW_REQUIREMENTS$'\n'$MESHLIB_EXTRA_BREW_REQUIREMENTS
fi

brew install --quiet $(echo "$MESHLIB_BREW_REQUIREMENTS" | tr '\n' ' ')

brew install --quiet pybind11

# Strip dylibs we'll bundle (brew keeps full symbol tables for symbolication).
# Walk Cellar/ directly because $BREW_PREFIX/lib contains only symlinks into it.
BREW_PREFIX=$(brew --prefix)
find "$BREW_PREFIX/Cellar" -type f -name '*.dylib' \
  -exec chmod u+w {} + -exec strip -x {} + 2>/dev/null || true

# check and upgrade python3 pip
python3.10 -m ensurepip --upgrade
python3.10 -m pip install --upgrade pip

# install requirements for python libs
python3.10 -m pip install -r requirements/python.txt

exit 0
