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
# Per-file loop, not `find -exec ... +`: strip prints a sig-invalidation warning
# and exits 1 on signed dylibs, which would abort the whole batched invocation
# and leave later files unstripped. codesign re-signs ad-hoc so dyld doesn't
# SIGKILL on load.
BREW_PREFIX=$(brew --prefix)
find "$BREW_PREFIX/Cellar" -type f -name '*.dylib' -print0 | while IFS= read -r -d '' f; do
  chmod u+w "$f"
  strip -x "$f" 2>/dev/null || true
  codesign --force --sign - "$f" 2>/dev/null || true
done

# check and upgrade python3 pip
python3.10 -m ensurepip --upgrade
python3.10 -m pip install --upgrade pip

# install requirements for python libs
python3.10 -m pip install -r requirements/python/requirements.txt

exit 0
