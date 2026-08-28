#!/bin/bash

# Installs everything needed to generate and build MRBind bindings.
# We assume `brew` is already installed. Automatic its installation is too much,
#   especially because of the conflicts that happen if several users install it.

brew update
brew install --quiet make grep

# `lld` links the bindings much faster than Apple's `ld`, but Homebrew has no bottle of it
# for Intel macOS since that became a Tier 3 configuration, and building it from source
# means building LLVM. Install it where it is bottled; `generate.mk` detects its absence.
if [[ "$(uname -m)" == "arm64" ]] ; then
  brew install --quiet lld
fi

if [[ -z "${LLVM_PREFIX}" ]] ; then
  # Read the Clang version from `clang_version_macos.txt`. `xargs` trims the whitespace.
  # Some versions of MacOS seem to lack `realpath`, so not using it here.
  SCRIPT_DIR="$(dirname "$BASH_SOURCE")"
  CLANG_VER="$(cat $SCRIPT_DIR/clang_version_macos.txt | xargs)"
  [[ $CLANG_VER ]] || (echo "Not sure what version of Clang to use." && false)

  brew install --quiet llvm@$CLANG_VER
fi
