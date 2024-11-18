#!/bin/bash

# Installs everything needed to generate and build MRBind bindings.
# We assume `brew` is already installed. Automatic its installation is too much,
#   especially because of the conflicts that happen if several users install it.

# Read the Clang version from `clang_version.txt`. `xargs` trims the whitespace.
# Some versions of MacOS seem to lack `realpath`, so not using it here.
SCRIPT_DIR="$(dirname "$BASH_SOURCE")"
CLANG_VER="$(cat $SCRIPT_DIR/clang_version.txt | xargs)"
[[ $CLANG_VER ]] || (echo "Not sure what version of Clang to use." && false)

brew update
brew install make gawk grep llvm@$CLANG_VER
