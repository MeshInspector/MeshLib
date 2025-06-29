#!/bin/bash

# Installs everything needed to generate and build MRBind bindings.
# We assume `brew` is already installed. Automatic its installation is too much,
#   especially because of the conflicts that happen if several users install it.

SCRIPT_DIR="$(dirname "$BASH_SOURCE")"
CLANG_VER="$("$SCRIPT_DIR/select_clang_version.sh")"
[[ $CLANG_VER ]] || (echo "Not sure what version of Clang to use." && false)

brew update
brew install make gawk grep llvm@$CLANG_VER
