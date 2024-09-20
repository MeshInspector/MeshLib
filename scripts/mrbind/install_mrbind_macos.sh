#!/bin/bash

# Downloads the source code for MRBind and builds it at `~/mrbind/build`.

set -euxo pipefail

MRBIND_DIR=~/mrbind

# Read the Clang version from `preferred_clang_version.txt`. `xargs` trims the whitespace.
# Some versions of MacOS seem to lack `realpath`, so not using it here.
SCRIPT_DIR="$(dirname "$BASH_SOURCE")"
CLANG_VER="$(cat $SCRIPT_DIR/preferred_clang_version.txt | xargs)"
[[ $CLANG_VER ]] || (echo "Not sure what version of Clang to use." && false)

# Clone mrbind, or pull the latest version.
# We don't install our own Git for this, because there's an official installer and the Brew package,
#   and I'm unsure what to choose. The user can choose that themselves.
if [[ -d $MRBIND_DIR ]]; then
    cd "$MRBIND_DIR"
    git checkout master
    git pull
else
    mkdir -p "$MRBIND_DIR"
    git clone https://github.com/MeshInspector/mrbind "$MRBIND_DIR"
    cd "$MRBIND_DIR"
fi

rm -rf build

# This seems to be the default location on Arm Macs, while x86 Macs use `/usr/local`.
HOMEBREW_DIR=/opt/homebrew
[[ -d $HOMEBREW_DIR ]] || HOMEBREW_DIR=/usr/local

# Add `make` to PATH.
export PATH="$HOMEBREW_DIR/opt/make/libexec/gnubin:$PATH"
# Add Clang to PATH.
export PATH="$HOMEBREW_DIR/opt/llvm@$CLANG_VER/bin:$PATH"
ls "$HOMEBREW_DIR/opt" || true
ls "$HOMEBREW_DIR/opt/llvm@$CLANG_VER" || true
ls "$HOMEBREW_DIR/opt/llvm@$CLANG_VER/bin" || true

CC=clang CXX=clang++ cmake -B build
cmake --build build -j4
