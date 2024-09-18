#/bin/bash

# Downloads the source code for MRBind and builds it at `~/mrbind/build`.

set -euxo pipefail

MRBIND_DIR=~/mrbind

# Read the Clang version from `preferred_clang_version.txt`. `xargs` trims the whitespace.
SCRIPT_DIR="$(realpath "$(dirname "$BASH_SOURCE")")"
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
    git clone https://github.com/MeshInspector/mrbind "$MRBIND_DIR"
    cd "$MRBIND_DIR"
fi

rm -rf build

# Add `make` to PATH.
export PATH="/opt/homebrew/opt/make/libexec/gnubin:$PATH"
# Add Clang to PATH.
export PATH="/opt/homebrew/opt/llvm@$CLANG_VER/bin:$PATH"

CC=clang CXX=clang++ cmake -B build
cmake --build build -j4
