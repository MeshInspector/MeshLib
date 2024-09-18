#!/bin/bash

set -euxo pipefail

# Installs everything needed to generate and build MRBind bindings.

CLANG_VER=18

apt update

# Add LLVM repositories if the required package is not accessible right now.
# If it's accessible, either we have already added the same repos,
if ! apt-cache show clang-18 >/dev/null 2>/dev/null; then
    # This is what `llvm.sh` needs (search `apt install` in it and see for yourself).
    apt install -y lsb-release wget software-properties-common gnupg
    # Download `llvm.sh`.
    DIR="$(mktemp -d)"
    pushd $DIR
    wget https://apt.llvm.org/llvm.sh
    chmod +x llvm.sh
    ./llvm.sh "$CLANG_VER"
    popd
    rm -rf "$DIR"
fi

# Install the packages.
apt install -y make cmake ninja-build gawk clang-$VER lld-$VER clang-tools-$VER libclang-$VER-dev llvm-$VER-dev

# Build Make from source, if ours is too old.
# Not exactly sure what versions are good for our purposes. I know that 4.3 is ok and 3.81 isn't.
MIN_MAKE_VER=4.3
MAKE_VER="$(make --version | grep -m1 -Po '(?<=GNU Make ).*')"

if ! printf '%s\n' "$MIN_MAKE_VER" "$MAKE_VER" | sort -CV; then
    apt install -y wget tar gcc

    # Attempt to figure out the most recent available GNU Make.
    LATEST_MAKE_VER="$(wget https://ftpmirror.gnu.org/make/ -O - 2>/dev/null | grep -Po '(?<=<a href=")make-[0-9\.]*(?=\.tar.gz")' | sort -rV | head -1)"
    [[ $LATEST_MAKE_VER ]] || (echo "Can't determine the version of Make to download!" && exit 1)

    DIR="$(mktemp -d)"
    pushd $DIR

    wget "https://ftpmirror.gnu.org/make/$LATEST_MAKE_VER.tar.gz"
    tar -xf "$LATEST_MAKE_VER.tar.gz"
    cd "$LATEST_MAKE_VER"
    ./configure
    make -j4
    make install

    popd
    rm -rf "$DIR"
fi
