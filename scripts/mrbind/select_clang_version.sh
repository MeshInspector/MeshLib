#!/bin/bash

# Returns the version of MSYS2 we want to use on each platform.

set -euo pipefail

# Windows, Clang installed via MSYS2.
if [[ $(uname -o) == Msys ]]; then
    echo "20.1.7-1"
    exit
fi

UNAME_S=$(uname -s)

# MacOS.
if [[ $UNAME_S == Darwin ]]; then
    echo "20"
    exit
fi

# Linux
if [[ $UNAME_S == Linux ]]; then
    # Here we use `type` to not rely on `which` existing, since it's getting removed from some distros.
    if ! type lsb_release >/dev/null 2>/dev/null; then
        echo "`lsb_release` is not installed!" >&2
    fi

    # Is `Ubuntu <= 22.04`?
    if ! (lsb_release -rs; echo "22.04") | sort -CV; then
        # Here we need the outdated Clang because the old Boost doesn't compile on the new Clang, because of this change: https://github.com/llvm/llvm-project/issues/59036
        echo "18"
        exit
    fi

    # Any other linux
    echo "20"
    exit
fi

echo "Unknown OS" >&2
exit 1
