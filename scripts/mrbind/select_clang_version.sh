#!/bin/bash

# Returns the version of Clang that we want to use on the current platform.
# If you also pass `--flags`, we return the extra flags it needs.

set -euo pipefail

if [[ $# > 0 ]]; then
    echo 'Expected no arguments.' >&2
    exit
fi


CLANG_18=18
CLANG_20=20
CLANG_MSYS2=20.1.7-1


# Windows, Clang installed via MSYS2.
if [[ $(uname -o) == Msys ]]; then
    echo $CLANG_MSYS2
    exit
fi

UNAME_S=$(uname -s)

# MacOS.
if [[ $UNAME_S == Darwin ]]; then
    # Clang 20 can't find the C++ standard library, so staying 18 for now.
    # I didn't dig too deep here, and we have to maintain 18 anyway for Ubuntu 20.04 and 22.04 (see below for why that is).
    echo 18
    exit
fi

# Linux.
if [[ $UNAME_S == Linux ]]; then
    # Here we use `type` to not rely on `which` existing, since it's getting removed from some distros.
    if ! type lsb_release >/dev/null 2>/dev/null; then
        echo "`lsb_release` is not installed!" >&2
    fi

    # Is `Ubuntu <= 22.04`?
    if (lsb_release -rs; echo "22.04") | sort -CV; then
        # Here we need the outdated Clang because the old Boost doesn't compile on the new Clang, because of this change: https://github.com/llvm/llvm-project/issues/59036
        # This is what is actually locking us to Clang 18 at the moment. Other platforms are using it for less scare reasons.
        echo 18
        exit
    fi

    # Is any other ubuntu?
    if [[ $(lsb_release -is) == Ubuntu ]]; then
        # This is because teh stock spdlog and libfmt fail with `call to consteval function ... is not a constant expression` somewhere in the formatting logic. Hmm.
        echo 18
        exit
    fi

    # Any other linux.
    echo 20
    exit
fi

echo "Unknown OS" >&2
exit 1
