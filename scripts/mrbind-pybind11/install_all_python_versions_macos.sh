#!/bin/bash

# Most of this wizardry is copied from our previous pip-build file, not sure why some of this is needed.

set -euxo pipefail

# Load the list of Python versions. `xargs` trims the whitespace and removes newlines.
SCRIPT_DIR="$(realpath "$(dirname "$BASH_SOURCE")")"
PY_VERSIONS="$(cat $SCRIPT_DIR/python_versions.txt | xargs)"

# Should be `/opt/homebrew` on ARM and `/usr/local` on x86.
[[ ${HOMEBREW_DIR:=} ]] || HOMEBREW_DIR="$(brew --prefix)"

if [[ ${ENABLE_SUDO:=} == 1 ]]; then
    SUDO=sudo
elif [[ ${ENABLE_SUDO:=} == 0 ]]; then
    SUDO=
elif which sudo >/dev/null 2>/dev/null; then
    SUDO=sudo
else
    SUDO=
fi

# ??
$SUDO find "$HOMEBREW_DIR/bin" -lname '*/Library/Frameworks/Python.framework/*' -delete

if [[ ${ALLOW_DELETING_EXISTING_PYTHON:=} == 1 ]]; then
    $SUDO rm -rf /Library/Frameworks/Python.framework/
elif [[ -d /Library/Frameworks/Python.framework/ ]]; then
    echo "WARNING: You have a potentially conflicting Python installation at: /Library/Frameworks/Python.framework/"
    echo "  Consider deleting it using the following command: sudo rm -rf /Library/Frameworks/Python.framework/"
fi

brew update

for ver in $PY_VERSIONS; do
    if [[ $ver == 3.8 ]]; then
        # ($HOMEBREW_DIR == /usr/local) for mac x64 and (/opt/homebrew == /opt/homebrew) for mac Arm
        # python 3.8 disabled on macOS since 2024-10-14 (according to our old pip-build file)
        continue
    fi

    FORMULA="python@$ver"
    # Homebrew stopped bottling for Intel macOS (announced 2026-08), and a formula rebuilt
    # after that loses its existing x86_64 bottle, so it is built from source and its
    # `post_install` then fails, failing this whole script. Pin such formulae to the last
    # revision that still has a bottle; the bottles themselves stay available.
    if [[ $(uname -m) == x86_64 ]]; then
        case $ver in
            3.12) FORMULA='https://raw.githubusercontent.com/Homebrew/homebrew-core/7b2c2f97093e29530dde60cf8bfd84f2ef2586a1/Formula/p/python%403.12.rb' ;;
        esac
    fi

    # ??
    # Note that Brew doesn't want to be ran in `sudo`.
    brew install "$FORMULA"
    brew unlink python@$ver
    brew link --overwrite python@$ver
done
