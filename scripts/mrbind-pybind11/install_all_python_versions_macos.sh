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

# Homebrew stopped bottling for Intel macOS (announced 2026-08), and a formula rebuilt
# after that loses the x86_64 bottles it already had, so it would be built from source
# and its `post_install` then fails. That already happened to python@3.12 on 2026-09-02,
# so pin every version to its newest revision that still has an x86_64 bottle: the
# bottles stay published, and a bottle's ghcr path comes from the formula name, so it is
# poured even from our own tap. Only the pinned formula is frozen, not its dependencies.
PIN_TAP=meshlib/pins
pinned_formula_commit() {
    case $1 in
        3.9)  echo 4e0140a8669ac57c55176fa96c85068e633ba93c ;; # 3.9.25_1
        3.10) echo 7542533c9af6d6f24ff9dd80447f5d2711d7fde6 ;; # 3.10.21
        3.11) echo 345885aec208c8cac4acc5c03457746df5187534 ;; # 3.11.16
        3.12) echo 7b2c2f97093e29530dde60cf8bfd84f2ef2586a1 ;; # 3.12.14, before the 2026-09-02 rebuild
        3.13) echo 6997357d29053aecb8f40eab3f583d209dcea1cc ;; # 3.13.15
        3.14) echo 7187bb7ac10ba3a2661758624147584d24e79d6b ;; # 3.14.7
    esac
}

brew update

for ver in $PY_VERSIONS; do
    if [[ $ver == 3.8 ]]; then
        # ($HOMEBREW_DIR == /usr/local) for mac x64 and (/opt/homebrew == /opt/homebrew) for mac Arm
        # python 3.8 disabled on macOS since 2024-10-14 (according to our old pip-build file)
        continue
    fi

    FORMULA="python@$ver"
    PIN_COMMIT="$(pinned_formula_commit $ver)"
    if [[ $(uname -m) == x86_64 && $PIN_COMMIT ]]; then
        brew tap-new $PIN_TAP --no-git >/dev/null 2>&1 || true
        curl -fsSL -o "$(brew --repo $PIN_TAP)/Formula/python@$ver.rb" "https://raw.githubusercontent.com/Homebrew/homebrew-core/$PIN_COMMIT/Formula/p/python%40$ver.rb"
        FORMULA="$PIN_TAP/python@$ver"
    fi

    # ??
    # Note that Brew doesn't want to be ran in `sudo`.
    brew install "$FORMULA"
    brew unlink "$FORMULA"
    brew link --overwrite "$FORMULA"
done
