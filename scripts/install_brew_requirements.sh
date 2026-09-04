#!/bin/bash -i
# expand aliases defined in ~/.bashrc
# this and -i flag may be required for multi-user configurations where brew is declared as an alias for a more complicated command
# some examples:
# - https://dev.to/cerico/using-brew-in-a-multi-user-system-2lnl
# - https://www.codejam.info/2021/11/homebrew-multi-user.html
shopt -s expand_aliases

set -e

# This script installs requirements by `brew` if not already installed

BASEDIR=$(dirname $(realpath "$0"))
MESHLIB_BREW_REQUIREMENTS=$(cat "$BASEDIR"/../requirements/macos.txt)
if [ -n "$MESHLIB_EXTRA_BREW_REQUIREMENTS" ] ; then
  MESHLIB_BREW_REQUIREMENTS=$MESHLIB_BREW_REQUIREMENTS$'\n'$MESHLIB_EXTRA_BREW_REQUIREMENTS
fi


# GitHub-hosted macOS images ship the untrusted aws/tap, which makes every brew command emit a warning
brew untap aws/tap 2>/dev/null || true

# A requirement already installed from another tap cannot be installed from homebrew/core,
# and brew fails the whole command over it. install_all_python_versions_macos.sh pins
# python@X to bottled homebrew-core revisions served from our own tap on Intel macOS,
# where `requirements/macos.txt` asks for python@3.10.
installed_from_other_tap() {
    local receipt
    for receipt in "$(brew --cellar)/$1"/*/INSTALL_RECEIPT.json; do
        [ -f "$receipt" ] || continue
        grep -q '"tap": *"homebrew/core"' "$receipt" || return 0
    done
    return 1
}

REQUIREMENTS=
for formula in $MESHLIB_BREW_REQUIREMENTS; do
    if installed_from_other_tap "$formula"; then
        echo "Keeping the already installed $formula, it does not come from homebrew/core"
    else
        REQUIREMENTS="$REQUIREMENTS $formula"
    fi
done

brew install --quiet $REQUIREMENTS
# FIXME: build w/o pybind11
brew install --quiet pybind11
