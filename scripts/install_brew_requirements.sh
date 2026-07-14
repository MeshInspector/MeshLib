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

brew install --quiet $(echo "$MESHLIB_BREW_REQUIREMENTS" | tr '\n' ' ')
# FIXME: build w/o pybind11
brew install --quiet pybind11
