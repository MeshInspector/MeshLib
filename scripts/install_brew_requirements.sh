#!/bin/bash

# This script installs requirements via `brew bundle` if not already
# installed. `brew bundle` skips already-installed-and-up-to-date formulae
# without emitting the GitHub Actions `::warning::`/`::error::` workflow
# commands that `brew install` does, keeping CI run summaries clean.

BASEDIR=$(dirname $(realpath "$0"))
{
  awk '/^[^[:space:]#]/{print "brew \""$1"\""}' "$BASEDIR"/../requirements/macos.txt
  echo 'brew "pybind11"'
  if [ -n "$MESHLIB_EXTRA_BREW_REQUIREMENTS" ] ; then
    echo "$MESHLIB_EXTRA_BREW_REQUIREMENTS" | awk '/^[^[:space:]#]/{print "brew \""$1"\""}'
  fi
} | brew bundle install --no-upgrade --file=-

# check and upgrade python3 pip
python3.10 -m ensurepip --upgrade
python3.10 -m pip install --upgrade pip

# install requirements for python libs
python3.10 -m pip install -r requirements/python.txt

exit 0
