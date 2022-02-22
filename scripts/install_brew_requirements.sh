#!/bin/bash

# run from repo root
# NODE: using this script is deprecated! Better install meshrus(-dev).rpm package
# This script installs requirements by `dnf` if not already installed
# `distribution.sh` uses this script as preinstall

requirements_file=requirements/macos.txt
for req in `cat $requirements_file`
do
  echo "brew \"${req}\"" >> requirements/Brewfile
done

brew bundle install --file=requirements/Brewfile

# check and upgrade python3 pip
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip

# install requirements for python libs
python3 -m pip install -r requirements/python.txt

exit 0
