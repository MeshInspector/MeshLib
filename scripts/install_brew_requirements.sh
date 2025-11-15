#!/bin/bash

# run from repo root
# NODE: using this script is deprecated! Better install meshlib(-dev).pkg package
# This script installs requirements by `brew` if not already installed

requirements_file=requirements/macos.txt
brew install $(cat $requirements_file | tr '\n' ' ')

brew install pybind11

# check and upgrade python3 pip
python3.10 -m ensurepip --upgrade
python3.10 -m pip install --upgrade pip

# install requirements for python libs
python3.10 -m pip install -r requirements/python.txt

exit 0
