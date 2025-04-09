#!/bin/bash
set -euxo pipefail

# Part 1 (uses sudo)
# This installs all system pkgs

# Load the list of Python versions. `xargs` trims the whitespace and removes newlines.
SCRIPT_DIR="$(realpath "$(dirname "$BASH_SOURCE")")"
PY_VERSIONS="$(cat $SCRIPT_DIR/python_versions.txt | xargs)"

echo "Python versions: $PY_VERSIONS"

# Add the PPA.
sudo apt -y update && sudo apt -y upgrade && sudo apt -y install software-properties-common curl
sudo add-apt-repository -y ppa:deadsnakes/ppa

# Install the packages.
sudo apt -y install $(echo $PY_VERSIONS | perl -pe 's/(\S+)/python\1-dev python\1-venv/g')

# Install distutils (needed for pip), but only for Python versions for which it exists.
for ver in $PY_VERSIONS; do
    if apt show python$ver-distutils >/dev/null 2>/dev/null; then
        sudo apt -y install python$ver-distutils
    fi
done
