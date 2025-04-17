#!/bin/bash
set -euxo pipefail

# Part 2 (wihtout sudo)
# This installs all Python versions listed in `python_versions.txt`.

# Load the list of Python versions. `xargs` trims the whitespace and removes newlines.
SCRIPT_DIR="$(realpath "$(dirname "$BASH_SOURCE")")"
PY_VERSIONS="$(cat $SCRIPT_DIR/python_versions.txt | xargs)"

echo "Python versions: $PY_VERSIONS"

# Install the dependencies.
for ver in $PY_VERSIONS; do
    curl -sS https://bootstrap.pypa.io/get-pip.py | python$ver
    python$ver -m pip install --upgrade -r ./requirements/python.txt
    python$ver -m pip install pytest
done
