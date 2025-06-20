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
    if [[ $ver == 3.8 ]]; then
        curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python$ver
    else
        curl -sS https://bootstrap.pypa.io/get-pip.py | python$ver
    fi
    python$ver -m pip install --upgrade -r ./requirements/python.txt
    python$ver -m pip install pytest
done
