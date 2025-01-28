#!/bin/bash

# Most of this wizardry is copied from our previous pip-build file, not sure why some of this is needed.

set -euxo pipefail

# Load the list of Python versions. `xargs` trims the whitespace and removes newlines.
SCRIPT_DIR="$(realpath "$(dirname "$BASH_SOURCE")")"
PY_VERSIONS="$(cat $SCRIPT_DIR/python_versions.txt | xargs)"

# Should be `/opt/homebrew` on ARM and `/usr/local` on x86.
[[ ${HOMEBREW_DIR:=} ]] || HOMEBREW_DIR="$(brew --prefix)"

brew update

for ver in $PY_VERSIONS; do
    if [[ $HOMEBREW_DIR == /usr/local && $ver == 3.8 ]]; then
        # python 3.8 disabled on x86 macOS since 2024-10-14 (according to our old pip-build file)
        continue
    fi

    # ??
    # Note that Brew doesn't want to be ran in `sudo`.
    brew install --force python@$ver
    brew unlink python@$ver
    brew link --overwrite python@$ver
done
