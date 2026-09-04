#!/usr/bin/env bash
set -euo pipefail

# Install the freshly-built meshlib wheel into the requested Python
# interpreter and run the wheel's pytest suite. Run from the repo root;
# the wheel artifact must already be unpacked in the working directory.

USAGE="usage: test_wheel_macos.sh <py-version> <platform: arm64|x86>"
PY_VER="${1:?${USAGE}}"
PLATFORM="${2:?${USAGE}}"

# Remove from PATH anything with the word `anaconda` in it.
# Even if Anaconda's Python works, it's probably a good idea to avoid it
# for consistency (without this, only one specific Python version may be
# picked up from Anaconda).
export PATH="$(echo "$PATH" | perl -pe 's/[^:]*anaconda[^:]*//g;s/::|^:|:$//g')"

WHEELS=$(ls ./meshlib*"${PLATFORM}"*.whl)

VENV=".venv-${PY_VER}"
rm -rf "${VENV}"
uv venv --python "${PY_VER}" "${VENV}"
# shellcheck disable=SC1091
. "${VENV}/bin/activate"

uv pip install \
    -r ./requirements/python/requirements.txt \
    pytest \
    ${WHEELS}

( cd test_python && python -m pytest -s -v )
