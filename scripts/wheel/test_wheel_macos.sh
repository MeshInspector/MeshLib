#!/usr/bin/env bash
set -euo pipefail

# Install the freshly-built meshlib wheel into the requested Python
# interpreter and run the wheel's pytest suite. Run from the repo root;
# the wheel artifact must already be unpacked in the working directory.

PY_VER="${1:?usage: test_wheel_macos.sh <py-version> <platform: arm64|x86>}"
PLATFORM="${2:?usage: test_wheel_macos.sh <py-version> <platform: arm64|x86>}"

# Remove from PATH anything with the word `anaconda` in it.
# Even if Anaconda's Python works, it's probably a good idea to avoid it
# for consistency (without this, only one specific Python version may be
# picked up from Anaconda).
export PATH="$(echo "$PATH" | perl -pe 's/[^:]*anaconda[^:]*//g;s/::|^:|:$//g')"

WHEEL=$(ls ./meshlib-*"${PLATFORM}"*.whl | head -n 1)

# Python 3.11 ships pre-installed on macos-15-intel; brew install fails there.
if [ "${PLATFORM}" != "x86" ] || [ "${PY_VER}" != "3.11" ]; then
  brew install --overwrite --quiet "python@${PY_VER}"
fi

PY_CMD="python${PY_VER}"
VENV=".venv-${PY_VER}"
rm -rf "${VENV}"
"${PY_CMD}" -m venv "${VENV}"
# shellcheck disable=SC1091
. "${VENV}/bin/activate"

python -m pip install --upgrade pip
python -m pip install --upgrade -r ./requirements/python.txt
python -m pip install pytest
python -m pip install "${WHEEL}"

( cd test_python && python -m pytest -s -v )
