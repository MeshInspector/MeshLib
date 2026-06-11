#!/usr/bin/env bash
set -euo pipefail

# Install the freshly-built meshlib wheel into the requested Python
# interpreter and run the wheel's pytest suite. Run from the repo root;
# the wheel artifact must already be unpacked in the working directory.

USAGE="usage: test_wheel_macos.sh <py-version> <platform: arm64|x86> <runner-tier: github|self-hosted>"
PY_VER="${1:?${USAGE}}"
PLATFORM="${2:?${USAGE}}"
RUNNER_TIER="${3:?${USAGE}}"

# Remove from PATH anything with the word `anaconda` in it.
# Even if Anaconda's Python works, it's probably a good idea to avoid it
# for consistency (without this, only one specific Python version may be
# picked up from Anaconda).
export PATH="$(echo "$PATH" | perl -pe 's/[^:]*anaconda[^:]*//g;s/::|^:|:$//g')"

WHEEL=$(ls ./meshlib-*"${PLATFORM}"*.whl | head -n 1)

# Python 3.11 ships pre-installed on the GitHub macos-15-intel image;
# `brew install python@3.11` trips its post-install there. Skip the brew
# step only on that specific runner -- the self-hosted x86 host
# (macos-x64-build) does need the install. `--quiet` is intentionally
# omitted so brew's post-install stderr streams to the workflow log; the
# generic "post-install step did not complete" summary it prints
# otherwise hides the real error.
if [ "${RUNNER_TIER}" != "github" ] || [ "${PLATFORM}" != "x86" ] || [ "${PY_VER}" != "3.11" ]; then
  brew install --overwrite "python@${PY_VER}"
fi

PY_CMD="python${PY_VER}"
VENV=".venv-${PY_VER}"
rm -rf "${VENV}"
"${PY_CMD}" -m venv "${VENV}"
# shellcheck disable=SC1091
. "${VENV}/bin/activate"

python -m pip install --upgrade pip
python -m pip install --upgrade -r ./requirements/python/requirements.txt
python -m pip install pytest
python -m pip install "${WHEEL}"

( cd test_python && python -m pytest -s -v )
