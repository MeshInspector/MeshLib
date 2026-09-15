#!/bin/bash

# This script downloads thirdparty sources with CPM from `thirdparty/cpm` without building them
# Usage: ./scripts/fetch_cpm_thirdparty.sh [PACKAGE...]
# Fetches the given packages (all of thirdparty/cpm/package-lock.cmake by default) into CPM_SOURCE_CACHE
# and prints one `SRC_<package>=<path>` line per package.

set -eo pipefail

# NOTE: realpath is not supported on older macOS versions
BASE_DIR=$( cd "$( dirname "$0" )"/.. ; pwd -P )

MESHLIB_THIRDPARTY_DIR=${BASE_DIR}/thirdparty/cpm/
MESHLIB_THIRDPARTY_FETCH_DIR=${BASE_DIR}/build/thirdparty_fetch/
MESHLIB_FETCH_OUTPUT=${MESHLIB_THIRDPARTY_FETCH_DIR}/thirdparty_sources.sh

export CPM_SOURCE_CACHE="${CPM_SOURCE_CACHE:-${BASE_DIR}/thirdparty_sources}"

if [ $# -gt 0 ] ; then
  PACKAGES=$( IFS=';' ; echo "$*" )
else
  PACKAGES=$( sed -n 's/^set(MESHLIB_PACKAGE_\(.*\)$/\1/p' "${MESHLIB_THIRDPARTY_DIR}/package-lock.cmake" | paste -sd ';' )
fi

cmake -S "${MESHLIB_THIRDPARTY_DIR}/fetch" -B "${MESHLIB_THIRDPARTY_FETCH_DIR}" \
  -D CPM_SOURCE_CACHE="${CPM_SOURCE_CACHE}" \
  -D MESHLIB_FETCH_PACKAGES="${PACKAGES}" \
  -D MESHLIB_FETCH_OUTPUT="${MESHLIB_FETCH_OUTPUT}" \
  1>&2
cat "${MESHLIB_FETCH_OUTPUT}"
