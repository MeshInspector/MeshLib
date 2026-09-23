#!/bin/sh
set -e

# NOTE: realpath is not supported on older macOS versions
BASE_DIR=$( cd "$( dirname "$0" )"/.. ; pwd -P )

MESHLIB_USE_VCPKG=ON \
VCPKG_MANIFEST_MODE=ON \
${BASE_DIR}/scripts/build_source.sh "$@"
