#!/bin/sh
set -e

# NOTE: realpath is not supported on older macOS versions
BASE_DIR=$( cd "$( dirname "$0" )"/.. ; pwd -P )

if [ -z "${VCPKG_DEFAULT_HOST_TRIPLET}" ] ; then
  if [ -z "${VCPKG_TRIPLET}" ] ; then
    VCPKG_TRIPLET=$("${BASE_DIR}/scripts/detect_vcpkg_triplet.sh")
  fi
  VCPKG_DEFAULT_HOST_TRIPLET="${VCPKG_TRIPLET}"
fi

# some vcpkg packages have host dependencies
if command -v brew >/dev/null 2>&1 ; then
  brew install --quiet autoconf autoconf-archive automake libtool
else
  echo "Make sure the following host dependencies are installed:"
  echo "    autoconf autoconf-archive automake libtool"
fi

vcpkg install \
    --x-manifest-root=${BASE_DIR}/thirdparty/vcpkg \
    --x-install-root=${BASE_DIR}/vcpkg_installed
