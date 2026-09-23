#!/bin/sh
set -e

if [ -z "${VCPKG_DEFAULT_HOST_TRIPLET}" ] ; then
  if [ -z "${VCPKG_TRIPLET}" ] ; then
    case "$(uname -m)" in
      x86_64|amd64)  arch=x64 ;;
      arm64|aarch64) arch=arm64 ;;
    esac

    case "$(uname -s)" in
      Linux)  os=linux ;;
      Darwin) os=osx ;;
    esac

    VCPKG_TRIPLET="${arch}-${os}-meshlib"
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

# NOTE: realpath is not supported on older macOS versions
BASE_DIR=$( cd "$( dirname "$0" )"/.. ; pwd -P )

vcpkg install \
    --x-manifest-root=${BASE_DIR}/thirdparty/vcpkg \
    --x-install-root=./vcpkg_installed
