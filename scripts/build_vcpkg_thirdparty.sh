#!/bin/sh
set -e

# NOTE: realpath is not supported on older macOS versions
BASE_DIR=$( cd "$( dirname "$0" )"/.. ; pwd -P )

if [ -z "${VCPKG_DEFAULT_HOST_TRIPLET}" ] ; then
  if [ -z "${VCPKG_TRIPLET}" ] ; then
    VCPKG_TRIPLET=$("${BASE_DIR}/scripts/detect_vcpkg_triplet.sh")
  fi
  export VCPKG_DEFAULT_HOST_TRIPLET="${VCPKG_TRIPLET}"
fi

# some vcpkg packages have host dependencies
if command -v brew >/dev/null 2>&1 ; then
  brew install --quiet autoconf autoconf-archive automake libtool
else
  echo "Make sure the following host dependencies are installed:"
  echo "    autoconf autoconf-archive automake libtool"
fi

version_greater() {
  [ "$(printf '%s\n' "$1" "$2" | sort -V | tail -n 1)" != "$2" ]
}

# extract tool information for the current triplet
vcpkg_tools() {
  IFS=- read -r VCPKG_ARCH VCPKG_OS _ <<EOF
${VCPKG_DEFAULT_HOST_TRIPLET}
EOF
  cat "${VCPKG_ROOT}/scripts/vcpkg-tools.json" | \
    sed -e 's/amd64/x64/g' | \
    jq --arg os "${VCPKG_OS}" --arg arch "${VCPKG_ARCH}" \
    '.tools[] | select(.os == $os and (.arch // $arch) == $arch)'
}

# the vcpkg binary cache key includes the CMake version
if command -v cmake >/dev/null 2>&1 ; then
  SYSTEM_CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
  VCPKG_CMAKE_VERSION=$(vcpkg_tools | jq -r 'select(.name == "cmake") | .version')
  if version_greater "${SYSTEM_CMAKE_VERSION}" "${VCPKG_CMAKE_VERSION}" ; then
    if [ -n "${CI}" ] ; then
      export VCPKG_FORCE_DOWNLOADED_BINARIES=1
    else
      echo "Set VCPKG_FORCE_DOWNLOADED_BINARIES=1 to reuse the CI binary cache"
    fi
  fi
fi

vcpkg install \
    --x-manifest-root=${BASE_DIR}/thirdparty/vcpkg \
    --x-install-root=./vcpkg_installed
