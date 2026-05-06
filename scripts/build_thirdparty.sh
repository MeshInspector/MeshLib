#!/bin/bash

# This script builds thirdparty
# Output libraries are stored in `./lib` directory

# exit if any command failed
set -eo pipefail

dt=$(date '+%d-%m-%Y_%H:%M:%S');
logfile="`pwd`/install_thirdparty_${dt}.log"
echo "Thirdparty build script started."
echo "You could find output in ${logfile}"

# NOTE: realpath is not supported on older macOS versions
BASE_DIR=$( cd "$( dirname "$0" )"/.. ; pwd -P )
SCRIPT_DIR=${BASE_DIR}/scripts/

MESHLIB_THIRDPARTY_DIR=${BASE_DIR}/thirdparty/
MESHLIB_THIRDPARTY_BUILD_DIR="${MESHLIB_THIRDPARTY_BUILD_DIR:-${BASE_DIR}/thirdparty_build/}"
MESHLIB_THIRDPARTY_ROOT_DIR="${MESHLIB_THIRDPARTY_ROOT_DIR:-${BASE_DIR}}"

if [[ $OSTYPE == 'darwin'* ]]; then
  echo "Host system: MacOS"
  INSTALL_REQUIREMENTS="install_brew_requirements.sh"
elif [[ $OSTYPE == 'linux'* ]]; then
  source /etc/os-release
  echo "Host system: ${NAME} ${DISTRIB_RELEASE}"
  if [[ "${ID}" == "ubuntu" ]] || [[ "${ID_LIKE}" == *"ubuntu"* ]]; then
    INSTALL_REQUIREMENTS="install_apt_requirements.sh"
  fi
else
  echo "Host system: ${OSTYPE}"
fi

. "$SCRIPT_DIR/ask_emscripten_mode.src"

if [ $MR_EMSCRIPTEN == "ON" ]; then
  true # Nothing.
elif [ -n "${INSTALL_REQUIREMENTS}" ]; then
  echo "Check requirements. Running ${INSTALL_REQUIREMENTS} ..."
  ${SCRIPT_DIR}/$INSTALL_REQUIREMENTS
else
  echo "Unsupported system. Installing dependencies is your responsibility."
fi

# FIXME: make it optional
rm -rf "${MESHLIB_THIRDPARTY_BUILD_DIR}"
mkdir -p "${MESHLIB_THIRDPARTY_BUILD_DIR}"
# FIXME: make it optional
for SUBDIR in lib include ; do
  rm -rf "${MESHLIB_THIRDPARTY_ROOT_DIR}"/${SUBDIR}
  mkdir -p "${MESHLIB_THIRDPARTY_ROOT_DIR}"/${SUBDIR}
done

MR_CMAKE_OPTIONS="\
  -D CMAKE_INSTALL_PREFIX=${MESHLIB_THIRDPARTY_ROOT_DIR} \
  -D CMAKE_BUILD_TYPE=Release \
"

if [ "${MR_EMSCRIPTEN}" != "ON" ] ; then
  CMAKE_C_COMPILER="${CMAKE_C_COMPILER:-${CC}}"
  if [ -n "${CMAKE_C_COMPILER}" ] ; then
    MR_CMAKE_OPTIONS="${MR_CMAKE_OPTIONS} -D CMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
  fi

  CMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER:-${CXX}}"
  if [ -n "${CMAKE_CXX_COMPILER}" ] ; then
    MR_CMAKE_OPTIONS="${MR_CMAKE_OPTIONS} -D CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"

    # special conditions for Clang on macOS
    if [[ $OSTYPE == 'darwin'* ]] && command -v brew >/dev/null 2>&1; then
      HOMEBREW_PREFIX=$(brew --prefix)
      if [[ "${CMAKE_CXX_COMPILER}" == "${HOMEBREW_PREFIX}"* ]] ; then
        # use system libc++ instead of Clang's one
        MACOS_SDK_PATH=$(xcrun --show-sdk-path | xargs)  # trim trailing whitespace
        CXXFLAGS="-nostdinc++ -isystem ${MACOS_SDK_PATH}/usr/include/c++/v1 -isysroot ${MACOS_SDK_PATH}"
        LDFLAGS="-nostdlib++ -L${MACOS_SDK_PATH}/usr/lib -lc++ -lc++abi"
        # use Homebrew zlib instead of system one
        MR_CMAKE_OPTIONS="${MR_CMAKE_OPTIONS} -D ZLIB_ROOT=$(brew --prefix zlib)"
      fi
    fi
  fi
fi

if command -v ninja >/dev/null 2>&1 ; then
  MR_CMAKE_OPTIONS="${MR_CMAKE_OPTIONS} -G Ninja"
fi

if [ "${MR_EMSCRIPTEN}" == "ON" ]; then
  if [ -z "${EMSDK}" ] ; then
    echo "Emscripten SDK not found"
    exit 1
  fi
  EMSCRIPTEN_ROOT="${EMSDK}/upstream/emscripten"

  export CFLAGS=""
  export CXXFLAGS=""
  export LDFLAGS=""
  MR_CMAKE_OPTIONS="${MR_CMAKE_OPTIONS} \
    -D CMAKE_TOOLCHAIN_FILE=${EMSCRIPTEN_ROOT}/cmake/Modules/Platform/Emscripten.cmake \
    -D CMAKE_FIND_ROOT_PATH=${MESHLIB_THIRDPARTY_ROOT_DIR} \
    -D MR_EMSCRIPTEN=1 \
    -D MR_EMSCRIPTEN_SINGLETHREAD=${MR_EMSCRIPTEN_SINGLETHREAD} \
    -D MR_EMSCRIPTEN_WASM64=${MR_EMSCRIPTEN_WASM64} \
  "
  if [[ ${MR_EMSCRIPTEN_SINGLETHREAD} == 0 ]] ; then
    CFLAGS="${CFLAGS} -pthread"
    CXXFLAGS="${CFLAGS} -pthread"
  fi
  if [[ ${MR_EMSCRIPTEN_WASM64} == 1 ]] ; then
    CFLAGS="${CFLAGS} -s MEMORY64=1"
    CXXFLAGS="${CFLAGS} -s MEMORY64=1"
    LDFLAGS="${LDFLAGS} -s MEMORY64=1"
  fi
fi

if [[ $OSTYPE == 'darwin'* ]]; then
  NPROC=$(sysctl -n hw.logicalcpu)
else
  NPROC=$(nproc)
fi

# build
echo "Starting build..."
pushd "${MESHLIB_THIRDPARTY_BUILD_DIR}"
if [ "${MR_EMSCRIPTEN}" == "ON" ]; then
  # build Boost libraries separately
  # TODO: build Boost.Locale as a standalone library
  ${SCRIPT_DIR}/thirdparty/boost-libs-download.sh ${MESHLIB_THIRDPARTY_DIR}/boost-libs
  CMAKE_OPTIONS="${MR_CMAKE_OPTIONS}" ${SCRIPT_DIR}/thirdparty/boost-libs.sh ${MESHLIB_THIRDPARTY_DIR}/boost-libs
  # remove excess header files as they're distributed by Emscripten
  find ${MESHLIB_THIRDPARTY_ROOT_DIR}/include/boost -mindepth 1 -maxdepth 1 -not -name 'locale*' -exec rm -r "{}" \;

  # build libjpeg-turbo separately
  CMAKE_OPTIONS="${MR_CMAKE_OPTIONS}" ${SCRIPT_DIR}/thirdparty/libjpeg-turbo.sh ${MESHLIB_THIRDPARTY_DIR}/libjpeg-turbo
  # build MbedTLS separately
  CMAKE_OPTIONS="${MR_CMAKE_OPTIONS}" ${SCRIPT_DIR}/thirdparty/mbedtls.sh ${MESHLIB_THIRDPARTY_DIR}/mbedtls

  cmake -S ${MESHLIB_THIRDPARTY_DIR} -B . ${MR_CMAKE_OPTIONS}
  cmake --build . -j ${NPROC}
  cmake --install .

  # build Eigen separately
  CMAKE_OPTIONS="${MR_CMAKE_OPTIONS}" ${SCRIPT_DIR}/thirdparty/eigen.sh ${MESHLIB_THIRDPARTY_DIR}/eigen
  # build libE57Format separately
  CMAKE_OPTIONS="${MR_CMAKE_OPTIONS}" ${SCRIPT_DIR}/thirdparty/libE57Format.sh ${MESHLIB_THIRDPARTY_DIR}/libE57Format
  # build OpenVDB separately
  CMAKE_OPTIONS="${MR_CMAKE_OPTIONS}" ${SCRIPT_DIR}/thirdparty/openvdb.sh ${MESHLIB_THIRDPARTY_DIR}/openvdb/v10/openvdb
else
  cmake -S ${MESHLIB_THIRDPARTY_DIR} -B . ${MR_CMAKE_OPTIONS}
  cmake --build . -j ${NPROC}
  cmake --install .

  # build clip separately
  CMAKE_OPTIONS="${MR_CMAKE_OPTIONS}" ${SCRIPT_DIR}/thirdparty/clip.sh ${MESHLIB_THIRDPARTY_DIR}/clip
  # build fastmcpp separately
  CMAKE_OPTIONS="${MR_CMAKE_OPTIONS}" ${SCRIPT_DIR}/thirdparty/fastmcpp.sh ${MESHLIB_THIRDPARTY_DIR}/fastmcpp
fi
popd

printf "\rThirdparty build script successfully finished. Required libs located in ./lib folder. You could run ./scripts/build_source.sh\n\n"
