#!/bin/bash

# This script builds thirdparty
# Output libraries are stored in `./lib` directory

# exit if any command failed
set -eo pipefail

dt=$(date '+%d-%m-%Y_%H:%M:%S');
logfile="`pwd`/install_thirdparty_${dt}.log"
printf "Thirdparty build script started.\nYou could find output in ${logfile}\n"

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
  elif [[ "${ID}" == "fedora" ]] || [[ "${ID_LIKE}" == *"fedora"* ]]; then
    INSTALL_REQUIREMENTS="install_dnf_requirements.sh"
  fi
else
  echo "Host system: ${OSTYPE}"
fi

MR_EMSCRIPTEN_SINGLETHREAD=0
if [[ $OSTYPE == "linux"* ]] && [ "${MR_STATE}" != "DOCKER_BUILD" ]; then
  if [ ! -n "$MR_EMSCRIPTEN" ]; then
    read -t 5 -p "Build with emscripten? Press (y) in 5 seconds to build (y/s/N) (s - singlethreaded)" -rsn 1
    echo;
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      MR_EMSCRIPTEN="ON"
    else
      if [[ $REPLY =~ ^[Ss]$ ]]; then
        MR_EMSCRIPTEN="ON"
        MR_EMSCRIPTEN_SINGLETHREAD=1
      else
        MR_EMSCRIPTEN="OFF"
      fi
    fi
  fi  
else
  if [ ! -n "$MR_EMSCRIPTEN" ]; then
    MR_EMSCRIPTEN="OFF"
  fi
fi
echo "Emscripten ${MR_EMSCRIPTEN}, singlethread ${MR_EMSCRIPTEN_SINGLETHREAD}"

if [ $MR_EMSCRIPTEN == "ON" ]; then
  if [[ $MR_EMSCRIPTEN_SINGLE == "ON" ]]; then
    MR_EMSCRIPTEN_SINGLETHREAD=1
  fi
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

MR_CMAKE_OPTIONS=""
if [ "${MR_EMSCRIPTEN}" == "ON" ]; then
  if [ -z "${EMSDK}" ] ; then
    echo "Emscripten SDK not found"
    exit 1
  fi
  EMSCRIPTEN_ROOT="${EMSDK}/upstream/emscripten"

  MR_CMAKE_OPTIONS="${MR_CMAKE_OPTIONS} \
    -D CMAKE_FIND_ROOT_PATH=${MESHLIB_THIRDPARTY_ROOT_DIR} \
    -D CMAKE_INSTALL_PREFIX=${MESHLIB_THIRDPARTY_ROOT_DIR} \
    -D CMAKE_TOOLCHAIN_FILE=${EMSCRIPTEN_ROOT}/cmake/Modules/Platform/Emscripten.cmake \
  "
  if [[ ${MR_EMSCRIPTEN_SINGLETHREAD} == 0 ]] ; then
    MR_CMAKE_OPTIONS="${MR_CMAKE_OPTIONS} \
      -D CMAKE_C_FLAGS=-pthread \
      -D CMAKE_CXX_FLAGS=-pthread \
    "
  fi
fi

# build
echo "Starting build..."
pushd "${MESHLIB_THIRDPARTY_BUILD_DIR}"
if [ "${MR_EMSCRIPTEN}" == "ON" ]; then
  # build libjpeg-turbo separately
  CMAKE_OPTIONS="${MR_CMAKE_OPTIONS}" ${SCRIPT_DIR}/thirdparty/libjpeg-turbo.sh ${MESHLIB_THIRDPARTY_DIR}/libjpeg-turbo

  emcmake cmake -DMR_EMSCRIPTEN=1 -DMR_EMSCRIPTEN_SINGLETHREAD=${MR_EMSCRIPTEN_SINGLETHREAD} ${MESHLIB_THIRDPARTY_DIR} -DCMAKE_INSTALL_PREFIX=${MESHLIB_THIRDPARTY_ROOT_DIR}
  emmake make -j `nproc` #VERBOSE=1
  make install

  # build OpenVDB separately
  CMAKE_OPTIONS="${MR_CMAKE_OPTIONS}" ${SCRIPT_DIR}/thirdparty/openvdb.sh ${MESHLIB_THIRDPARTY_DIR}/openvdb/v10/openvdb
else
  cmake ${MESHLIB_THIRDPARTY_DIR} -DCMAKE_INSTALL_PREFIX=${MESHLIB_THIRDPARTY_ROOT_DIR} -DCMAKE_BUILD_TYPE=Release
  cmake --build . -j `nproc`  #VERBOSE=1
  cmake --install .
fi
popd

# copy libs (some of them are handled by their `cmake --install`, but some are not)
echo "Copying thirdparty libs.."
if [[ $OSTYPE == 'darwin'* ]]; then
  LIB_SUFFIX="*.dylib"
elif [ "${MR_EMSCRIPTEN}" = "ON" ]; then
  LIB_SUFFIX="*.a"
else
  LIB_SUFFIX="*.so"
fi
cp "${MESHLIB_THIRDPARTY_BUILD_DIR}"/${LIB_SUFFIX} "${MESHLIB_THIRDPARTY_ROOT_DIR}/lib/"

printf "\rThirdparty build script successfully finished. Required libs located in ./lib folder. You could run ./scripts/build_source.sh\n\n"
