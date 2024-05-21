#!/bin/bash

# This script builds thirdparty
# Output libraries are stored in `./lib` directory

# exit if any command failed
set -eo pipefail

dt=$(date '+%d-%m-%Y_%H:%M:%S');
logfile="`pwd`/install_thirdparty_${dt}.log"
printf "Thirdparty build script started.\nYou could find output in ${logfile}\n"

SCRIPT_DIR=$(dirname $(realpath "$0"))
BASE_DIR=$(realpath "${SCRIPT_DIR}/..")

THIRDPARTY_DIR=${BASE_DIR}/thirdparty/
THIRDPARTY_BUILD_DIR="${THIRDPARTY_BUILD_DIR:-${BASE_DIR}/thirdparty_build/}"
THIRDPARTY_ROOT_DIR="${THIRDPARTY_ROOT_DIR:-${BASE_DIR}/thirdparty_root/}"

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
rm -rf "${THIRDPARTY_BUILD_DIR}"
mkdir -p "${THIRDPARTY_BUILD_DIR}"
# FIXME: make it optional
rm -rf "${THIRDPARTY_ROOT_DIR}"
mkdir -p "${THIRDPARTY_ROOT_DIR}"

# build
echo "Starting build..."
pushd "${THIRDPARTY_BUILD_DIR}"
if [ "${MR_EMSCRIPTEN}" == "ON" ]; then
  emcmake cmake -DMR_EMSCRIPTEN=1 -DMR_EMSCRIPTEN_SINGLETHREAD=${MR_EMSCRIPTEN_SINGLETHREAD} ${THIRDPARTY_DIR} -DCMAKE_INSTALL_PREFIX=${THIRDPARTY_ROOT_DIR}
  emmake make -j `nproc` #VERBOSE=1
  make install

  # build OpenVDB separately
  OPENVDB_CMAKE_OPTIONS="-DOPENVDB_ENABLE_UNINSTALL=OFF -DOPENVDB_ENABLE_INSTALL=OFF -DOPENVDB_CORE_SHARED=OFF -DOPENVDB_CORE_STATIC=ON -DOPENVDB_BUILD_BINARIES=OFF -DOPENVDB_BUILD_VDB_PRINT=OFF -DUSE_BLOSC=OFF -DUSE_EXPLICIT_INSTANTIATION=OFF"
  # fix TBB lookup
  OPENVDB_CMAKE_OPTIONS="${OPENVDB_CMAKE_OPTIONS} -DCMAKE_FIND_ROOT_PATH=${THIRDPARTY_ROOT_DIR} -DTbb_VERSION=2021.12"
  # fix linkage
  if [[ ${MR_EMSCRIPTEN_SINGLETHREAD} == 0 ]] ; then
    OPENVDB_CMAKE_OPTIONS="${OPENVDB_CMAKE_OPTIONS} -DCMAKE_CXX_FLAGS=-pthread"
  fi
  # mock the missing dependencies that are not actually used
  # if you want to use the OpenVDB file functions (see MRMESH_OPENVDB_USE_IO), you'll have to build these libraries by yourself
  touch ${THIRDPARTY_ROOT_DIR}/lib/libboost_iostreams.a ${THIRDPARTY_ROOT_DIR}/lib/libboost_regex.a
  emcmake cmake -S ${THIRDPARTY_DIR}/openvdb/v10/openvdb/ -B openvdb_build -DCMAKE_INSTALL_PREFIX="${THIRDPARTY_ROOT_DIR}" ${OPENVDB_CMAKE_OPTIONS}
  emmake cmake --build openvdb_build -j `nproc`
  emmake cmake --install openvdb_build
else
  cmake ${THIRDPARTY_DIR} -DCMAKE_INSTALL_PREFIX=${THIRDPARTY_ROOT_DIR} -DCMAKE_BUILD_TYPE=Release
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
cp "${THIRDPARTY_BUILD_DIR}"/${LIB_SUFFIX} "${THIRDPARTY_ROOT_DIR}/lib/"

printf "\rThirdparty build script successfully finished. Required libs located in ./lib folder. You could run ./scripts/build_source.sh\n\n"
