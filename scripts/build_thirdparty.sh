#!/bin/bash

# This script builds thirdparty
# Output libraries are stored in `./lib` directory

# exit if any command failed
set -eo pipefail

dt=$(date '+%d-%m-%Y_%H:%M:%S');
logfile="`pwd`/install_thirdparty_${dt}.log"
printf "Thirdparty build script started.\nYou could find output in ${logfile}\n"


if [[ $OSTYPE == 'darwin'* ]]; then
  echo "MacOS"
  FILE_NAME="install_brew_requirements.sh"
else
  source /etc/os-release
  FILE_NAME="install_apt_requirements.sh"

  if [ "${NAME}" == "Fedora Linux" ]; then
   FILE_NAME="install_dnf_requirements.sh"
  else
    . /etc/lsb-release
  fi
  echo "${NAME}" "${DISTRIB_RELEASE}"
fi

MR_EMSCRIPTEN_SINGLETHREAD=0
if [ "${NAME}" == "Ubuntu" ] && [ "${MR_STATE}" != "DOCKER_BUILD" ]; then
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
printf "Emscripten ${MR_EMSCRIPTEN}, singlethread ${MR_EMSCRIPTEN_SINGLETHREAD}\n"

if [ $MR_EMSCRIPTEN == "ON" ]; then
 if [[ $MR_EMSCRIPTEN_SINGLE == "ON" ]]; then
  MR_EMSCRIPTEN_SINGLETHREAD=1
 fi
else
 printf "Check requirements. Running ${FILE_NAME} ...\n"
 ./scripts/$FILE_NAME
fi


MR_THIRDPARTY_DIR="thirdparty/"
MR_THIRDPARTY_BUILD_DIR="thirdparty_build"
MR_THIRDPARTY_LIB_DIR="lib/"
MR_THIRDPARTY_INCLUDE_DIR="include/"

#build Third party
rm -rf "${MR_THIRDPARTY_BUILD_DIR}"
rm -rf "${MR_THIRDPARTY_LIB_DIR}"
rm -rf "${MR_THIRDPARTY_INCLUDE_DIR}"

mkdir -p "${MR_THIRDPARTY_BUILD_DIR}"
mkdir -p "${MR_THIRDPARTY_LIB_DIR}"
mkdir -p "${MR_THIRDPARTY_INCLUDE_DIR}"

# build
echo "Starting build..."
if [ "${MR_EMSCRIPTEN}" == "ON" ]; then
  cd "${MR_THIRDPARTY_BUILD_DIR}"
  emcmake cmake -DMR_EMSCRIPTEN=1 -DMR_EMSCRIPTEN_SINGLETHREAD=${MR_EMSCRIPTEN_SINGLETHREAD} ../${MR_THIRDPARTY_DIR} -DCMAKE_INSTALL_PREFIX=../
  emmake make -j `nproc` #VERBOSE=1
  make install
  cd ..

  cd thirdparty/wasmtbb
  EMCC_DEBUG=0 emmake make  extra_inc=big_iron.inc VERBOSE=1  tbb
  cd ../..
else
  cd "${MR_THIRDPARTY_BUILD_DIR}"
  cmake ../${MR_THIRDPARTY_DIR} -DCMAKE_INSTALL_PREFIX=../
  cmake --build . -j `nproc`  #VERBOSE=1
  cmake --install .
  cd ..
fi

# copy .so libs (some of them are handled by their cmake --install, but some are not)
echo "Copying shared libs.."
cp "${MR_THIRDPARTY_BUILD_DIR}"/*.so "${MR_THIRDPARTY_LIB_DIR}"/

printf "\rThirdparty build script successfully finished. Required libs located in ./lib folder. You could run ./scripts/build_source.sh\n\n"
