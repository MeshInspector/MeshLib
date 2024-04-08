#!/bin/bash

#This script builds project by CMakeLists.txt
#Assumption: thirdparty packages are already installed by `apt` or built into `./lib`

dt=$(date '+%d-%m-%Y_%H:%M:%S');
logfile="`pwd`/build_source_${dt}.log"
printf "Project build script started.\nYou could find output in ${logfile}\n"

if [[ $OSTYPE == 'darwin'* ]]; then
  PYTHON_VERSION="3.10"
  if [ "${MESHLIB_PYTHON_VERSION}" != "" ]; then
    PYTHON_VERSION="${MESHLIB_PYTHON_VERSION}"
  fi
  PYTHON_PREFIX=$(python"${PYTHON_VERSION}"-config --prefix)
  echo "PYTHON_PREFIX=${PYTHON_PREFIX}"
  PYTHON_EXECUTABLE=$(which python"${PYTHON_VERSION}")
  PYTHON_LIBRARY=${PYTHON_PREFIX}/lib/libpython${PYTHON_VERSION}.dylib
  PYTHON_INCLUDE_DIR=${PYTHON_PREFIX}/include/python${PYTHON_VERSION}
fi

MR_EMSCRIPTEN_SINGLETHREAD=0
if [[ $OSTYPE == "linux"* ]]; then
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
fi

if [ ! -n "$MESHLIB_BUILD_RELEASE" ]; then
  read -t 5 -p "Build MeshLib Release? Press (n) in 5 seconds to cancel (Y/n)" -rsn 1
  echo;
  if [[ $REPLY =~ ^[Nn]$ ]]; then
    MESHLIB_BUILD_RELEASE="OFF"
  else
    MESHLIB_BUILD_RELEASE="ON"
  fi
  printf "Release ${MESHLIB_BUILD_RELEASE}\n"
fi

if [ ! -n "$MESHLIB_BUILD_DEBUG" ]; then
  read -t 5 -p "Build MeshLib Debug? Press (y) in 5 seconds to build (y/N)" -rsn 1
  echo;
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    MESHLIB_BUILD_DEBUG="ON"
  else
    MESHLIB_BUILD_DEBUG="OFF"
  fi
  printf "Debug ${MESHLIB_BUILD_DEBUG}\n"
fi

# build MeshLib
if [ "${MESHLIB_KEEP_BUILD}" != "ON" ]; then
  rm -rf ./build
  mkdir build
  cd build
fi

# add env options to cmake
if [[ -z "${MR_CMAKE_OPTIONS}" ]]; then
  MR_CMAKE_OPTIONS="" # set
else
  MR_CMAKE_OPTIONS="${MR_CMAKE_OPTIONS}"
fi
if [ "${MR_EMSCRIPTEN}" != "ON" ]; then
  if [ -n "${CMAKE_C_COMPILER}" ]; then
    MR_CMAKE_OPTIONS="${MR_CMAKE_OPTIONS} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
  fi
  if [ -n "${CMAKE_CXX_COMPILER}" ]; then
    MR_CMAKE_OPTIONS="${MR_CMAKE_OPTIONS} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
  fi
fi

# macos sometimes does not have nproc
nproc_fn () {
  if [[ $OSTYPE == 'darwin'* ]]; then
    sysctl -n hw.logicalcpu
  else
    nproc
  fi
}

# exit if any command failed
set -eo pipefail

#build Release
if [ "${MESHLIB_BUILD_RELEASE}" = "ON" ]; then
  if [ "${MESHLIB_KEEP_BUILD}" != "ON" ]; then
    mkdir -p Release
  fi
  cd Release
  if [[ $OSTYPE == 'darwin'* ]]; then
    cmake ../.. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DPYTHON_LIBRARY="${PYTHON_LIBRARY}" -DPYTHON_INCLUDE_DIR="${PYTHON_INCLUDE_DIR}" -DPYTHON_EXECUTABLE:FILEPATH="${PYTHON_EXECUTABLE}" ${MR_CMAKE_OPTIONS} | tee ${logfile}
  else
    if [ "${MR_EMSCRIPTEN}" != "ON" ]; then
      cmake ../.. -DCMAKE_BUILD_TYPE=Release ${MR_CMAKE_OPTIONS} | tee ${logfile}
    else
      emcmake cmake ../.. -DMR_EMSCRIPTEN=1 -DMR_EMSCRIPTEN_SINGLETHREAD=${MR_EMSCRIPTEN_SINGLETHREAD} -DCMAKE_BUILD_TYPE=Release ${MR_CMAKE_OPTIONS} | tee ${logfile}
    fi
  fi 
  if [ "${MR_EMSCRIPTEN}" != "ON" ]; then
    cmake --build . -j `nproc_fn` | tee ${logfile}
  else
    emmake make -j `nproc_fn` | tee ${logfile}
  fi
  cd ..
fi

#build Debug
if [ "${MESHLIB_BUILD_DEBUG}" = "ON" ]; then
  if [ "${MESHLIB_KEEP_BUILD}" != "ON" ]; then
    mkdir Debug
  fi
  cd Debug
  if [[ $OSTYPE == 'darwin'* ]]; then
    cmake ../.. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DPYTHON_LIBRARY="${PYTHON_LIBRARY}" -DPYTHON_INCLUDE_DIR="${PYTHON_INCLUDE_DIR}" -DPYTHON_EXECUTABLE:FILEPATH="${PYTHON_EXECUTABLE}" ${MR_CMAKE_OPTIONS} | tee ${logfile}
  else
    if [ "${MR_EMSCRIPTEN}" != "ON" ]; then
      cmake ../.. -DCMAKE_BUILD_TYPE=Debug ${MR_CMAKE_OPTIONS} | tee ${logfile}
    else
      emcmake cmake ../.. -DMR_EMSCRIPTEN=1 -DMR_EMSCRIPTEN_SINGLETHREAD=${MR_EMSCRIPTEN_SINGLETHREAD} -DCMAKE_BUILD_TYPE=Debug ${MR_CMAKE_OPTIONS} | tee ${logfile}
    fi
  fi
  if [ "${MR_EMSCRIPTEN}" != "ON" ]; then
    cmake --build . -j `nproc_fn` | tee ${logfile}
  else
    emmake make -j `nproc_fn` | tee ${logfile}
  fi
  cd ..
fi

if [ "${MESHLIB_BUILD_RELEASE}" = "ON" ]; then
  printf "\rAutoinstall script successfully finished. You could run ./build/Release/bin/MRTest next\n\n"
else
  if [ "${MESHLIB_BUILD_DEBUG}" = "ON" ]; then
    printf "\rAutoinstall script successfully finished. You could run ./build/Debug/bin/MRTest next\n\n"
  else
    printf "\rNothing was built\n\n"
  fi
fi
cd ..
