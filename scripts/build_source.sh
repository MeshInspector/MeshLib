#!/bin/bash

#This script builds project by CMakeLists.txt
#Assumption: thirdparty packages are already installed by `apt` or built into `./lib`

dt=$(date '+%d-%m-%Y_%H:%M:%S');
logfile="`pwd`/build_source_${dt}.log"
printf "Project build script started.\nYou could find output in ${logfile}\n"

if [[ $OSTYPE != 'darwin'* ]]; then
  source /etc/os-release
  printf "${NAME} ${VERSION_ID}\n"
  if [ "${NAME}" == "Fedora Linux" ]; then
   if [ "${CMAKE_C_COMPILER}" = "" ]; then
    CMAKE_C_COMPILER=/usr/bin/gcc
   fi
   if [ "${CMAKE_CXX_COMPILER}" = "" ]; then
    CMAKE_CXX_COMPILER=/usr/bin/g++
   fi
  elif [ "${NAME}" == "Ubuntu" ] && [ "${VERSION_ID}" != "20.04" ]; then
    if [ "${CMAKE_C_COMPILER}" = "" ]; then
        CMAKE_C_COMPILER=/usr/bin/gcc
    fi
    if [ "${CMAKE_CXX_COMPILER}" = "" ]; then
      CMAKE_CXX_COMPILER=/usr/bin/g++
    fi
  else
   if [ "${CMAKE_C_COMPILER}" = "" ]; then
    CMAKE_C_COMPILER=/usr/bin/gcc-10
   fi
   if [ "${CMAKE_CXX_COMPILER}" = "" ]; then
    CMAKE_CXX_COMPILER=/usr/bin/g++-10
   fi
  fi
fi

if [ "${NAME}" == "Ubuntu" ]; then
 if [ ! -n "$MR_EMSCRIPTEN" ]; then
  read -t 5 -p "Build with emscripten? Press (y) in 5 seconds to build (y/N)" -rsn 1
  echo;
  if [[ $REPLY =~ ^[Yy]$ ]]; then
   MR_EMSCRIPTEN="ON"
  else
   MR_EMSCRIPTEN="OFF"
  fi
  printf "Emscripten ${MR_EMSCRIPTEN}\n"
 fi  
fi

if [ ! -n "$MESHRUS_BUILD_RELEASE" ]; then
 read -t 5 -p "Build MeshLib Release? Press (n) in 5 seconds to cancel (Y/n)" -rsn 1
 echo;
 if [[ $REPLY =~ ^[Nn]$ ]]; then
  MESHRUS_BUILD_RELEASE="OFF"
 else
  MESHRUS_BUILD_RELEASE="ON"
 fi
 printf "Release ${MESHRUS_BUILD_RELEASE}\n"
fi

if [ ! -n "$MESHRUS_BUILD_DEBUG" ]; then
 read -t 5 -p "Build MeshLib Debug? Press (y) in 5 seconds to build (y/N)" -rsn 1
 echo;
 if [[ $REPLY =~ ^[Yy]$ ]]; then
  MESHRUS_BUILD_DEBUG="ON"
 else
  MESHRUS_BUILD_DEBUG="OFF"
 fi
  printf "Debug ${MESHRUS_BUILD_DEBUG}\n"
fi

# build MeshLib
if [ "${MESHRUS_KEEP_BUILD}" != "ON" ]; then
 rm -rf ./build
 mkdir build
 cd build
fi

# exit if any command failed
set -eo pipefail

#build Release
if [ "${MESHRUS_BUILD_RELEASE}" = "ON" ]; then
 if [ "${MESHRUS_KEEP_BUILD}" != "ON" ]; then
  mkdir -p Release
 fi
 cd Release
 if [[ $OSTYPE == 'darwin'* ]]; then
    cmake ../.. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DPYTHON_LIBRARY=$(python3-config --prefix)/lib/libpython3.9.dylib -DPYTHON_INCLUDE_DIR=$(python3-config --prefix)/include/python3.9 | tee ${logfile}
 else
    if [ "${MR_EMSCRIPTEN}" != "ON" ]; then
      cmake ../.. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} | tee ${logfile}
    else
      emcmake cmake ../.. -DMR_EMSCRIPTEN=1 -DCMAKE_BUILD_TYPE=Release | tee ${logfile}
    fi
 fi 
 if [ "${MR_EMSCRIPTEN}" != "ON" ]; then
    cmake --build . -j `nproc` | tee ${logfile}
 else
    emmake make -j `nproc` | tee ${logfile}
 fi
 cd ..
fi

#build Debug
if [ "${MESHRUS_BUILD_DEBUG}" = "ON" ]; then
 if [ "${MESHRUS_KEEP_BUILD}" != "ON" ]; then
  mkdir Debug
 fi
 cd Debug
 if [[ $OSTYPE == 'darwin'* ]]; then
    cmake ../.. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DPYTHON_LIBRARY=$(python3-config --prefix)/lib/libpython3.9.dylib -DPYTHON_INCLUDE_DIR=$(python3-config --prefix)/include/python3.9 | tee ${logfile}
 else
    if [ "${MR_EMSCRIPTEN}" != "ON" ]; then
      cmake ../.. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} | tee ${logfile}
    else
      emcmake cmake ../.. -DMR_EMSCRIPTEN=1 -DCMAKE_BUILD_TYPE=Debug | tee ${logfile}
    fi
 fi
 if [ "${MR_EMSCRIPTEN}" != "ON" ]; then
    cmake --build . -j `nproc` | tee ${logfile}
 else
    emmake make -j `nproc` | tee ${logfile}
 fi
 cd ..
fi

if [ "${MESHRUS_BUILD_RELEASE}" = "ON" ]; then
 printf "\rAutoinstall script successfully finished. You could run ./build/Release/bin/MRTest next\n\n"
else
 if [ "${MESHRUS_BUILD_DEBUG}" = "ON" ]; then
  printf "\rAutoinstall script successfully finished. You could run ./build/Debug/bin/MRTest next\n\n"
 else
  printf "\rNothing was built\n\n"
 fi
fi
cd ..
