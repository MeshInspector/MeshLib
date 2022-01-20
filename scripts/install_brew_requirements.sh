#!/bin/bash

# run from repo root
# NODE: using this script is deprecated! Better install meshrus(-dev).rpm package
# This script installs requirements by `dnf` if not already installed
# `distribution.sh` uses this script as preinstall

ALL_REQUIRED_PACKAGES_INSTALLED=true
MISSED_PACKAGES=""

function checkPackage {
 PKG_OK=$( brew list ${1})
 if [ "${?}" != 0 ]; then
  ALL_REQUIRED_PACKAGES_INSTALLED=false
  MISSED_PACKAGES="${MISSED_PACKAGES} ${1}"
 fi
}
checkPackage automake
checkPackage cmake
checkPackage llvm@11
checkPackage wget
checkPackage zip
checkPackage libzip
checkPackage unzip
checkPackage cpr
checkPackage openvdb
checkPackage openblas
checkPackage tbb
checkPackage ilmbase
checkPackage openexr
checkPackage boost
checkPackage lz4
checkPackage snappy

checkPackage suite-sparse
#https://stackoverflow.com/questions/60942254/i-cant-get-gtkmm-to-compile-on-mac-using-cmake
checkPackage gtkmm3
checkPackage gtk+3
checkPackage openssl@3
checkPackage libzip
checkPackage zlib
checkPackage spdlog
checkPackage tinyxml
checkPackage googletest
checkPackage fmt
checkPackage gdcm
checkPackage eigen
checkPackage tinyxml2
checkPackage glfw
checkPackage c-blosc
checkPackage podofo
checkPackage jsoncpp	
checkPackage llvm
checkPackage libpng
checkPackage pybind11
checkPackage libsigc++
checkPackage jpeg-turbo

if $ALL_REQUIRED_PACKAGES_INSTALLED; then
 printf "\rAll required packages are already installed!                    \n"
 exit 0
fi

printf "\rSome required package(s) are not installed!                     \n"
printf "${MISSED_PACKAGES}\n"

brew install ${MISSED_PACKAGES}

#export PATH=$(brew --prefix openssl)/bin:$PATH in ~/.bash_profile

#check python3 pip
PIP_OK=$(python3 -m pip --vesrion | grep "No module named pip")
if [ "" != "$PIP_OK" ]; then
 printf "no pip for python3 found. installing...\n"
 wget https://bootstrap.pypa.io/get-pip.py
 python3 get-pip.py
 rm get-pip.py
fi

# install requirements for python libs
python3 -m pip install -r python_requirements.txt

exit 0
